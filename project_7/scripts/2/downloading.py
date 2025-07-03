import os
import re
import random
import asyncio
import pandas as pd
import aiohttp
from io import StringIO
from tqdm import tqdm

ORGANISMS = {'E._coli': 83333, 'B._subtilis': 224308, 'C._elegans': 6239,'Human': 9606, 'Yeast': 559292,
            'A._thaliana': 3702, 'D._melanogaster': 7227, 'Mouse': 10090, 'Zebrafish': 7955}
TAXONOMIES = {
    "Bacteria": 2,
    "Viruses": 10239,
    "Archaea": 2157,
    "Eukaryota": 2759
}
DATA_DIR = "../../data"
RAW_DIR = f"{DATA_DIR}/raw/reference_proteomes"

async def download_file(url, output_path, session=None):
    if os.path.exists(output_path):
        return True
        
    close_session = False
    if not session:
        session = aiohttp.ClientSession()
        close_session = True
        
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(output_path, 'wb') as f:
                    f.write(await response.read())
                return True
    finally:
        if close_session:
            await session.close()
    return False

async def download_organisms():
    results = {}
    organisms_list = list(ORGANISMS.items())
    
    async with aiohttp.ClientSession() as session:
        for organism, taxon_id in tqdm(organisms_list, desc="Downloading Organisms",
                                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
            file_path = f"{RAW_DIR}/{organism}.fasta"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if not os.path.exists(file_path):
                async with session.get(f"https://rest.uniprot.org/proteomes/stream?query=(taxonomy_id:{taxon_id})&format=tsv") as response:
                    if response.status == 200:
                        text = await response.text()
                        proteome_data = pd.read_csv(StringIO(text), sep='\t')
                        if not proteome_data.empty:
                            proteome_id = proteome_data['Proteome Id'].iloc[0]
                            await download_file(f"https://rest.uniprot.org/uniprotkb/stream?query=proteome:{proteome_id}&format=fasta",
                                              file_path, session)
    
    return [f"{RAW_DIR}/{organism}.fasta" for organism in ORGANISMS.keys()]

async def download_pdb():
    pdb_file = f"{RAW_DIR}/pdb_seqres.txt"
    os.makedirs(os.path.dirname(pdb_file), exist_ok=True)
    
    if not os.path.exists(pdb_file):
        print("Downloading PDB sequences...")
        await download_file("https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt", pdb_file)
    
    return pdb_file

async def download_swissprot():
    swissprot_gz = f"{RAW_DIR}/swissprot_uniprot.fasta.gz"
    swissprot_file = f"{RAW_DIR}/swissprot_uniprot.fasta"
    os.makedirs(os.path.dirname(swissprot_file), exist_ok=True)
    
    if not os.path.exists(swissprot_file):
        print("Downloading SwissProt data...")
        if not os.path.exists(swissprot_gz):
            await download_file("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz", swissprot_gz)
            
        print("Extracting SwissProt data...")
        import gzip
        with gzip.open(swissprot_gz, 'rb') as f_in:
            with open(swissprot_file, 'wb') as f_out:
                f_out.write(f_in.read())
    
    return swissprot_file

async def download_taxonomies():
    files_by_taxonomy = {}
    
    async with aiohttp.ClientSession() as session:
        for taxonomy_name, taxonomy_id in TAXONOMIES.items():
            tax_dir = f"{RAW_DIR}/{taxonomy_name}"
            os.makedirs(tax_dir, exist_ok=True)
            
            files = [f.path for f in os.scandir(tax_dir) if f.is_file() and f.name.endswith(".fasta")]
            
            if len(files) < 100:
                try:
                    api_url = f"https://rest.uniprot.org/proteomes/search?query=taxonomy_id:{taxonomy_id}+AND+reference:true&format=json&size=500"
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            async with session.get(api_url) as response:
                                if response.status == 200:
                                    proteomes_data = await response.json()
                                    if "results" in proteomes_data and proteomes_data["results"]:
                                        available_proteomes = proteomes_data["results"]
                                        # randomly select 100 proteomes
                                        
                                        selected_proteomes = random.sample(available_proteomes, min(100, len(available_proteomes)))
                                        
                                        for proteome in tqdm(selected_proteomes, desc=f"Downloading {taxonomy_name} proteomes"):
                                            proteome_id = proteome["id"]
                                            # organism name or ID
                                            organism_name = proteome.get("organism", {}).get("scientificName", 
                                                          proteome.get("taxonomy", {}).get("scientificName", proteome_id))
                                            
                                            
                                            safe_name = re.sub(r'[^\w\-\.]', '_', str(organism_name))
                                            file_path = f"{tax_dir}/{safe_name}.fasta"

                                            # Download
                                            download_url = f"https://rest.uniprot.org/uniprotkb/stream?query=proteome:{proteome_id}&format=fasta"
                                            await download_file(download_url, file_path, session)
                                            await asyncio.sleep(0.5)
                                    break 
                            
                        except aiohttp.ClientError as e:
                            if attempt == max_retries - 1:
                                print(f"Failed after {max_retries} attempts: {e}")
                            else:
                                print(f"Attempt {attempt+1} failed, retrying...")
                                await asyncio.sleep(1 * (2 ** attempt))
                
                except Exception as e:
                    print(f"Error downloading proteomes for {taxonomy_name}: {e}")
  
            files = [f.path for f in os.scandir(tax_dir) if f.is_file() and f.name.endswith(".fasta")]
            files_by_taxonomy[taxonomy_name] = files
    
    return files_by_taxonomy

async def download_all_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    await download_organisms()
    await download_pdb()
    await download_swissprot()
    await download_taxonomies()


if __name__ == "__main__":
    asyncio.run(download_all_data())
