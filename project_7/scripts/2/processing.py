
import os
import re
import random
import mmap
import numpy as np
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import csv

#constants
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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

def parse_fasta(file_path):
    try:
        with open(file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            seq_id, desc, seqs = None, None, []
            mol_pattern = re.compile(r'mol:(\w+)')
            for line in iter(mm.readline, b''):
                line = line.decode('utf-8').strip()
                if line.startswith(">"):
                    if seq_id and seqs:
                        yield SeqRecord(Seq.Seq("".join(seqs)), id=seq_id, description=desc)
                    seq_id = line[1:].split()[0]
                    desc = mol_pattern.search(line).group(1) if mol_pattern.search(line) else None
                    seqs = []
                else:
                    seqs.append(line)
            if seq_id and seqs:
                yield SeqRecord(Seq.Seq("".join(seqs)), id=seq_id, description=desc)
            mm.close()
    except Exception:
        for record in SeqIO.parse(file_path, "fasta"):
            yield record

def calculate_stats(sequences):
    if not sequences:
        return None
        
    lengths = [len(seq.seq) for seq in sequences]
    aa_counts = defaultdict(int)
    n_term_counts = defaultdict(int)
    
    for seq in sequences:
        seq_str = str(seq.seq).upper()
        for aa in seq_str:
            if aa in AA_LIST:
                aa_counts[aa] += 1
                
        if seq_str and seq_str[0] in AA_LIST:
            n_term_counts[seq_str[0]] += 1
            
    total_aa = sum(aa_counts.values())
    aa_pct = {aa: (aa_counts[aa]/total_aa*100) if total_aa else 0 for aa in AA_LIST}
    n_term_pct = {aa: (n_term_counts[aa]/len(sequences)*100) if sequences else 0 for aa in AA_LIST}
    
    return {
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'aa_percentages': aa_pct,
        'n_terminus_percentages': n_term_pct,
        'sequence_count': len(sequences),
        'total_aa':total_aa
    }

def bootstrap_stats(sequences_tuple, n_bootstrap=100):
    sequences = [SeqRecord(Seq.Seq(seq), id=id) for id, seq in sequences_tuple]
    length_samples = []
    aa_samples = {aa: [] for aa in AA_LIST}
    
    for _ in range(n_bootstrap):
        sample_stats = calculate_stats(random.choices(sequences, k=len(sequences)))
        length_samples.append(sample_stats['avg_length'])
        for aa in AA_LIST:
            aa_samples[aa].append(sample_stats['aa_percentages'][aa])
            
    return np.std(length_samples), {aa: np.std(aa_samples[aa]) for aa in AA_LIST}

def process_data(file_path, data_type=None, taxonomy=None, bootstrap=True):
    if isinstance(file_path, list):
        sequences = file_path
    else:
        sequences = list(parse_fasta(file_path))
        
    if data_type == "pdb":
        sequences = [seq for seq in sequences if seq.description == "protein"]
        
    if not sequences:
        return None
    
    stats = calculate_stats(sequences)

    if data_type:
        stats['data_type'] = data_type
        
    elif taxonomy:
        stats['taxonomy_group'] = taxonomy
        stats['organism'] = os.path.basename(file_path).replace(".fasta", "")

    if bootstrap and len(sequences) >= 10:
        sample = sequences[:min(1000, len(sequences))]
        seq_tuples = tuple((seq.id, str(seq.seq)) for seq in sample)
        bs_length_std, bs_aa_std = bootstrap_stats(seq_tuples)
        stats['bootstrap_length_std'] = bs_length_std
        stats['bootstrap_aa_std'] = bs_aa_std
    else:
        stats['bootstrap_length_std'] = None
        stats['bootstrap_aa_std'] = None  
        
    return stats

def save_to_csv(results, output_file):
    if not results:
        return
        
    if not isinstance(results, list):
        results = [results]
        
    headers = set()
    for result in results:
        headers.update(result.keys())
        
    aa_percentage_fields = [f'%{aa}' for aa in AA_LIST]
    n_term_percentage_fields = [f'N_term_%{aa}' for aa in AA_LIST]
    bootstrap_aa_std_fields = [f'bootstrap_std_%{aa}' for aa in AA_LIST]
    
    base_headers = [h for h in headers if h not in ['aa_percentages', 'n_terminus_percentages', 'bootstrap_aa_std']]
    all_headers = base_headers + aa_percentage_fields + n_term_percentage_fields + bootstrap_aa_std_fields
    
    rows = []
    for result in results:
        row = {k: v for k, v in result.items() if k in base_headers}
        
        # amino acid percentages
        if 'aa_percentages' in result:
            for aa in AA_LIST:
                row[f'%{aa}'] = result['aa_percentages'].get(aa, 0.0)
        
        # N-terminus percentages
        if 'n_terminus_percentages' in result:
            for aa in AA_LIST:
                row[f'N_term_%{aa}'] = result['n_terminus_percentages'].get(aa, 0.0)
        
        # bootstrap standard deviations for amino acids
        if 'bootstrap_aa_std' in result and result['bootstrap_aa_std']:
            for aa in AA_LIST:
                row[f'bootstrap_std_%{aa}'] = result['bootstrap_aa_std'].get(aa, 0.0)
        
        rows.append(row)
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_headers)
        writer.writeheader()
        writer.writerows(rows)

def process_organism_files(organism_files):
    results = {}
    official = ['E. coli', 'B. subtilis', 'C. elegans','H. sapiens', 'S. cerevisiae', 'A. thaliana', 'D. melanogaster', 'M. musculus', 'D. rerio']

    with tqdm(organism_files, desc="Processing Organisms") as pbar:
        for file_path in pbar:
            if os.path.exists(file_path):
                organism = os.path.basename(file_path).replace(".fasta", "")
                i=list(ORGANISMS).index(organism)
                pbar.set_description(f"Processing {organism}")
                stats = process_data(file_path)
                if stats:
                    stats['organism'] = official[i]
                    results[organism] = stats
    
    if results:
        save_to_csv(list(results.values()), f"{DATA_DIR}/organism_stats.csv")
    return results

def process_pdb_file(pdb_file):
    if os.path.exists(pdb_file):
        with tqdm(total=1, desc="Processing PDB") as pbar:
            result = process_data(pdb_file, data_type="pdb", bootstrap=True)
            pbar.update(1)
            
            if result:
                save_to_csv(result, f"{DATA_DIR}/pdb_stats.csv")
                return result
    return None
def process_swissprot_file(swissprot_file):
    if os.path.exists(swissprot_file):
        with tqdm(total=1, desc="Processing SwissProt") as pbar:
            result = process_data(swissprot_file, data_type="swissprot", bootstrap=True)
            pbar.update(1)
            
            if result:
                save_to_csv(result, f"{DATA_DIR}/swissprot_stats.csv")
                return result
    return None

def process_taxonomy_files(taxonomy_files):
    all_results = []
    
    for taxonomy_name, files in taxonomy_files.items():
        taxonomy_results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_data, f, taxonomy=taxonomy_name) for f in files]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {taxonomy_name}",
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
                result = future.result()
                if result:
                    taxonomy_results.append(result)
        
        all_results.extend(taxonomy_results)
    
    if all_results:
        save_to_csv(all_results, f"{DATA_DIR}/taxonomy_stats.csv")
        return all_results
    else:
        print("No taxonomy results to save")
        return []

def process_all_data(data_files):
    organism_results = process_organism_files(data_files.get("organism_files", []))
    pdb_result = process_pdb_file(data_files.get("pdb_file"))
    swissprot_result = process_swissprot_file(data_files.get("swissprot_file"))
    taxonomy_results = process_taxonomy_files(data_files.get("taxonomy_files", {}))
    
    return {
        "organism_results": organism_results,
        "pdb_result": pdb_result,
        "swissprot_result": swissprot_result,
        "taxonomy_results": taxonomy_results
    }

if __name__ == "__main__":
    data_files = {
        "organism_files": [],
        "pdb_file": "",
        "swissprot_file": "",
        "taxonomy_files": {}
    }
    
    organism_files = [f"{RAW_DIR}/{organism}.fasta" for organism in ORGANISMS.keys()]
    data_files["organism_files"] = [f for f in organism_files if os.path.exists(f)]
    
    pdb_path = f"{RAW_DIR}/pdb_seqres.txt"  
    if os.path.exists(pdb_path):
        data_files["pdb_file"] = pdb_path
    
    swissprot_path = f"{RAW_DIR}/swissprot_uniprot.fasta"  
    if os.path.exists(swissprot_path):
        data_files["swissprot_file"] = swissprot_path
    
    for taxonomy_name in TAXONOMIES.keys():
        taxonomy_dir = f"{RAW_DIR}/{taxonomy_name}"
        if os.path.exists(taxonomy_dir):
            data_files["taxonomy_files"][taxonomy_name] = [
                os.path.join(taxonomy_dir, f) for f in os.listdir(taxonomy_dir)
                if f.endswith(".fasta")
            ]
    
    # Process all data files
    process_all_data(data_files)
    print("Data processing complete!")