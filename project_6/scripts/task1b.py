from task1a import Parser
import hashlib
import random

'''
b) Count how many proteins/peptides have pI < 5.0.

Let's create a dataset of acidic proteins and peptides:
- From the protein dataset, take all proteins with pI < 5.0 (e.g. 550 items --> positives, label 1).
- Then, take 550 most basic proteins with pI > 10 (negatives, label 0).

For the larger peptide dataset:
- Extract all peptides with pI < 5.0 and those with pI > 10.
- Randomly select 550 from each group (550 positives and 550 negatives).

Finally, merge the protein and peptide records into one dataset of 2200 items.

Format:
>UID|peptide/protein|pI|1/0
sequence

Use MD5 hash of the sequence as UID:

python
import hashlib
sequence = "YDNSLTVVSNASCTTNCLAPLAK"
res = hashlib.md5(sequence.encode())
MD5_hash_uid = res.hexdigest()
print(MD5_hash_uid)
# Output: ec22b6bb20f548f06b81a1a0760d78d6

Save as: IPC_classification_dataset_100.fasta'''


def create_balanced_dataset(protein_data, peptide_data):
    acidic_proteins = [p for p in protein_data if p['pI'] is not None and p['pI'] < 5]
    basic_proteins = [p for p in protein_data if p['pI'] is not None and p['pI'] > 10]
    
    # acidic/basic proteins from proteins
    acidic_proteins = acidic_proteins[:550]
    basic_proteins = sorted(basic_proteins, key=lambda x: x['pI'], reverse=True)[:550]
    
    # Randomly sample peptides
    acidic_peptides = [p for p in peptide_data if p['pI'] is not None and p['pI'] < 5]
    basic_peptides = [p for p in peptide_data if p['pI'] is not None and p['pI'] > 10]
    acidic_peptides_sample = random.sample(acidic_peptides, 550)
    basic_peptides_sample = random.sample(basic_peptides, 550)
    
    combined_data = []
    
    for item in acidic_proteins + basic_proteins + acidic_peptides_sample + basic_peptides_sample:
        uid = hashlib.md5(item['sequence'].encode()).hexdigest()        
        source_type = 'protein' if item in (acidic_proteins + basic_proteins) else 'peptide'
        record = {
            'uid': uid,
            'source_type': source_type,
            'pI': item['pI'],
            'label': item['label'],
            'sequence': item['sequence']
        }
        
        combined_data.append(record)
    
    return combined_data

def main():
    p = Parser
    peptide = p.parse_peptide('data/Gauci_PHENYX_SEQUEST_0.99_duplicates_out.fasta')
    protein = p.parse_protein('data/pip_ch2d19_2_1st_isoform_outliers_3units_cleaned_0.99.fasta')

    #Count how many proteins/peptides have pI < 5.0.
    proteins_count = sum(1 for p in protein if p['pI'] is not None and p['pI'] < 5)
    peptides_count = sum(1 for p in peptide if p['pI'] is not None and p['pI'] < 5)
    print(f"Number of proteins with pI < 5.0: {proteins_count}")
    print(f"Number of peptides with pI < 5.0: {peptides_count}")    
    
    proteins_count_10 = sum(1 for p in protein if p['pI'] is not None and p['pI'] > 10)
    peptides_count_10 = sum(1 for p in peptide if p['pI'] is not None and p['pI'] > 10)
    print(f"Number of proteins with pI > 10.0: {proteins_count_10}")
    print(f"Number of peptides with pI > 10.0: {peptides_count_10}")


    data = create_balanced_dataset(protein, peptide)

    with open("data/IPC_classification_dataset_100.fasta", 'w') as f:
        for item in data:
            header = f">{item['uid']}|{item['source_type']}|{item['pI']}|{item['label']}"
            f.write(f"{header}\n{item['sequence']}\n")


if __name__ == "__main__":
    main()
