import os
import pandas as pd
from Bio import SeqIO
import gzip
from prettytable import PrettyTable
from difflib import SequenceMatcher


def load_orfs(file_path):
    orfs = set()
    with open(file_path, 'r') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    orfs.add(current_seq)
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            orfs.add(current_seq)
    return orfs

def load_official_proteins(file_path):
    proteins = set()
        
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq)
        if seq.endswith('*'):
            seq = seq[:-1]
        proteins.add((seq, len(seq)))
    print("Official: ", len(proteins))
    return proteins

def load_gzipped_fasta(file_path):
    sequences = set()
    
    with gzip.open(file_path, 'rt') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    if current_seq.endswith('*'):
                        current_seq = current_seq[:-1]
                    sequences.add(current_seq)
                current_seq = ""
            else:
                current_seq += line
        
        if current_seq:
            if current_seq.endswith('*'):
                current_seq = current_seq[:-1]
            sequences.add(current_seq)
    
    print("Uniprot: ", len(sequences))
    return sequences


def compare_with_reference(method_orfs, reference_seqs, reference_name, error_tolerance=0.9):
    method_seqs = {seq for seq in method_orfs}
    
    common_proteins = set()
    for method_seq in method_seqs:
        for ref_seq in reference_seqs:
            similarity = SequenceMatcher(None, method_seq, ref_seq).ratio()
            if similarity >= error_tolerance:
                common_proteins.add(method_seq)
                break
    
    return {
        'total_method': len(method_seqs),
        'total_reference': len(reference_seqs),
        'common': len(common_proteins),
        'match_percentage': (len(common_proteins)/len(method_seqs))*100 if len(method_seqs) > 0 else 0
    }

def analyze_chromosome(chromosome, runtimes_df):
    print(f"\nAnalyzing chromosome {chromosome}...")
    
    table = PrettyTable()
    table.field_names = ["Method", "Runtime (s)", "Total ORFs", 
                        "UniProt Matches", "UniProt Match %", 
                        "CHM13 Matches", "CHM13 Match %"]
    
    uniprot_seqs = load_gzipped_fasta("data/uniprotkb_human_proteome.fasta.gz")
    chm13_proteins = load_official_proteins("data/chm13.draft_v1.1.gene_annotation.protein.fasta")
    chm13_seqs = {seq for seq, _ in chm13_proteins}
    
    methods = {
        'biopython': 'BioPython',
        'emboss': 'EMBOSS',
        'pure python': 'Pure Python'
    }
    
    chr_runtimes = runtimes_df[runtimes_df['input_file'] == f"{chromosome}.fa"]
    
    for method_key, method_name in methods.items():
        method_runtime = chr_runtimes[chr_runtimes['method'] == method_key]['runtime_seconds'].values
        runtime = method_runtime[0] if len(method_runtime) > 0 else "N/A"
        
        orf_file = f"results/{chromosome}_{method_key.replace(' ', '_')}_orfs.fa"
        orfs = load_orfs(orf_file)
        
        # UniProt
        uniprot_comparison = compare_with_reference(orfs, uniprot_seqs, "UniProt")
        
        # CHM13
        chm13_comparison = compare_with_reference(orfs, chm13_seqs, "CHM13")
        
        table.add_row([
            method_name,
            runtime,
            len(orfs),
            uniprot_comparison['common'],
            f"{uniprot_comparison['match_percentage']:.2f}",
            chm13_comparison['common'],
            f"{chm13_comparison['match_percentage']:.2f}"
        ])
    
    with open(f"tables/{chromosome}_comparison_table.tex", "w") as f:
        f.write(f"Comparison for {chromosome}\n")
        f.write(table.get_latex_string())

    return table

if __name__ == "__main__":
    runtimes_df = pd.read_csv("tables/runtimes.csv")
    
    chrM_table = analyze_chromosome("chrM", runtimes_df)
    chr1_table = analyze_chromosome("chr1", runtimes_df)
    
    print("\nComparison for chrM:")
    print(chrM_table)
    
    print("\nComparison for chr1:")
    print(chr1_table)
    uniprot_seqs = load_gzipped_fasta("data/uniprotkb_human_proteome.fasta.gz")
    chm13_proteins = load_official_proteins("data/chm13.draft_v1.1.gene_annotation.protein.fasta")
 