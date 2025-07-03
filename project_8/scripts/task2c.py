import time
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Six-frame ORF finder using pure Python')
    parser.add_argument('fasta_file', help='Input FASTA file')
    parser.add_argument('--output', '-o', default='python_orfs.fasta',
                        help='Output file for ORFs (default: python_orfs.fasta)')
    parser.add_argument('--table', '-t', type=int, default=1,
                        help='Genetic code table to use (default: 1, Standard; 2 for Vertebrate Mitochondrial)')
    return parser.parse_args()

def read_fasta(filename):
    sequences = {}
    current_seq = ""
    current_id = ""
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = current_seq
                current_id = line[1:].split()[0] 
                current_seq = ""
            else:
                current_seq += line
        
        if current_id and current_seq:
            sequences[current_id] = current_seq
            
    return sequences

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(dna))

def get_genetic_code(table_id):
    standard_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    }
    
    mitochondrial_table = {
        'ATA':'M', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'*', 'AGG':'*',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'W', 'TGG':'W',
    }
    
    if table_id == 2:
        return mitochondrial_table
    else:
        return standard_table

def translate_dna(sequence, genetic_code, frame=0):
    protein = []
    
    for i in range(frame, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        
        if len(codon) < 3:
            continue
        if 'N' in codon:
            protein.append('X')
            continue
            
        aa = genetic_code.get(codon, 'X') 
        protein.append(aa)
        
    return ''.join(protein)

def find_orfs(seq_id, seq, genetic_code):
    orfs = []
    min_protein_length = 100
    num = 1
    
    # forward frames
    for frame in range(3):
        prot_seq = translate_dna(seq, genetic_code, frame)
        
        start = 0
        while True:
            start = prot_seq.find('M', start)
            if start == -1:
                break     

            end = prot_seq.find('*', start)
            if end == -1: 
                start += 1
                continue
            
            if end - start <= min_protein_length:
                start += 1
                continue
            
            orf_seq = prot_seq[start:end]
            
            dna_start = start * 3 + frame
            dna_end = end * 3 + frame
            
            if orfs and orfs[-1][0].split('|')[1].split(':')[1].split('-')[0] <= str(dna_start) <= orfs[-1][0].split('|')[1].split(':')[1].split('-')[1]:
                start += 1
                continue
            
            frame_num = frame + 1 
            orf_id = f"{seq_id}|ORF{num}:{dna_start}-{dna_end}|frame={frame_num}|len={end-start}"
            orfs.append((orf_id, orf_seq))
            
            num += 1
            start = end + 1 
    
    # reverse frames 
    rev_seq = reverse_complement(seq)
    for frame in range(3):
        prot_seq = translate_dna(rev_seq, genetic_code, frame)
        
        start = 0
        while True:
            start = prot_seq.find('M', start)
            if start == -1:
                break
                
            end = prot_seq.find('*', start)
            if end == -1:  
                start += 1
                continue
            
            if end - start <= min_protein_length:
                start += 1
                continue
            
            orf_seq = prot_seq[start:end]

            dna_start = len(seq) - (end * 3 + frame + 3)  
            dna_end = len(seq) - (start * 3 + frame)
            
            if orfs and orfs[-1][0].split('|')[1].split(':')[1].split('-')[0] <= str(dna_start) <= orfs[-1][0].split('|')[1].split(':')[1].split('-')[1]:
                start += 1
                continue
            
            frame_num = frame + 4 
            orf_id = f"{seq_id}|ORF{num}:{dna_start}-{dna_end}|frame={frame_num}|len={end-start}"
            orfs.append((orf_id, orf_seq))
            
            num += 1
            start = end + 1 
    
    return orfs

def main():
    print(f"Running Pure Python ORF Finder")
    args = parse_args()
    sequences = read_fasta(args.fasta_file)
    genetic_code = get_genetic_code(args.table)
    
    start_time = time.time()
    all_orfs = []
    
    for seq_id, seq in sequences.items():
        orfs = find_orfs(seq_id, seq, genetic_code)
        all_orfs.extend(orfs)
    
    # output file
    with open(f"results/{args.output}", 'w') as output_file:
        for orf_id, orf in all_orfs:
            output_file.write(f">{orf_id}\n{orf}\n")
    
    end_time = time.time()
    run_time = end_time - start_time

    # runtime data
    runtime_data = {
        'method': 'pure python',
        'input_file': args.fasta_file.split('/')[-1],
        'total_orfs': len(all_orfs),
        'runtime_seconds': run_time
    }
    
    with open('tables/runtimes.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=runtime_data.keys())
        if not csvfile.tell():
            writer.writeheader()
        writer.writerow(runtime_data)

    print(f"Results written to: {args.output}")
    print(f"Runtime data saved to: runtimes.csv")

if __name__ == "__main__":
    main()