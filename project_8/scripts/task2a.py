import time
import argparse
from Bio import SeqIO
import re
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Six-frame ORF finder using BioPython')
    parser.add_argument('fasta_file', help='Input FASTA file')
    parser.add_argument('--output', '-o', default='biopython_orfs.fasta',
                        help='Output file for ORFs (default: biopython_orfs.fasta)')
    parser.add_argument('--table', '-t', type=int, default=1,
                        help='Genetic code table to use (default: 1, Standard, 2 for Vertebrate Mitochondrial)')
    
    return parser.parse_args()

def merge_frame_translations(translations_raw):
    header_section, translation_section = str(translations_raw).strip().split("\n\n\n", 1)
    translation_blocks = translation_section.strip().split("\n\n")
    
    merged_translations = {i: "" for i in range(1, 7)}
    
    for block in translation_blocks:
        lines = block.strip().split("\n")
        
        # Forward frames (1-3)
        for i, line in enumerate(lines[1:4], 1):
            merged_translations[i] += line.replace(" ", "").replace("-", "")

        # Reverse frames (4-6)
        for i, line in enumerate(lines[6:9], 4):
            merged_translations[i] += line.replace(" ", "").replace("-", "")
    
    return merged_translations

def find_orfs(seq_record, table=1):
    orfs = []
    min_length=100
    seq_id = seq_record.id
    seq = seq_record.seq
    num = 1
    
    # forward frames 
    for frame in range(3):
        translated = seq[frame:].translate(table=table)
        matches = re.finditer(r'M[^*]*\*', str(translated))
        
        for match in matches:
            orf = match.group()[:-1] 
            if len(orf) < min_length:
                continue
                
            start_pos = match.start()
            end_pos = match.end() - 1
            
            dna_start = frame + start_pos * 3
            dna_end = frame + end_pos * 3 + 3
            
            frame_num = frame + 1 
            orf_id = f"{seq_id}|ORF{num}:{dna_start}-{dna_end}|frame={frame_num}|len={len(orf)}"
            orfs.append((orf_id, orf))
            num += 1

    # reverse frames 
    rev_seq = seq.reverse_complement()
    for frame in range(3):
        translated = rev_seq[frame:].translate(table=table)
        matches = re.finditer(r'M[^*]*\*', str(translated))
        
        for match in matches:
            orf = match.group()[:-1] 
            if len(orf) < min_length:
                continue
                
            start_pos = match.start()
            end_pos = match.end() - 1
            
            rev_start = frame + start_pos * 3
            rev_end = frame + end_pos * 3 + 3
            
            dna_start = len(seq) - rev_end
            dna_end = len(seq) - rev_start
            
            frame_num = frame + 4 
            orf_id = f"{seq_id}|ORF{num}:{dna_start}-{dna_end}|frame={frame_num}|len={len(orf)}"
            orfs.append((orf_id, orf))
            num += 1

    return orfs

def main():
    print(f"Running BioPython ORF Finder")
    args = parse_args()
    seq_records = list(SeqIO.parse(args.fasta_file, "fasta"))
    
    start_time = time.time()
    all_orfs = []

    for record in seq_records:
        orfs = find_orfs(record, args.table)
        all_orfs.extend(orfs)
      
    # output file
    with open(f"results/{args.output}", 'w') as output_file:
        for orf_id, orf in all_orfs:
            output_file.write(f">{orf_id}\n{orf}\n")
    
    end_time = time.time()
    run_time = end_time - start_time

    # runtime data
    runtime_data = {
        'method': 'biopython',
        'input_file': args.fasta_file.split('/')[-1],
        'total_orfs': len(all_orfs),
        'runtime_seconds': run_time
    }
    with open('tables/runtimes.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=runtime_data.keys())
        if csvfile.tell() == 0: 
            writer.writeheader()
        writer.writerow(runtime_data)

    print(f"Results written to: results/{args.output}")
    print(f"Runtime data saved to: results/runtimes.csv")

if __name__ == "__main__":
    main()