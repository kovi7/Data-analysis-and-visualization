import argparse
import subprocess
import time
import csv
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Six-frame ORF finder using EMBOSS getorf')
    parser.add_argument('fasta_file', help='Input FASTA file')
    parser.add_argument('--output', '-o', default='emboss_orfs.fasta',
                       help='Output file for ORFs (default: emboss_orfs.fasta)')
    parser.add_argument('--table', "-t",type=int, default=0,
                       help='Genetic code table to use (default: 0, Standard; 2 for Vertebrate Mitochondrial)')
    return parser.parse_args()

def run_getorf(input_file, output_file, table=0):
    cmd = [
        'getorf',
        '-sequence', input_file,
        '-outseq', output_file,
        '-table', str(table),
        '-minsize', '300',
        '-find', str(1)
        ]

    subprocess.run(cmd, check=True)
    temp_output = output_file + '.temp'
    format_fasta_output(output_file, temp_output)
    subprocess.run(['mv', temp_output, output_file], check=True)


def format_fasta_output(input_file, output_file, table=0):
    with open(input_file, 'r') as f:
        content = f.read()
    
    orf_sections = content.split('>')[1:]
    
    formatted_content = []
    num = 1
    for section in orf_sections:
        lines = section.strip().split('\n')
        header = lines[0]
        sequence = ''.join(lines[1:]).upper()
        
        is_reverse = 'REVERSE' in header
        
        match = re.match(r'(chr\w+)_\d+\s+\[(\d+)\s*-\s*(\d+)\](?:\s*\(REVERSE\s*SENSE\))?', header)
        if match:
            seq_name, start, end = match.groups()
            
            if is_reverse:
                frame = 4 + ((int(end) - 1) % 3)
            else:
                frame = 1 + ((int(start) - 1) % 3)
            
            
            new_header = f">{seq_name}|ORF{num}:{start}-{end}|frame={frame}|len={len(sequence)}"
            
            formatted_content.append(new_header)
            formatted_content.append(sequence)
            num +=1

    with open(output_file, 'w') as f:
        f.write('\n'.join(formatted_content))

def main():
    print("Running EMBOSS ORF Finder")
    args = parse_args()
    output_path = f"results/{args.output}"
    
    start_time = time.time()
    
    run_getorf(
        args.fasta_file,
        output_path,
        table=args.table
    )

    # number of ORFs
    with open(output_path, 'r') as f:
        orf_count = sum(1 for line in f if line.startswith('>'))
    
    end_time = time.time()
    run_time = end_time - start_time
    
    # runtime data
    runtime_data = {
        'method': 'emboss',
        'input_file': args.fasta_file.split('/')[-1],
        'total_orfs': orf_count,
        'runtime_seconds': run_time
    }
    
    with open('tables/runtimes.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=runtime_data.keys())
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(runtime_data)
    
    print(f"Results written to: {output_path}")
    print(f"Runtime data saved to: results/runtimes.csv")


if __name__ == "__main__":
    main()