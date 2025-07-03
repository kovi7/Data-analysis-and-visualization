'''Task 1:
Write the python program that will calculate:

1) for full human genome:
- length
- nucleotides numbers and frequencies
- GC content

2) for each chromosome:
- length
- nucleotides numbers and frequencies
- GC content

The program should read line after line* and gather statistics and then show them on the screen (prettytable) and additionally store them in a CSV file. 

* Avoid loading whole file into memory (3GB).

Hint: it is advised to inspect the content, or/and create shorter version of the file for testing (unix commands: head, tail, cut, grep, wc, etc.).

Result: python script that can generate csv 
chr_id/total,GC%,G,C,T,A,N,G%,C%,T%,A%,N%,len'''

import csv
from collections import Counter
import gzip
from prettytable import PrettyTable

genome_file = 'data/chm13v2.0.fa.gz'
output = 'tables/stats.csv'

total_stats = {
    'length': 0,
    'nucleotides': Counter(),
    'gc_content': 0
}
chromosome_stats = {}
current_chr = None
current_seq = []

def process():
    sequence = ''.join(current_seq).upper()
    length = len(sequence)
    nucleotides = Counter(sequence)
    
    gc_count = nucleotides['G'] + nucleotides['C']
    gc_content = (gc_count / length) * 100 if length > 0 else 0
    
    chromosome_stats[current_chr] = {
        'length': length,
        'nucleotides': nucleotides,
        'gc_content': gc_content
    }
    
    total_stats['length'] += length
    total_stats['nucleotides'].update(nucleotides)
    
    gc_total = total_stats['nucleotides']['G'] + total_stats['nucleotides']['C']
    total_stats['gc_content'] = (gc_total / total_stats['length']) * 100



opener = gzip.open if genome_file.endswith('.gz') else open
        
with opener(genome_file, 'rt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('>'):
            if current_chr:
                process()
            current_chr = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    
    if current_chr:
        process()

table = PrettyTable()
table.field_names = ['chr', 'GC%', 'G', 'C', 'T', 'A', 'N', 'G%', 'C%', 'T%', 'A%', 'N%', 'len(bp)']
with open(output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(table.field_names)
    
    for chr_id, stats in chromosome_stats.items():
        length = stats['length']
        nucleotides = stats['nucleotides']
        chr_row = [
            chr_id,
            f"{stats['gc_content']:.2f}",
            nucleotides['G'],
            nucleotides['C'],
            nucleotides['T'],
            nucleotides['A'],
            nucleotides['N'],
            f"{(nucleotides['G'] / length * 100):.2f}",
            f"{(nucleotides['C'] / length * 100):.2f}",
            f"{(nucleotides['T'] / length * 100):.2f}",
            f"{(nucleotides['A'] / length * 100):.2f}",
            f"{(nucleotides['N'] / length * 100):.2f}",
            length
        ]
        table.add_row(chr_row)
        writer.writerow(chr_row)

    total_length = total_stats['length']
    nucleotides = total_stats['nucleotides']
    total_row = [
        'Total',
        f"{total_stats['gc_content']:.2f}",
        nucleotides['G'],
        nucleotides['C'],
        nucleotides['T'],
        nucleotides['A'],
        nucleotides['N'],
        f"{(nucleotides['G'] / total_length * 100):.2f}",
        f"{(nucleotides['C'] / total_length * 100):.2f}",
        f"{(nucleotides['T'] / total_length * 100):.2f}",
        f"{(nucleotides['A'] / total_length * 100):.2f}",
        f"{(nucleotides['N'] / total_length * 100):.2f}",
        total_length
    ]
    table.add_row(total_row)
    writer.writerow(total_row)

print(table)



