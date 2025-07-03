import sys
import os
from collections import defaultdict

def parse_fasta_files(disorder_file, ss_file):
    disorder_data = {}
    ss_data = {}
    
    with open(disorder_file) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            if i+3 >= len(lines): break
            header, seq, disorder, scores = [l.strip() for l in lines[i:i+4]]
            uniprot_id = header.lstrip('>')
            disorder_data[uniprot_id] = {
                'header': header,
                'seq': seq,
                'disorder': disorder,
                'scores': scores
            }
    
    with open(ss_file) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if i+2 >= len(lines): break
            header, seq, ss = [l.strip() for l in lines[i:i+3]]
            uniprot_id = header.lstrip('>')
            ss_data[uniprot_id] = {
                'header': header,
                'seq': seq,
                'ss': ss
            }
    
    return disorder_data, ss_data

def find_disorder_fragments(disorder_string, min_length=40):
    fragments = []
    start = None
    
    for i, char in enumerate(disorder_string):
        if char == 'D':
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_length:
                    fragments.append((start, i-1))
                start = None
    
    if start is not None and len(disorder_string) - start >= min_length:
        fragments.append((start, len(disorder_string)-1))
    
    return fragments

def calculate_induced_score(ss_fragment):
    H_count = ss_fragment.count('H')
    E_count = ss_fragment.count('E')
    dash_count = ss_fragment.count('-')
    
    score = (2 * H_count) + (2 * E_count) - (4 * dash_count)
    
    return score, H_count, E_count, dash_count

def main():
    if len(sys.argv) != 7:
        print("Usage: python3 filter_induced_disorder.py <disorder_file> <ss_file> <top_n> <min_length> <max_coil_percent> <remove_dups>")
        sys.exit(1)
    
    disorder_file = sys.argv[1]
    ss_file = sys.argv[2]
    top_n = int(sys.argv[3])
    min_length = int(sys.argv[4])
    max_coil_percent = float(sys.argv[5])
    remove_dups = int(sys.argv[6])
    
    disorder_data, ss_data = parse_fasta_files(disorder_file, ss_file)
    
    common_proteins = set(disorder_data.keys()) & set(ss_data.keys())
    
    results = []
    seen_sequences = set() if remove_dups else None
    
    for protein_id in common_proteins:
        disorder_info = disorder_data[protein_id]
        ss_info = ss_data[protein_id]
        
        if len(disorder_info['disorder']) != len(ss_info['ss']):
            continue
        
        if remove_dups and disorder_info['seq'] in seen_sequences:
            continue
        fragments = find_disorder_fragments(disorder_info['disorder'], min_length)
        
        if not fragments:
            continue
        fragment_details = []
        max_score = float('-inf')
        
        for start, end in fragments:
            ss_fragment = ss_info['ss'][start:end+1]
            fragment_length = end - start + 1
            coil_percent = (ss_fragment.count('-') / fragment_length) * 100
            
            if coil_percent > max_coil_percent:
                continue
            
            # InducedDIS-score
            score, H_count, E_count, dash_count = calculate_induced_score(ss_fragment)
            
            fragment_info = {
                'start': start + 1,
                'end': end + 1,
                'H': H_count,
                'E': E_count,
                '-': dash_count,
                'score': score,
                'ss_fragment': ss_fragment,
                'disorder_fragment': disorder_info['disorder'][start:end+1]
            }
            
            fragment_details.append(fragment_info)
            max_score = max(max_score, score)
        
        if fragment_details:
            fragment_details.sort(key=lambda x: x['score'], reverse=True)
            
            if remove_dups:
                seen_sequences.add(disorder_info['seq'])
            
            results.append({
                'protein_id': protein_id,
                'fragments': fragment_details,
                'max_score': max_score,
                'seq': disorder_info['seq'],
                'disorder': disorder_info['disorder'],
                'ss': ss_info['ss']
            })
    results.sort(key=lambda x: x['max_score'], reverse=True)
    
    results = results[:top_n]
    
    output_file = f"predictions/{os.path.basename(disorder_file.split('.')[0])}_top{top_n}_{min_length}_{int(max_coil_percent)}_{remove_dups}.fasta"
    
    with open(output_file, 'w') as f:
        for result in results:
            fragment_descriptions = []
            for frag in result['fragments']:
                desc = f"disorder[{frag['start']}-{frag['end']}]:H:{frag['H']},E:{frag['E']},-:{frag['-']},InducedDIS-score={frag['score']}"
                fragment_descriptions.append(desc)
            header = f">{result['protein_id']}|{', '.join(fragment_descriptions)}"
            f.write(f"{header}\n")
            f.write(f"{result['seq']}\n")
            f.write(f"{result['disorder']}\n")
            f.write(f"{result['ss']}\n")
if __name__ == "__main__":
    main()
