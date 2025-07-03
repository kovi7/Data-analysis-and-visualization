import csv
import os
from sklearn.metrics import matthews_corrcoef

def parse_iupred_multiline_fasta(filepath):
    records = {}
    with open(filepath) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            if i+3 >= len(lines): break
            header, seq, disorder, scores = [l.strip() for l in lines[i:i+4]]
            uniprot_id = header.split('|')[1] if '|' in header else header.lstrip('>')
            records[uniprot_id] = {
                'header': header,
                'seq': seq,
                'disorder': disorder,
                'scores': scores
            }
    return records

def get_disorder_mask(disorder_string):
    return [1 if x == 'D' else 0 for x in disorder_string]

def get_ground_truth_mask(length, regions):
    mask = [0]*length
    for start, end in regions:
        for i in range(start-1, end):
            if i < length:
                mask[i] = 1
    return mask

def parse_disprot_csv(disprot_csv):
    disprot = {}
    with open(disprot_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            acc, frags = row
            fragments = []
            for frag in frags.strip().replace('][',';').replace('[','').replace(']','').split(';'):
                if frag:
                    s,e = frag.split('-')
                    fragments.append((int(s),int(e)))
            disprot[acc] = fragments
    return disprot

def main(iupred_fasta, disprot_csv):
    iupred = parse_iupred_multiline_fasta(iupred_fasta)
    disprot = parse_disprot_csv(disprot_csv)
    y_true, y_pred = [], []
    n_found = 0
    for protid in disprot:
        if protid in iupred:
            mask_true = get_ground_truth_mask(len(iupred[protid]['disorder']), disprot[protid])
            mask_pred = get_disorder_mask(iupred[protid]['disorder'])
            y_true.extend(mask_true)
            y_pred.extend(mask_pred)
            n_found += 1
    print(f"Matched {n_found} proteins between IUPred and DisProt.")
    mcc_c = matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc_c)
    sum_line = f'{os.path.basename(sys.argv[1])}| MCC: {mcc_c}\n'
    with open('../mcc_tab.txt','a') as file:
       file.write(sum_line)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python calc_mcc_iupred.py UP000005640_9606_flat_iupred_long.fasta DisProt_human_fragments_flat.csv")
    else:
        main(sys.argv[1], sys.argv[2])
