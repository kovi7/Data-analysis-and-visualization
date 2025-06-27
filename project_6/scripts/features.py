import csv
from collections import Counter
import hashlib
from Bio import SeqIO

def load_dataset(file_path):
    records = []
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            header = record.description
            sequence = str(record.seq)
            
            parts = header.split('|')
            uid = parts[0][1:]  
            source_type = parts[1]
            pI = float(parts[2])
            label = int(parts[3])
            
            records.append({
                'uid': uid,
                'source_type': source_type,
                'pI': pI,
                'label': label,
                'sequence': sequence
            })
    return records

def compute_hydrophobicity(seq):
    '''Hydrophobicity: Kyte-Doolittle scale'''
    kd_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    values = [kd_scale.get(aa.upper(), 0.0) for aa in seq]
    total = round(sum(values), 3)
    mean = round(total / len(seq), 3) if seq else 0.0
    return {'hydro_total': total, 'hydro_mean': mean}

def compute_aa_counts(seq):
    '''Amino acid counts'''
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    counts = Counter(seq.upper())
    return {f'aa_{aa}': counts.get(aa, 0) for aa in aa_list}

def compute_molecular_weight(seq):
    """Molecular weight calculation with error handling"""
    massDict = {
        'D':115.0886, 'E': 129.1155, 'C': 103.1388, 'Y':163.1760, 'H':137.1411, 
        'K':128.1741, 'R': 156.1875,  'M': 131.1926, 'F':147.1766, 'L':113.1594,
        'V':99.1326,  'A':  71.0788,  'G':  57.0519, 'Q':128.1307, 'N':114.1038, 
        'I':113.1594, 'W': 186.2132,  'S':  87.0782, 'P': 97.1167, 'T':101.1051, 
        'U':141.05,   'h2o':18.01524, 'X':        0, 'Z':128.6231, 'O':255.31, 
        'B':114.5962, 'J': 113.1594,
    }
    molecular_weight = massDict['h2o']
    for aa in seq.upper():
        molecular_weight += massDict.get(aa, 0.0)  # Handle unknown AAs
    return round(molecular_weight, 2)

def compute_charge(seq, pH=7.0, scale='Sillero'):
    '''Charge computation with pKa scale selection'''
    scales = {
        "EMBOSS":     {'Cterm': 3.6, 'pKAsp': 3.9,  'pKGlu': 4.1, 'pKCys': 8.5, 'pKTyr': 10.1, 'pkHis': 6.5, 'Nterm': 8.6, 'pKLys': 10.8, 'pKArg': 12.5},
        "DTASelect":  {'Cterm': 3.1, 'pKAsp': 4.4,  'pKGlu': 4.4, 'pKCys': 8.5, 'pKTyr': 10.0, 'pkHis': 6.5, 'Nterm': 8.0, 'pKLys': 10.0, 'pKArg': 12.0},
        "Solomon":    {'Cterm': 2.4, 'pKAsp': 3.9,  'pKGlu': 4.3, 'pKCys': 8.3, 'pKTyr': 10.1, 'pkHis': 6.0, 'Nterm': 9.6, 'pKLys': 10.5, 'pKArg': 12.5},
        "Sillero":    {'Cterm': 3.2, 'pKAsp': 4.0,  'pKGlu': 4.5, 'pKCys': 9.0, 'pKTyr': 10.0, 'pkHis': 6.4, 'Nterm': 8.2, 'pKLys': 10.4, 'pKArg': 12.0},
        "Rodwell":    {'Cterm': 3.1, 'pKAsp': 3.68, 'pKGlu': 4.25,'pKCys': 8.33,'pKTyr': 10.07,'pkHis': 6.0, 'Nterm': 8.0, 'pKLys': 11.5, 'pKArg': 11.5},
        "Patrickios": {'Cterm': 4.2, 'pKAsp': 4.2,  'pKGlu': 4.2, 'pKCys': 0.0, 'pKTyr':  0.0, 'pkHis': 0.0, 'Nterm': 11.2,'pKLys': 11.2, 'pKArg': 11.2},
        "Wikipedia":  {'Cterm': 3.65,'pKAsp': 3.9,  'pKGlu': 4.07,'pKCys': 8.18,'pKTyr': 10.46,'pkHis': 6.04,'Nterm': 8.2, 'pKLys': 10.54,'pKArg': 12.48},
        "Grimsley":   {'Cterm': 3.3, 'pKAsp': 3.5,  'pKGlu': 4.2, 'pKCys': 6.8, 'pKTyr': 10.3, 'pkHis': 6.6, 'Nterm': 7.7, 'pKLys': 10.5, 'pKArg': 12.04},
        'Lehninger':  {'Cterm': 2.34,'pKAsp': 3.86, 'pKGlu': 4.25,'pKCys': 8.33,'pKTyr': 10.0, 'pkHis': 6.0, 'Nterm': 9.69,'pKLys': 10.5, 'pKArg': 12.4},
        'Bjellqvist': {'Cterm': 3.55,'pKAsp': 4.05, 'pKGlu': 4.45,'pKCys': 9.0, 'pKTyr': 10.0, 'pkHis': 5.98,'Nterm': 7.5, 'pKLys': 10.0, 'pKArg': 12.0},
        'Toseland':   {'Cterm': 3.19,'pKAsp': 3.6,  'pKGlu': 4.29,'pKCys': 6.87,'pKTyr': 9.61, 'pkHis': 6.33,'Nterm': 8.71, 'pKLys': 10.45, 'pKArg':  12},
        'Thurlkill':  {'Cterm': 3.67,'pKAsp': 3.67, 'pKGlu': 4.25,'pKCys': 8.55,'pKTyr': 9.84, 'pkHis': 6.54,'Nterm': 8.0, 'pKLys': 10.4, 'pKArg': 12.0},
        'Nozaki':     {'Cterm': 3.8, 'pKAsp': 4.0,  'pKGlu': 4.4, 'pKCys': 9.5, 'pKTyr': 9.6,  'pkHis': 6.3, 'Nterm': 7.5, 'pKLys': 10.4, 'pKArg': 12},   
        'Dawson':     {'Cterm': 3.2, 'pKAsp': 3.9,  'pKGlu': 4.3, 'pKCys': 8.3, 'pKTyr': 10.1, 'pkHis': 6.0, 'Nterm': 8.2, 'pKLys': 10.5, 'pKArg':  12},   
        'IPC_peptide':{'Cterm': 2.383, 'pKAsp': 3.887, 'pKGlu': 4.317, 'pKCys': 8.297, 'pKTyr': 10.071, 'pkHis': 6.018, 'Nterm': 9.564, 'pKLys': 10.517, 'pKArg': 12.503},
        'IPC_protein': {'Cterm': 2.869, 'pKAsp': 3.872, 'pKGlu': 4.412, 'pKCys': 7.555, 'pKTyr': 10.85, 'pkHis': 5.637, 'Nterm': 9.094, 'pKLys': 9.052,  'pKArg': 11.84}
    }
    
    pKa = scales.get(scale)
    if not pKa:
        raise ValueError(f"Invalid scale: {scale}")

    seq = seq.upper()
    counts = Counter(seq)

    # Positive charges
    pos = (
        (10 ** pKa['Nterm']) / (10 ** pKa['Nterm'] + 10 ** pH) +
        counts['K'] * (10 ** pKa['pKLys']) / (10 ** pKa['pKLys'] + 10 ** pH) +
        counts['R'] * (10 ** pKa['pKArg']) / (10 ** pKa['pKArg'] + 10 ** pH) +
        counts['H'] * (10 ** pKa['pkHis']) / (10 ** pKa['pkHis'] + 10 ** pH)
    )

    # Negative charges
    neg = (
        (10 ** pH) / (10 ** pKa['Cterm'] + 10 ** pH) +
        counts['D'] * (10 ** pH) / (10 ** pKa['pKAsp'] + 10 ** pH) +
        counts['E'] * (10 ** pH) / (10 ** pKa['pKGlu'] + 10 ** pH) +
        counts['C'] * (10 ** pH) / (10 ** pKa['pKCys'] + 10 ** pH) +
        counts['Y'] * (10 ** pH) / (10 ** pKa['pKTyr'] + 10 ** pH)
    )

    return round(pos - neg, 3)

def extract_features(record):
    '''Extract all features for a single record'''
    try:
        seq = record['sequence']
        pI = record['pI']
        data_type = record['source_type']
        label = record['label']
        
        # Generate UID from sequence
        uid = hashlib.md5(seq.encode()).hexdigest()
        
        # Basic features
        features = {
            'uid': uid,
            'data_type': data_type,
            'pI': pI,
            'length': len(seq),
            'molecular_weight': compute_molecular_weight(seq),
            'label': label,
        }
        
        # Add amino acid counts
        features.update(compute_aa_counts(seq))
        
        # Add hydrophobicity features
        features.update(compute_hydrophobicity(seq))
        
        # Add charge features for all scales
        charge_scales = ["Sillero", "EMBOSS", "DTASelect", "Solomon", "Rodwell"]
        for scale in charge_scales:
            features[f'charge_{scale}'] = compute_charge(seq, scale=scale)
        
        return features
    
    except Exception as e:
        print(f"Error processing sequence {record.get('uid', 'unknown')}: {e}")
        return None

def export_to_csv(records, filename):
    if not records:
        print("No records to write.")
        return
    
    # Filter out None values
    valid_records = [r for r in records if r is not None]
    
    if not valid_records:
        print("No valid records to write.")
        return
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=valid_records[0].keys())
        writer.writeheader()
        writer.writerows(valid_records)

if __name__ == "__main__":
    # Load datasets
    train_data = load_dataset("data/IPC_classification_dataset_60_train.fasta")
    test_data = load_dataset("data/IPC_classification_dataset_20_test.fasta")
    val_data = load_dataset("data/IPC_classification_dataset_20_val.fasta")
    
    # Process features with error handling
    train_features = [extract_features(record) for record in train_data]
    test_features = [extract_features(record) for record in test_data]
    val_features = [extract_features(record) for record in val_data]
    
    # Export to CSV
    export_to_csv(train_features, "data/IPC_classification_features_train.csv")
    export_to_csv(test_features, "data/IPC_classification_features_test.csv")
    export_to_csv(val_features, "data/IPC_classification_features_val.csv")
