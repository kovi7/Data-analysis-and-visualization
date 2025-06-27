import random
from Bio import SeqIO

'''
Now split the data into training, testing, and validation sets (60/20/20):

First iteration:
- Randomly split the data. Count how many items in each split come from proteins/peptides and how many are positives/negatives.

Due to dataset size, proportions may slightly vary.

Second iteration:
- To ensure balanced source types, split proteins and peptides separately, then randomly divide and merge the splits.

Compare the distributions between the two methods.

Save the splits:
    IPC_classification_dataset_60_train.fasta
    IPC_classification_dataset_20_test.fasta
    IPC_classification_dataset_20_val.fasta
'''

def load_dataset(file_path):
    records = []
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            header = record.description
            sequence = str(record.seq)
            
            parts = header.split('|')
            uid = parts[0][1:]  # Skipping '>'
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

def random_split(data):
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    train_idx = int(0.6 * len(data_copy))
    test_idx = int(0.8 * len(data_copy))
    
    train = data_copy[:train_idx]
    test = data_copy[train_idx:test_idx]
    val = data_copy[test_idx:]
    
    return train, test, val

def save_fasta_dataset(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            header = f">{item['uid']}|{item['source_type']}|{item['pI']}|{item['label']}"
            f.write(f"{header}\n{item['sequence']}\n")

def analyze_split(split_data, split_name):
    total = len(split_data)
    proteins = sum(1 for item in split_data if item['source_type'] == 'protein')
    peptides = sum(1 for item in split_data if item['source_type'] == 'peptide')
    positives = sum(1 for item in split_data if item['label'] == 1)
    negatives = sum(1 for item in split_data if item['label'] == 0)
    
    print(f"{split_name} set: {total} items")
    print(f"  - Proteins: {proteins} ({proteins/total*100:.2f}%)")
    print(f"  - Peptides: {peptides} ({peptides/total*100:.2f}%)")
    print(f"  - Positives (acidic): {positives} ({positives/total*100:.2f}%)")
    print(f"  - Negatives (non-acidic): {negatives} ({negatives/total*100:.2f}%)")

def main():
    data = load_dataset("data/IPC_classification_dataset_100.fasta")
    
    train1, test1, val1 = random_split(data)
    analyze_split(train1, "Training")
    analyze_split(test1, "Testing")
    analyze_split(val1, "Validation")
    
    save_fasta_dataset(train1, "data/IPC_classification_dataset_60_train.fasta")
    save_fasta_dataset(test1, "data/IPC_classification_dataset_20_test.fasta")
    save_fasta_dataset(val1, "data/IPC_classification_dataset_20_val.fasta")
    
if __name__ == "__main__":
    main()
