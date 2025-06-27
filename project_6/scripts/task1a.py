from Bio import SeqIO
import re
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#Our task is to classify proteins/peptides into acidic (pI < 5) and non-acidic (pI >= 5).

'''
There are two main datasets:
- IPC_peptide (16,882 items)
- IPC_protein (2,324 items)

Since we aim for binary classification, we will label data based on a threshold of pI = 5.0:
- pI < 5 --> acidic (label: 1)
- pI >= 5 --> non-acidic (label: 0)
'''

class Parser():
    def parse_peptide(file_path):
        records = []  
        
        with open(file_path, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                header = record.description
                sequence = str(record.seq)
                length = len(sequence)

                pI_match = re.search(r"mean exp pI:\s([0-9.]+)", header)
                pI = float(pI_match.group(1)) if pI_match else None

                label = 1 if pI and pI < 5 else 0

                record_data = {
                    'header': header,
                    'pI': pI,
                    'label': label,
                    'sequence': sequence,
                    'length': length
                }
                records.append(record_data)
        
        return records


    def parse_protein(file_path):
        records = []

        with open(file_path, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                header = record.description
                sequence = str(record.seq)
                length = len(sequence)


                pI_match_case1 = re.findall(r"'([0-9.]+)\/", header)
                pI_match_case2 = re.findall(r"\s([0-9.]+)", header)

                if pI_match_case1:
                    pI_values = [float(p) for p in pI_match_case1]
                    pI = sum(pI_values) / len(pI_values)
                elif pI_match_case2:
                    pI_values = [float(p) for p in pI_match_case2]
                    pI = sum(pI_values) / len(pI_values)
                else:
                    pI = None

                label = 1 if pI and pI < 5 else 0

                record_data = {
                    'header': header,
                    'pI': pI,
                    'label': label,
                    'sequence': sequence,
                    'length': length
                }
                records.append(record_data)

        return records


def plot_hist(data, name):
    quantiles_to_compute = [25, 75]
    quantiles = np.percentile(
        data,
        quantiles_to_compute
    )

    plt.figure(figsize=(10,6))
    path = f"results/plots/plot_{name}.png"
    _, bins, patches = plt.hist(data, bins=20, edgecolor='black', linewidth=1)

    for i, patch in enumerate(patches):
        if bins[i] < quantiles[0]:
            patch.set_facecolor('blue')
        elif bins[i] < quantiles[1]:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')
    if 'len' in name:
        plt.title('Histogram of sequence length')
        plt.xlabel('Sequence length')
    else:
        plt.title('Histogram of mean experimental pI values')
        plt.xlabel('Mean exp pI')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(quantiles[0], color='black', linestyle='--', label=f'Q1 = {quantiles[0]:.2f}')
    plt.axvline(quantiles[1], color='black', linestyle='--', label=f'Q3 = {quantiles[1]:.2f}')
    plt.legend()
    plt.savefig(path)


def main():

    p = Parser
    peptide = Parser.parse_peptide('data/Gauci_PHENYX_SEQUEST_0.99_duplicates_out.fasta')
    protein = p.parse_protein('data/pip_ch2d19_2_1st_isoform_outliers_3units_cleaned_0.99.fasta')

    pI_peptide = [record['pI'] for record in peptide if record['pI'] is not None]
    len_peptide = [record['length'] for record in peptide if record['pI'] is not None]
    pI_protein = [record['pI'] for record in protein if record['pI'] is not None]
    len_protein = [record['length'] for record in protein if record['pI'] is not None]

    plot_hist(pI_peptide, "peptide_pI")
    plot_hist(len_peptide, 'peptide_len')
    plot_hist(pI_protein, 'protein_pI')
    plot_hist(len_protein, 'protein_len')


if __name__ == "__main__":
    main()