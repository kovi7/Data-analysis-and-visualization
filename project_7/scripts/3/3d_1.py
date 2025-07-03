import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import os
import matplotlib.ticker as ticker

def value_format(value, pos):
    if value >= 1e9:
        return f'{value*1e-9:.1f}B'
    elif value >= 1e6:
        return f'{value*1e-6:.1f}M'
    elif value >= 1e3:
        return f'{value*1e-3:.1f}K'
    else:
        return f'{value:.0f}'

#constans
official_names = {
    'Human': 'H. sapiens',
    'Yeast': 'S. cerevisiae',
    'Mouse': 'M. musculus',
    'Zebrafish': 'D. rerio',
}

organisms = ['E._coli','Human','Yeast','A._thaliana', 'D._melanogaster', 'C._elegans', 'Mouse', 'Zebrafish', 'B._subtilis']

# load data
organism_files = [f"../../data/raw/reference_proteomes/{organism}.fasta" for organism in organisms]

#processing files
all_lengths = {}
all_medians = {}
for file in organism_files:
    organism_name = os.path.splitext(os.path.basename(file))[0]
    lengths=[]
    for record in SeqIO.parse(file, "fasta"):
        lengths.append(len(record.seq))

    organism_name = organism_name.replace("_", " ")
    if organism_name in official_names:
        organism_name = official_names[organism_name]

    all_medians[organism_name] = np.median(lengths)
    all_lengths[organism_name] = lengths

formatter = ticker.FuncFormatter(value_format)


#histograms
for i, (organism, lengths) in enumerate(all_lengths.items()):
    plt.figure(figsize=(15, 10))
    plt.hist(lengths, bins=30, alpha=0.7)
    plt.title(f"Protein length distribution for {organism}", size = 30)
    plt.xlabel("Protein length[aa]", size = 20)
    plt.ylabel("Frequency", size = 20)
    plt.tick_params(axis='both', labelsize=15)
    plt.axvline(x=all_medians[organism], color='r', linestyle='--', linewidth = 2,
                   label=f'Median={all_medians[organism]:.2f}')
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(f"../../images/{organism.replace(" ","_")}_histogram.png")

#boxplot
plt.figure(figsize=(15, 15))
plt.title("Protein length distribution by organism", size =30)
plt.xlabel("Organism", size=20)
plt.ylabel("Protein length[aa]", size = 20)
plt.tick_params(axis='both', labelsize=15)
plt.xticks(rotation=35)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_formatter(formatter)

data = []
labels = []
for organism, lengths in all_lengths.items():
    data.append(lengths)
    labels.append(organism)
colors = plt.cm.tab20(np.linspace(0, 0.5, len(labels)))
box = plt.boxplot(data, labels=labels, patch_artist=True)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig("../../images/organisms_boxplot.png")