import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import os
import glob
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
kingdoms = ['Bacteria', 'Archaea', 'Eukaryota', 'Viruses']
kingdom_dirs = {k: f"../../data/raw/reference_proteomes/{k}/" for k in kingdoms}

kingdom_lengths = {k: [] for k in kingdoms}

# processing files
for kingdom, directory in kingdom_dirs.items():
    kingdom_files = glob.glob(os.path.join(directory, "*.fasta"))
    for file in kingdom_files:
        for record in SeqIO.parse(file, "fasta"):
            kingdom_lengths[kingdom].append(len(record.seq))
#medians
kingdom_medians = {k: np.median(lengths) for k, lengths in kingdom_lengths.items()}

formatter = ticker.FuncFormatter(value_format)

#histograms
for kingdom, lengths in kingdom_lengths.items():
    plt.figure(figsize=(15, 10))
    plt.hist(lengths, bins=30, alpha=0.7)
    plt.title(f"Protein length distribution for {kingdom}", size=30)
    plt.xlabel("Protein length [aa]", size=20)
    plt.ylabel("Frequency", size=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.axvline(x=kingdom_medians[kingdom], color='r', linestyle='--', linewidth=2,
               label=f'Median={kingdom_medians[kingdom]:.2f}')
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(f"../../images/{kingdom}_histogram.png")

#boxplots
plt.figure(figsize=(15, 15))
plt.title("Protein length distribution by kingdom", size=30)
plt.xlabel("Kingdom", size=20)
plt.ylabel("Protein length [aa]", size=20, labelpad=10)
plt.tick_params(axis='both', labelsize=15)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_formatter(formatter)

data = [lengths for kingdom, lengths in kingdom_lengths.items()]
labels = list(kingdom_lengths.keys())
colors = plt.cm.tab20(np.linspace(0, 0.5, len(labels)))

box = plt.boxplot(data, labels=labels, patch_artist=True)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig("../../images/taxonomy_boxplot.png")
