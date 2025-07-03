import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#data
organism = pd.read_csv("../../data/organism_stats.csv")

#features
names = list(organism['organism'])
names.sort()
colors = plt.cm.tab20(np.linspace(0, 0.5, len(names)))
avg_lengths = organism.groupby('organism')['avg_length'].mean().tolist()

#plot
plt.figure(figsize=(15, 10))
plt.bar(names, avg_lengths, capsize=5, color = colors)
plt.ylabel('Average protein length', size = 20)
plt.title('Average protein length in selected organisms', size = 25, pad=10)
plt.xlabel('Organism', size = 20)
plt.tick_params(axis='both', labelsize=15)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig('../../images/fig4.png')
