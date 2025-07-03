import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#data
taxonomy = pd.read_csv("../../data/taxonomy_stats.csv")

#feature
tax = list(taxonomy['taxonomy_group'].unique())
tax.sort()
colors = plt.cm.tab20(np.linspace(0, 0.5, len(tax)))
avg_lengths = taxonomy.groupby('taxonomy_group')['avg_length'].mean().tolist()


#plot
plt.figure(figsize=(15, 10))
plt.bar(tax, avg_lengths, capsize=5, color = colors)
plt.title('Average protein length in selected taxonomy groups', size = 30, pad=10)
plt.ylabel('Average protein length [aa]', size = 25, labelpad=10)
plt.xlabel('Taxonomy group', size = 25, labelpad=10)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('../../images/fig6.png')
plt.close()

