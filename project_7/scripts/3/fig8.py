import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
organism = pd.read_csv("../../data/organism_stats.csv")

#constans
selected_orgs = ['E. coli', 'H. sapiens', 'S. cerevisiae']
aa_list = [col[1:] for col in organism.columns if col.startswith('%')]

#plot
plt.figure(figsize=(15, 10))

bar_width = 0.25
index = np.arange(len(aa_list))
for i, org in enumerate(selected_orgs):
    org_data = organism[organism['organism'] == org]
    if not org_data.empty:
        aa_values = [org_data['%' + aa].values[0] for aa in aa_list]
        plt.bar(index + i*bar_width, aa_values, bar_width, label=org)

plt.xlabel('Amino Acid', size=20)
plt.ylabel('Percentage (%)', size=20)
plt.title('Amino acid composition in selected organisms', size=30)
plt.xticks(index + bar_width, aa_list)
plt.tick_params(axis='both', labelsize=15)
plt.legend(loc='upper left', fontsize = 14)
plt.savefig('../../images/fig8.png')
