import matplotlib.pyplot as plt
import pandas as pd

#data
pdb = pd.read_csv("../../data/pdb_stats.csv")
swissprot = pd.read_csv("../../data/swissprot_stats.csv")

#features
joined= pd.concat((pdb, swissprot),axis=0)
colors = ['red','green']
avg_lengths = joined.groupby('data_type')['avg_length'].mean().to_list()
xticks = ['PDB',"Uniprot"]

#plot
plt.figure(figsize=(15, 10))
plt.bar(xticks, avg_lengths, capsize=5, color = colors)
plt.ylabel('Average protein length [aa]', size = 25, labelpad=10)
plt.title('Average protein length in selected databases', size = 30, pad=10)
plt.xlabel('Protein database', size = 25)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('../../images/fig5.png')
plt.close()
