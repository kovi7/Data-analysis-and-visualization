import matplotlib.pyplot as plt
import pandas as pd

#loading data
df_final = pd.read_csv('../../data/dataset1.csv')

#preparing data
total_metric = df_final.groupby('method')['metric'].sum()

#plot
plt.figure(figsize=(15,10))
plt.pie(total_metric.values, labels=total_metric.index, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.title('Distribution of structural data across\ndifferent techniques in PDB (2005–2024)', fontsize=30)
plt.savefig('../../images/fig3.png')
plt.set_cmap('tab20c')
# plt.show()

