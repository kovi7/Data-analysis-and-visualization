import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

# data
swissprot_data = pd.read_csv("../../data/swissprot_stats.csv")
taxonomy_data = pd.read_csv("../../data/taxonomy_stats.csv")

#constans
taxonomies = ['Bacteria', 'Viruses', 'Archaea', 'Eukaryota']
group_colors = {
    'Bacteria': 'red',
    'Viruses': 'green',
    'Archaea': 'orange',
    'Eukaryota': 'blue',
    'full UniProt': 'purple'
}

# joined data frame
combined_data = pd.DataFrame()
swissprot_row = swissprot_data.copy()
swissprot_row['group'] = 'full UniProt'
combined_data = pd.concat([combined_data, swissprot_row], ignore_index=True)
for tax in taxonomies:
    tax_data = taxonomy_data[taxonomy_data['taxonomy_group'] == tax].copy()
    tax_data['group'] = tax
    combined_data = pd.concat([combined_data, tax_data], ignore_index=True)

group_stats = combined_data.groupby('group').agg({
    'avg_length': 'mean',
    'bootstrap_length_std': 'mean'
}).reset_index()


#plot
plt.figure(figsize=(15, 10))
bar_positions = np.arange(len(group_stats['group']))

for i, (_, row) in enumerate(group_stats.iterrows()):
    group = row['group']
    color = group_colors.get(group, 'gray')
    plt.bar(bar_positions[i], row['avg_length'], 
            yerr=row['bootstrap_length_std'], 
            capsize=5, 
            color=color, 
            label=group)

# title, labels etc.
plt.title('Average protein length by group', size=35)
plt.ylabel('Average protein length [aa]', size=25)
plt.xlabel('Group', size=25)
plt.xticks(bar_positions, group_stats['group'])
plt.tick_params(axis='both', labelsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, labels, loc='upper left', fontsize=14)
plt.setp(legend.get_title(), fontsize='xx-large')

plt.tight_layout()
plt.savefig('../../images/fig9.png')


#aa table
aa_list = [col[1:] for col in combined_data.columns if col.startswith('%')]
aa_table = PrettyTable()
aa_table.field_names = ["Group"] + aa_list
aa_table.title="Table 7: Percantage content of amino acid by kingdom"

uniprot_data = combined_data[combined_data['group'] == 'full UniProt']
row = ['full UniProt']
for aa in aa_list:
    row.append(f"{uniprot_data['%' + aa].values[0]:.2}%")
aa_table.add_row(row)


for tax in taxonomies:
    tax_data = combined_data[combined_data['group'] == tax]
    if not tax_data.empty:
        row = [tax]
        for aa in aa_list:
            tax_avg =  tax_data['%' + aa].mean()
            row.append(f"{tax_avg:.2}%")
        aa_table.add_row(row)

with open('../../tables/table_7.txt', 'w') as f:
    f.write(aa_table.get_string())

html_string = aa_table.get_html_string()
html_with_style = f"""
    <style>
    caption {{
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 10px;}}
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th {{
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 8px;
        text-align: left;
    }}
    td {{
        padding: 8px;
        text-align: left;
    }}
    tr:last-child td {{
        border-bottom: 2px solid black;
    }}
</style>
{html_string}
"""

with open(f'../../tables/table_7.html', 'w') as f:
    f.write(html_with_style)
