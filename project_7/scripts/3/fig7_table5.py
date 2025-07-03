import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

#data
organism = pd.read_csv("../../data/organism_stats.csv")

#constans
organisms = ['E. coli', 'H. sapiens', 'S. cerevisiae', 'A. thaliana', 'D. melanogaster', 'C. elegans', 
             'M. musculus', 'D. rerio', 'B. subtilis']
organism_groups = {
    'Bacteria': ['E. coli', 'B. subtilis'],
    'Vertebrates': ['H. sapiens', 'M. musculus', 'D. rerio'],
    'Invertebrates': ['D. melanogaster', 'C. elegans'],
    'Fungi': ['S. cerevisiae'],
    'Plants': ['A. thaliana']
}
group_colors = {
    'Bacteria': 'red',
    'Vertebrates': 'blue',
    'Invertebrates': 'green',
    'Fungi': 'orange',
    'Plants': 'purple'
}

#plot
plt.figure(figsize=(15, 10))

legend_handles = []
legend_labels = []
bar_positions = np.arange(len(organisms))

for group, group_organisms in organism_groups.items():
    group_data = organism[organism['organism'].isin(group_organisms)]
    
    lengths = []
    errors = []
    positions = []
    
    for org in group_organisms:
        org_data = group_data[group_data['organism'] == org]
        if not org_data.empty:
            lengths.append(org_data['avg_length'].values[0])
            errors.append(org_data['bootstrap_length_std'].values[0] if 'bootstrap_length_std' in org_data.columns else org_data['std_length'].values[0])
            positions.append(bar_positions[organisms.index(org)])
    
    if lengths:
        bars = plt.bar(positions, lengths, yerr=errors, capsize=5, color=group_colors[group], label=group)
        legend_handles.append(bars[0])
        legend_labels.append(group)

# title, labels, etc.
plt.xlabel('Organism', size=25)
plt.ylabel('Average protein length [aa]', size=25)
plt.title('Average protein length by organism', size=35)
plt.xticks(bar_positions, organisms, rotation=45)
plt.tick_params(axis='both', labelsize=20)
legend = plt.legend(legend_handles, legend_labels, title='Taxonomy group', loc='upper left', fontsize=14)
plt.setp(legend.get_title(),fontsize='xx-large')
plt.tight_layout()
plt.savefig('../../images/fig7.png')


#aa table
aa_table = PrettyTable()
aa_list = [col[1:] for col in organism.columns if col.startswith('%')]
aa_table.field_names = ["Organism"] + aa_list
aa_table.title="Table 5: Percantage content of amino acid by organism"
for org in organisms:
    org_data = organism[organism['organism'] == org]
    if not org_data.empty:
        row = [org]
        for aa in aa_list:
            row.append(f"{org_data['%' + aa].values[0]:.2}%")
        aa_table.add_row(row)

with open('../../tables/table_5.txt', 'w') as f:
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

with open(f'../../tables/table_5.html', 'w') as f:
    f.write(html_with_style)