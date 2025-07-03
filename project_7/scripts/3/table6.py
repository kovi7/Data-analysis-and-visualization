import pandas as pd
from prettytable import PrettyTable

# data
pdb_data = pd.read_csv("../../data/pdb_stats.csv")

#length + std
avg_length = pdb_data['avg_length'].values[0]
std_length = pdb_data['bootstrap_length_std'].values[0]

#aa
aa_list = [col[1:] for col in pdb_data.columns if col.startswith('%')]
aa_percentages = {aa: pdb_data['%' + aa].values[0] for aa in aa_list}

aa_table = PrettyTable()
aa_table.field_names = ["Average length [aa]", "std"]+aa_list
aa_table.title="Table 6: PBD statistics"


row=[f"{avg_length:.2}", f"{std_length:.2}"]
for aa, percentage in aa_percentages.items():
    row.append(f"{percentage:.2}%")

aa_table.add_row(row)
with open('../../tables/table_6.txt', 'w') as f:
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

with open(f'../../tables/table_6.html', 'w') as f:
    f.write(html_with_style)