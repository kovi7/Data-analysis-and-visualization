import pandas as pd
from prettytable import PrettyTable

#data
taxonomy = pd.read_csv("../../data/taxonomy_stats.csv")
aa_columns = [col for col in taxonomy.columns if col.startswith("%")]
bootstrap_column = [boot for boot in taxonomy.columns if boot.startswith("bootstrap_std_%")]
organism = pd.read_csv("../../data/organism_stats.csv")

def create_pretty_table(df, title):
    table = PrettyTable()
    table.title=title
    if 'length' in title:
        table.field_names  = ['Index','Average length', 'STD']
        table.float_format = ".2"
        for idx, row in df.iterrows():
                table.add_row([idx] + list(row))
    else:
        table.field_names = ["Index"] + [col.replace("%","") for col in aa_columns]
        for idx, row in df.iterrows():
            modified_row = [idx]
            for aa_col, bootstrap_col in zip(aa_columns, bootstrap_column):
                aa_value = row[aa_col]
                bootstrap_value = row[bootstrap_col]
                modified_row.append(f"{aa_value:.2} ± {bootstrap_value:.2}")
            table.add_row(modified_row)

    # print(f"\n{title}")
    # print(table)
    
    with open(f'../../tables/{title.split(":")[0].lower().replace(" ", "_")}.txt', 'w') as f:
        f.write(table.get_string())

    html_string = table.get_html_string()
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
    
    with open(f'../../tables/{title.split(":")[0].lower().replace(" ", "_")}.html', 'w') as f:
        f.write(html_with_style)

# Table 1
table1 = organism.groupby("organism")[aa_columns + bootstrap_column].mean()
create_pretty_table(table1, "Table 1: Average amino acid content by organism")

# Table 2
cols = ["avg_length","bootstrap_length_std"]
table2 = organism.groupby("organism")[cols].mean()
create_pretty_table(table2, "Table 2: Protein length by organism")

# Table 3
table3 = taxonomy.groupby("taxonomy_group")[aa_columns + bootstrap_column].mean()
create_pretty_table(table3, "Table 3: Average amino acid content by taxonomy")

# Table 4
cols = ["avg_length","bootstrap_length_std"]
table4 = taxonomy.groupby("taxonomy_group")[cols].mean()
create_pretty_table(table4, "Table 4: Protein length by taxonomy")
