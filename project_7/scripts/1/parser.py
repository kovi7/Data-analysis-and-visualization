import pandas as pd
import re

protein_residues = {}
with open('../../data/raw/pdb_seqres.txt', 'r') as file:
    for line in file:
        if line.startswith('>') and 'mol:protein' in line:
            match = re.match(r'^>(\w+)_\w+\s+mol:protein\s+length:(\d+)', line)
            if match:
                pdb_id = match.group(1).lower()
                num_residues = int(match.group(2))
                protein_residues[pdb_id] = num_residues

entries_data = {}
with open('../../data/raw/entries.idx', 'r') as file:
    for line in file:
        if re.match(r'^\w{4}\s', line):
            parts = [p.strip() for p in re.split(r'\t+', line.strip()) if p.strip()]
            date_idx = next((i for i, p in enumerate(parts) if re.match(r'\d{2}/\d{2}/\d{2}', p)), None)
            if date_idx is None or len(parts) < date_idx + 3:
                continue
            pdb_id = parts[0].lower()
            date = parts[date_idx]
            resolution_str = parts[-2]
            methods_raw = parts[-1].upper().split('/')
            try:
                resolution = float(resolution_str)
            except ValueError:
                resolution = None
            entries_data[pdb_id] = {'date': date, 'resolution': resolution, 'methods': methods_raw}

protein_only_entries = set()
with open('../../data/raw/pdb_entry_type.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 3 and parts[1] == 'prot':
            protein_only_entries.add(parts[0].lower())

combined_data = []
for pdb_id in protein_only_entries:
    if pdb_id in protein_residues and pdb_id in entries_data:
        num_residues = protein_residues[pdb_id]
        date_str = entries_data[pdb_id]['date']
        year_suffix = int(date_str.split('/')[-1])
        year = 1900 + year_suffix if year_suffix > 25 else 2000 + year_suffix
        methods_raw = entries_data[pdb_id]['methods']
        resolution = entries_data[pdb_id]['resolution']

        for method_raw in methods_raw:
            if 'NMR' in method_raw:
                method, effective_resolution = 'NMR', 2.5
            elif 'ELECTRON MICROSCOPY' in method_raw:
                method, effective_resolution = 'ELECTRON MICROSCOPY', resolution
                if effective_resolution is None: continue
            elif 'X-RAY DIFFRACTION' in method_raw or method_raw == '':
                method, effective_resolution = 'X-RAY DIFFRACTION', resolution
                if effective_resolution is None: continue
            else:
                continue 

            if effective_resolution and effective_resolution != 0:
                metric = num_residues * (1/ effective_resolution)
            else:
                metric = None

            combined_data.append({
                'pdb_id': pdb_id,
                'year': year,
                'method': method,
                'metric': metric
            })

df_final = pd.DataFrame(combined_data)
df_final.to_csv('../../data/dataset1.csv', index=False)
