import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
IMAGE_DIR = "../../images"


def analyze_n_terminus_frequencies(file):
    stats_df = pd.read_csv(file)
    
    # N-terminus frequency columns
    n_term_columns = [col for col in stats_df.columns if col.startswith("N_term_%")]
    
    n_term_freqs = stats_df[n_term_columns].mean()
    
    freq_data = []
    for col in n_term_columns:
        aa = col.replace("N_term_%", "")
        freq_data.append({"amino_acid": aa, "frequency": n_term_freqs[col]})
    
    freq_df = pd.DataFrame(freq_data)
    freq_df = freq_df.sort_values("frequency", ascending=False)
    
    # the most frequent amino acid
    top_aa = freq_df.iloc[0]
    most_frequent_aa = top_aa["amino_acid"]
    max_frequency = top_aa["frequency"]
    
    print(f"Most frequent N-terminal amino acid for {file}: {most_frequent_aa} ({max_frequency:.2f}%)")

if __name__ == "__main__":
    analyze_n_terminus_frequencies('../../data/organism_stats.csv')
    analyze_n_terminus_frequencies('../../data/taxonomy_stats.csv')
    analyze_n_terminus_frequencies('../../data/pdb_stats.csv')
    analyze_n_terminus_frequencies('../../data/swissprot_stats.csv')
