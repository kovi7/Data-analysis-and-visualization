"""
Exercise 3

a) E.coli, human, yeast, A. thaliana, D. melanogaster, C. elegans, Mouse,
Zebrafish (D. rerio), Bacillus subtilis

Prepare the barplot (matplotlib, seaborn, plotly) showing protein length for all 9 organisms:
- x,y axes should have a description
- aggregate the bars for the same group with different color (vertebrates one
color, bacteria second color, etc.)
- add the legend (upper-left corner)
- add the error bar to each bar

Calculate the percentage content of all amino acids and prepare the table (PrettyTable module).
Additionally, prepare bar plot for the percentage content of all amino acids for E.coli,
human, yeast (thus group three bars for each amino acid).

b) PDB
- calculate the average length of protein and percentage content of all amino acids (just numbers)

Compare the result with point (a). Can you explain the difference
(hint: open ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt in the text editor)?

c) UniProt
- full UniProt (Swiss-Prot)
- 100 randomly selected Bacteria
- 100 randomly selected Viruses
- 100 randomly selected Archaea
- 100 randomly selected Eukaryota

Prepare similar box plots and table as in (a).

d) data exploration:
- for each organism (a) and kingdom (c) make a separate histogram for protein length
- calculate and plot median instead arithmetic mean
- instead bar plots, use the "boxplot" function (only protein length)
https://en.wikipedia.org/wiki/Box_plot

Discuss which is better: median or arithmetic mean (prons and cons)?

================================================

Homework
Prepare a short report (pdf) containing all plots, tables, and answers to the above questions. Additionally, 
add also scripts for generating the plots and tables. 
"""