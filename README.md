# Data Analysis & Visualization Projects


This repository contains a collection of Python-based projects focused on data analysis and animated data visualization. The goal is to present clear, insightful, and visually appealing graphics using real-world datasets.



## Projects Overview

| Project | Title                        | Description                                                             |
|---------|------------------------------|-------------------------------------------------------------------------|
| 1       | Global Population Analysis   | Animated and static visualizations based on World Bank population data |
| 2       | Global Temperature Trends    | Historical climate analysis using time series and distribution plots   |
| 3       | The Importance of Plotting   | Visual diagnostics on datasets with identical statistics but different visuals (Anscombe datasets)  |
| 4       | Interactive Plots            | Web-ready, interactive visualizations (Plotly, Bokeh, mbek)              |
| 5       | Time Series Forecasting      | Forecasting country temperatures up to the year 2263                    |
| 6       | Isoelectric Point Classifier | Protein/peptide acidity classification via ML and DL                    |
| 7       | The Importance of (Big) Data | Structural & sequence analysis using PDB and UniProt data              |
| 8       | Human Genome Analysis                | GC content & ORF analysis for full human genome using FASTA files      |
| 9       | High-Throughput Protein Feature Prediction | Parallel prediction of disorder and secondary structure using IUPred & ProtBert |


---

## 🌍 Project 1: Global Population Analysis


Using [World Bank population data](https://data.worldbank.org/indicator/SP.POP.TOTL?end=2023&start=1960) & [UW schedule 25/26](https://monitor.uw.edu.pl/Lists/Uchway/Attachments/7221/M.2025.15.Post.1.pdf):

- **Animated bar/line/bubble plots** (color & B&W) for:
  - Top 5 most populated countries
  - Chile/Poland with 4 closest neighbors (by population)
- **Contextual animations**: highlight events (genocity in Cambodia)
- **Gantt charts**: event timelines from PDF academic document, in color and grayscale
- All visuals use `matplotlib.animation` and are saved as `.gif` and `.pdf`

📁 `project_1/`

---


## 🌡️ Project 2: Global Temperature Trends

Based on temperature records (1750–2023) for multiple cities and countries. Source [global temperature data](https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/temperature.csv) available in data folder.

- **Data cleaning**: conversion to Celsius, NA removal, column simplification
- **Scatter, box, and violin plots**
- **Grouped time series plots**: multi-subplots, color-coded per country and city

Deliverables:
- `.py` scripts for each plot (`task2a.py`, ..., `task5e.py`)
- Final HTML report with plots only 
- All plots exported as `.png`

📁 `project_2/`

---

## 📉 Project 3: The Importance of Plotting

Inspired by the famous Anscombe's Quartet and its modern extensions, this project demonstrates how datasets with **identical summary statistics** can look dramatically different.

- Dataset `ans.csv`: classic Anscombe's Quartet in one composite plot
- Dataset `ans2.tsv`: multi-subplot visualization for hidden distributions
- Calculations include: mean, variance, correlation, regression line, R²
- Jupyter Notebook with analysis, plots, and tables
- Rendered HTML report with all visuals

🔗 Suggested reading: [Same Stats, Different Graphs](https://www.autodeskresearch.com/publications/samestats)

📁 `project_3/`

---

## 🧭 Project 4: Interactive Plots

Exploration of interactive Python visualization libraries with a focus on **cross-library consistency**.

### Libraries used:
- `mpld3`
- `pygal`
- `bokeh`

### Task:
For **each library**, two **different types of plots** were implemented:

- **Scatter plot**
- **Boxplot**

This results in a total of **6 interactive plots** (3 scatter plots + 3 boxplots).  
Each library recreates the **same two plots** using its own syntax and tools, while maintaining **similar structure, layout, labels, colors, and styling** — to allow for visual comparison across technologies.

🎯 The goal was to evaluate capabilities, strengths, and limitations of each library in rendering the same visual concept.

Each script supports a positional parameter `[0/1]`:
```bash
python mpld3_scatter.py 0   # show interactive plot
python mpld3_scatter.py 1   # save the plot (HTML)
```
📁 `project_4/`

---

## 🔮 Project 5: Time Series Forecasting
Forecasting country-level temperatures for the next 250 years, based on cleaned data from Project 2.

- **Data aggregation**: calculate average annual temperature for 8 countries
- **initial visualizations**: 8 scatter plots (one per country) to identify trends and seasonality
- **Forecasting**: for Japan/France/Brazil/New Zealand/Poland/Sweden/Ukraine 3 forecasting models were used including 95% confidence intervals in all predictions

### Forecasting methods used:
- `Autoregression model(AR)`
- `Autoregressive integrated moving average (ARIMA)`
- `Holt-Winters expotential smoothing(HWES)`


### Cross-validation results (MAE):
| Country   |     AR |   ARIMA |   HWES |
|-----------|--------|---------|--------|
| Brazil    | 0.3673 |  0.3932 | 0.3621 |
| France    | 0.5431 |  0.5875 | 0.5801 |

📁 `project_5/`

---

## 🧪 Project 6: Isoelectric Point Classifier
Model to classify proteins/peptides as **acidic** (pI < 5.0) or **non-acidic** using features extracted from sequences.

Source:
- [IPC Protein](http://isoelectric.org/datasets/pip_ch2d19_2_1st_isoform_outliers_3units_cleaned_0.99.fasta)
- [IPC Peptide](http://isoelectric.org/datasets/Gauci_PHENYX_SEQUEST_0.99_duplicates_out.fasta)

### Preprocess:
- Clean and analyze raw `.fasta` sequences
- Visualize pI and sequence length (histograms, IQRs)
- Extract:
  - 550 acidic proteins (pI < 5.0)
  - 550 basic proteins (pI > 10.0)
  - 550 acidic peptides
  - 550 the most basic peptides
- Merge and label into unified FASTA:
``` bash
uid|protein/peptide|pI|label
SEQUENCE
```
- Output file: `IPC_classification_dataset_100.fasta`

---

### Features:

| Feature             | Type     | Description                                         |
|---------------------|----------|-----------------------------------------------------|
| `seq_length`        | Integer  | Sequence length                                     |
| `mw`                | Float    | Molecular weight (Da)                              |
| `avg_hydrophobicity`| Float    | Kyte-Doolittle mean hydrophobicity                 |
| `fraction_A`–`fraction_Y` | Float | Relative amino acid composition                    |
| `num_acidic`        | Integer  | Count of D and E residues                          |
| `num_basic`         | Integer  | Count of K, R, H residues                          |
| `charge_at_pH7`     | Float    | Estimated net charge at pH 7                       |
| `aromaticity`       | Float    | Proportion of F, Y, W residues                     |
| `aliphatic_index`   | Float    | Aliphatic side chain volume index                 |
| `instability_index` | Float    | Sequence stability index                           |
| `pI`                | Float    | Calculated isoelectric point                       |
| `label_acidic`      | Binary   | 1 = acidic, 0 = non-acidic                         |

Split sets into:
- `IPC_classification_dataset_60_train.fasta`
- `IPC_classification_dataset_20_test.fasta`
- `IPC_classification_dataset_20_val.fasta`

---

### Data exploration
 - Histograms, scatter plots, boxplots, and heatmaps for features
 - Feature selection and correlation analysis

### Classical ML methods
 - `DecisionTreeClassifier`:
 - `RandomForestClassifier`
 - `KNeighborsClassifier`
 - `SVMClassifier + GridSearchCV`

**Model Evaluation**:
 - Accuracy, precision, recall, F1-score tables
 - Confusion matrices

---

### Deep Learning Model

Using `PyTorch`:

- Several Dense Neural Networks were trained, experimenting with:
  - Different numbers of hidden layers and units
  - Activation functions (ReLU, GeLu)
- Most models achieved high classification performance, indicating that the selected features are well-suited for the task
- Best-performing models were saved in both .json (architecture) and .h5 (weights) formats.

#### Inference Script:

```bash
python acidic_protein_dl_predictor.py -i input.fasta -o output_pred.fasta
```

Example output format:
```bash
>AF-A0A782FH33 | non-acidic 0.13989039
MWRVRIFFGKRQTCAFWLCVTGTCASTMPISERHRAMKGDSIDVVNGRRLPGYGLCIKNKPV
```
📁 `project_6/`

---


## 🧠 Project 7: The Importance of (Big) Data

Exploreation of structural and sequence data from [**PDB**](https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz) and [**UniProt**](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/)

- **Animated bar & pie charts**: structure techniques over time (2005–2024)
- **Organism/kingdom analysis**: average protein length, amino acid composition
- **N-terminus study**: most frequent residues
- **Other visuals**: bar plots, boxplots, histograms
- Includes statistical summaries with bootstrapped errors

📁 `project_7/`

---
## 🧬 Project 8: Human Genome Analysis

#### Genome reference:
- [T2T paper (2021)](https://www.biorxiv.org/content/10.1101/2021.05.26.445798v1)
- [Science (2022)](https://www.science.org/doi/10.1126/science.abj6987)
- [T2T Data](https://www.mimuw.edu.pl/~lukaskoz/teaching/adp/labs/lab_human_genome/)

Analysis of the **complete human genome (T2T Consortium v2.0)**:

- Data `chm13v2.0.fa.gz` (∼936MB)
- Calculate per-chromosome:
  - Total length
  - Base composition (A/T/G/C/N) and GC content
  - **prettytable summary** + `.csv`
- Implemented **6-frame ORF prediction** in:
  - Pure Python
  - Biopython
  - EMBOSS (via wrapper)
- Compared runtimes and number of CDS predictions on chr1 and chrM
- Basic matching against known proteins (UniProt + annotation FASTA)

📁 `project_8/`

---

## ⚙️ Project 9: High-Throughput Protein Feature Prediction

#### Data Sources:
- E. coli: [UP000000625_83333.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000000625/UP000000625_83333.fasta.gz)
- H. sapiens: [UP000005640_9606.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz)
- SwissProt: [uniprot_sprot.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)

Prediction of protein disorder and secondary structure for E. coli, human, and SwissProt datasets using HPC (CPU/GPU):

- **IUPred**: disorder prediction in CPU environment (single & multi-job SLURM setup)
- **ProtBert-BFD-SS3**: secondary structure prediction with PyTorch on GPU
- **Efficiency tests**: compare runtimes on CPU vs GPU
- **Pseudo-FASTA outputs**: compact format for visualization
- **Post-processing scripts**:
  - Identify highly helical proteins (SHS-score)
  - Match predictions with DisProt experimental data (MCC metric)
  - Extract top “induced fit” disorder candidates based on hybrid predictions

Final results:
- Structured reports with runtime benchmarks
- Disorder/structure percentage tables
- SwissProt: top 5000 helical & induced-fit proteins

📁 `project_9/`

---
