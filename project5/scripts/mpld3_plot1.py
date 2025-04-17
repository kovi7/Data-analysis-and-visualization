import sys
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

def main(save_flag):
    data = pd.read_csv('data/ans2.tsv', sep='\t')

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=15)
    
    datasets = data['dataset'].unique()
    
    lines = []
    labels = []
    for dataset in datasets:
        subset = data[data['dataset'] == dataset]
        line, = ax.plot(subset['x'], subset['y'], marker='o', linestyle='', label={dataset}, alpha=0.8)
        lines.append(line)
        labels.append(dataset)

    ax.set_title("Interactive Scatter Plot from ans2 data using mpld3", size = 30)
    ax.set_xlabel("x", size = 25)
    ax.set_ylabel("y", size = 25)
    ax.tick_params(axis='both', labelsize=20)
    
    interactive_legend = mpld3.plugins.InteractiveLegendPlugin(lines, labels, alpha_unsel=0.2, alpha_over=1.0,font_size=20)
    mpld3.plugins.connect(fig, interactive_legend)
    fig.subplots_adjust(right=0.75) 

    if save_flag == "1":
        path = "plots/mpld3_plot1.html"
        mpld3.save_html(fig, path)
        # fig.subplots_adjust(right=None) 
        # fig.savefig("plots/mpld3_plot1.png",bbox_inches='tight')
        # plt.close(fig)
        print(f"Plot saved to {path}")
    else:
        mpld3.show()

if __name__ == "__main__":
    main(sys.argv[1])
