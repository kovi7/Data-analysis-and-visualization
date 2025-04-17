import sys
import pandas as pd
from bokeh.plotting import figure, show, save
from bokeh.palettes import Category10

def main(save_flag):
    data = pd.read_csv('data/ans2.tsv', sep='\t')
    
    p = figure(title="Interactive scatter plot of ans2 data using Bokeh", x_axis_label='x', y_axis_label='y', width=1200, height=1200)

    datasets = data['dataset'].unique()
    for i, dataset in enumerate(datasets):
        subset = data[data['dataset'] == dataset]
        palette_size = len(Category10[10])  
        color = Category10[palette_size][i % palette_size]
        p.scatter(x=subset['x'], y=subset['y'], size=8, color=color, alpha=0.8, legend_label=f"{dataset}")

    p.title.text_font_size = "30pt"
    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.grid.grid_line_alpha = 0.5
    p.legend.title = "Legend"
    p.legend.click_policy = "hide"

    if save_flag == "1":
        path = "plots/bokeh_plot1.html"
        save(p, filename=path, title="Interactive scatter plot of ans2 data using bokeh")
        print(f"Plot saved to {path}")
    else:
        show(p)

if __name__ == "__main__":
    main(sys.argv[1])
