import sys
import pandas as pd
from bokeh.plotting import figure, show, save
from bokeh.io import output_file
from bokeh.models import Band, ColumnDataSource

def main(save_flag):
    df = pd.read_csv('data/weather.csv', usecols=['MinTemp', 'MaxTemp','Temp9am', 'Temp3pm'])
    df.columns = ['min temperature', 'max temperature', 'temperature at 9am','temperature at 3pm']
    source = ColumnDataSource(df.reset_index())

    p = figure(
        title="Interactive Line Plot with Bokeh",
        x_axis_label='Day Number',
        y_axis_label='Temperature (°C)',
        width=1500,
        height=800
    )
    p.title.text_font_size = "30pt"
    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.grid.grid_line_alpha = 0.5
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(df.columns):
        line = p.line(
            x='index',
            y=column,
            source=source,
            line_width=2,
            legend_label=column,
            color=colors[idx]
        )

    p.legend.title = "Legend"
    p.legend.title_text_font_size = "15pt"
    p.legend.label_text_font_size = "13pt"
    p.legend.location = "bottom_left"

    if save_flag == "1":
        path = "plots/bokeh_plot2.html"
        output_file(path)
        save(p)
        print(f"Plot saved to {path}")
    else:
        show(p)

if __name__ == "__main__":
    main(sys.argv[1])
