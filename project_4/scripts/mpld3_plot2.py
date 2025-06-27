import sys
import matplotlib.pyplot as plt
import pandas as pd
import mpld3

def main(save_flag):
    df = pd.read_csv('data/weather.csv', usecols=['MinTemp', 'MaxTemp','Temp9am', 'Temp3pm'])
    df.columns = ['min temperature', 'max temperature','temperature at 9am', 'temperature at 3pm']

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.grid(True, alpha=0.3)

    lines = []
    labels = []
    for column in df.columns: 
        line, = ax.plot(df.index, df[column], label=column)
        lines.append(line)
        labels.append(column)

    interactive_legend = mpld3.plugins.InteractiveLegendPlugin(
        lines,
        labels,
        alpha_unsel=0.5,
        alpha_over=1.5,
        start_visible=True, 
        font_size=20
    )

    ax.set_title("Interactive Scatter Plot from weather data using mpld3", size=30)
    ax.set_xlabel("Day number", size=25)
    ax.set_ylabel("Temperature °C", size=25)
    ax.tick_params(axis='both', labelsize=20)

    mpld3.plugins.connect(fig, interactive_legend)
    fig.subplots_adjust(right=0.75)

    if save_flag == "1":
        path = "plots/mpld3_plot2.html"
        mpld3.save_html(fig, path)
        print(f"Plot saved to {path}")
    else:
        mpld3.show()

if __name__ == "__main__":
    main(sys.argv[1])
