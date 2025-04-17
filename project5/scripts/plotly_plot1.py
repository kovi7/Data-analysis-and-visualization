import sys
import pandas as pd
import plotly.graph_objects as go

def main(save_flag):
    data = pd.read_csv('data/ans2.tsv', sep='\t')
    fig = go.Figure()

    datasets = data['dataset'].unique()
    for dataset in datasets:
        subset = data[data['dataset'] == dataset]
        fig.add_trace(go.Scatter(
            x=subset['x'], 
            y=subset['y'], 
            mode='markers',
            name=dataset,
            marker=dict(size=10)
        ))
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="x",
                font=dict(size=25)
            ),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title=dict(
                text="y",
                font=dict(size=25)
            ), 
            tickfont=dict(size=20)
        ),
        legend=dict(
            title=dict(
                text="Datasets",
                font=dict(size=25)
            ),
            font=dict(size=20),
            itemsizing="constant"
        ),
        title=dict(
            text="Interactive plot of ans2 data using plotly",
            font=dict(size=30, weight='bold')
        ), 
        showlegend=True,
        autosize = False,
        width=1200,
        height=1200,
    )


    if save_flag == "1":
        path = "plots/plotly_plot1.html"
        fig.write_html(path)
        print(f"Plot saved to {path}")
    else:
        fig.show()

if __name__ == "__main__":
    main(sys.argv[1])
