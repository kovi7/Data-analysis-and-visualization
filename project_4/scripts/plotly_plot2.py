import sys
import pandas as pd
import plotly.graph_objects as go

def main(save_flag):
    df = pd.read_csv('data/weather.csv', usecols=['MinTemp', 'MaxTemp', 'Temp9am','Temp3pm'])
    df.columns = ['min temperature', 'max temperature', 'temperature at 9am','temperature at 3pm']
    
    fig = go.Figure()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            name=column,
            line=dict(color=colors[idx], width=2),
            mode='lines'
        ))

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Day Number",
                font=dict(size=25)
            ),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title=dict(
                text="Temperature (°C)",
                font=dict(size=25)
            ), 
            tickfont=dict(size=20)
        ),
        legend=dict(
            title=dict(
                text="Legend",
                font=dict(size=25)
            ),
            font=dict(size=20),
            itemsizing="constant"
        ),
        title=dict(
            text="Interactive plot of weather data using plotly",
            font=dict(size=30, weight='bold')
        ), 
        showlegend=True  
    )
    if save_flag == "1":
        path = "plots/plotly_plot2.html"
        fig.write_html(path)
        print(f"Plot saved to {path}")
    else:
        fig.show()

if __name__ == "__main__":
    main(sys.argv[1])
