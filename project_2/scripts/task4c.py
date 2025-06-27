'''
Task 4: Time series
a) calculate average temperature per year per each country 
- in this part, you need to group data by year and country 
- then calculate mean for each group
- do line plot with years on x axis and temperature on y axis

The result should look like: fig7.png

Summary: the plot is quite unreadable

b) to avoid too much of information split graphed data by Country

The result should look like: fig8.png

Summary: better

c) and add color

The result should look like: fig9.png

Summary: you start to see anything'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_graph(option):
    #data
    df = pd.read_csv('../data/temperatures_clean.csv')
    average_temp = df.groupby(['year', 'Country'], as_index=False)['AverageTemperatureCelsius'].mean().reset_index()
    average_temp = average_temp.sort_values(by='Country')

    #plot
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    sns.lineplot(data=average_temp, x='year', y='AverageTemperatureCelsius', hue='Country', palette='tab10')
    plt.xlabel("Year", size = 20)
    legend = plt.legend(title = 'Country',loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', fontsize=12)
    legend._legend_box.align = "left" 

    plt.tick_params(axis='both', labelsize= 15)
    plt.ylabel("Average Temperature [°C]", size = 20)
    plt.title('Global temperature trends by year for selected countries', size = 25)
    plt.tight_layout()

    if option == 0:
        plt.show()
    elif option == 1:
        path = '../images/fig10.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task4c.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")