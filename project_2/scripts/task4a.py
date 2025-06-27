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
import sys
import matplotlib.pyplot as plt

def plot_graph(option):
    #data
    df = pd.read_csv('../data/temperatures_clean.csv')
    average_temp = df.groupby(['year', 'Country'])['AverageTemperatureCelsius'].mean().reset_index()

    #plot
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    plt.plot(average_temp['year'], average_temp['AverageTemperatureCelsius'], color='black')
    plt.tick_params(axis='both', labelsize= 15)
    plt.xlabel("Year", size = 20)
    plt.ylabel("Average Temperature [°C]", size = 20)
    plt.title('Global temperature trends by year for all countries in dataset', size = 25)
    plt.tight_layout()

    if option == 0:
        plt.show()
    elif option == 1:
        path = '../images/fig8.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task4a.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")