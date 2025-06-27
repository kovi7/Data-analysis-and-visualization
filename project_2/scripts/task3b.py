"""
Task 3: Box plots
a) visualising the distribution of temperatures within each country

The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig5.png

Summary: we start to see some interesting facts

b) add jitter to boxplot

You will need to play with the parameters e.g. transparency
https://stackoverflow.com/questions/29779079/adding-a-scatter-of-points-to-a-boxplot-using-matplotlib

The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig6.png

c) change boxplot to violin plot (sometimes known as a beanplot)
Boxplots are useful summaries, but they hide the shape of the distribution. 
For instance, if there is a bimodal distribution, this would not be observed
with a boxplot. An alternative is a violin plot, where the shape 
(of the density of points) is drawn.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_graph(option):
    #data
    df = pd.read_csv('../data/temperatures_clean.csv')

    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')

    names, vals, xs = [], [], []
    countries = df['country_id'].unique()
    for i, country in enumerate(countries):
        names.append(country)
        country_data = df[df['country_id'] == country]['AverageTemperatureCelsius'].tolist()
        vals.append(country_data)
        # jitter
        xs.append(np.random.normal(i+1, 0.1, len(country_data)))

    box = plt.boxplot(vals, labels=names, patch_artist=True, widths=0.5)

    for patch in box['boxes']:
        patch.set(facecolor='none') 
        patch.set(edgecolor='black', linewidth=1.5)

    for x, val in zip(xs, vals):
        plt.scatter(x, val, color='#FF6347', alpha=0.5)

    plt.grid(True, linestyle='-', alpha=0.7)

    plt.tick_params(axis='both', labelsize= 15)
    plt.xlabel('Country', fontsize=20)
    plt.ylabel('Average Temperature [°C]', fontsize=20)
    plt.title('Temperature distribution by country', size = 25)
    plt.tight_layout()

    if option == 0:
        plt.show()
    elif option == 1:
        path = '../images/fig6.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task3b.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")