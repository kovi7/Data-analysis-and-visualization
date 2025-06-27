"""
Task 2: Scatter plot
a) first plot all AverageTemperatureCelsius vs. year
The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig1.png

Summary: the plot is quite unreadable (we used to big 'circles' for data point)

b) recreate the plot using 'points' instead 'circles', add grid
The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig2.png

Summary: better, but still not very informative

c) add transparency
The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig3.png

Summary: you start to see anything

d) add color
The result should look like:
https://www.mimuw.edu.pl/~lukaskoz/teaching/dav/labs/lab4/fig4.png

Summary: this did not help to much, we move to another plot type and look more deeply 
"""



import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_graph(option):
    #data
    df = pd.read_csv('../data/temperatures_clean.csv')

    #plot
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    plt.scatter(df['year'], df['AverageTemperatureCelsius'], s=100, marker='.', alpha=0.3,  facecolors='None', color='blue')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Year', size = 20)
    plt.ylabel('Average Temperature [°C]', size =20)
    plt.tick_params(axis='both', labelsize= 15)
    plt.tight_layout()

    if option == 0:
        plt.show()
    elif option == 1:
        path = '../images/fig4.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task2d.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")