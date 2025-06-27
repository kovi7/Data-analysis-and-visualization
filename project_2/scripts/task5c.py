'''Task 5: Grouping multiple subplots
a) make one (!) plot containing multiple subplots

The result should look like: fig10.png


b) split line in each subplot by city of each country

The result should look like: fig11.png

c) clean background 

Frequently, plotting libraries by default adds a grey grid. While
it may look nice at first glance, but, for print and better 
clarity, it is wise to reverse the grid coloring

The result should look like: fig12.png

d) divide into cities

As we gain some space by dividing the data into subplots, we can
use it to show more data. For each subplot/country add lines for 
individual data. You can have one legend or multiple legends 
(separate for each subplot). Mark the cities in different colors.

The result should look like: fig13.png

e) change labels, add title, customize fonts, rotate elements, etc.

The result should look like: fig14.png, fig15.png

Note: you need to use python, but you do not need limit yourself to 
matplotlib (seabonr is nice alternative plotting library)'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_graph(option):
    # data
    df = pd.read_csv('../data/temperatures_clean.csv')
    average_temp = df.groupby(['year', 'Country', 'City'], as_index=False)['AverageTemperatureCelsius'].mean()

    countries = sorted(average_temp['Country'].unique())

    # Plot
    sns.set_theme(style="whitegrid", rc={'ytick.left':True, 'xtick.bottom': True, 'axes.labelsize': 20})
    palette = dict(zip(countries, sns.color_palette("husl", len(countries))))

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.delaxes(axes[-1]) 

    #subplots
    for i, country in enumerate(countries):
        country_data = average_temp[average_temp['Country'] == country]
        cities = sorted(country_data['City'].unique())
        for city in cities:
            city_data = country_data[country_data['City'] == city]
            sns.lineplot(
                data=city_data,
                x='year',
                y='AverageTemperatureCelsius',
                ax=axes[i],
                linewidth=1.5,
                color=palette[country],
                err_style=None
            )

        axes[i].set_title(country, fontsize=13, fontweight='bold')
        axes[i].tick_params(axis='both', which='major', labelsize=11)
        if i%2==0:
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

    handles = [plt.Line2D([0], [0],color = palette[country],label=country, linewidth=2) 
        for country in countries]
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    legend = plt.figlegend(
        handles=handles, 
        title='Country', 
        loc='center right', 
        bbox_to_anchor=(1,0.5),
        fontsize=12,
        title_fontsize=16,
        facecolor='white'
    )
    legend._legend_box.align = "left" 
    
    if option == 0:
        plt.show()
    elif option == 1:
        path = '../images/fig13.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task5c.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")
