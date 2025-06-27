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
    all_cities = average_temp.groupby('Country')['City'].unique().explode().unique().tolist()

    # Plot
    sns.set_theme(style="whitegrid", rc={'ytick.left':True, 'xtick.bottom': True, 'axes.labelsize': 20})
    city_palette = dict(zip(all_cities, sns.color_palette("tab20", len(all_cities))))
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.delaxes(axes[-1]) 

    #subplots
    for i, country in enumerate(countries):
        country_data = average_temp[average_temp['Country'] == country]
        cities = sorted(country_data['City'].unique())

        for city in cities:
            city_data = country_data[country_data['City'] == city]

            line = sns.lineplot(
                data=city_data,
                x='year',
                y='AverageTemperatureCelsius',
                ax=axes[i],
                linewidth=1.5,
                hue='City',
                palette= city_palette,
                legend = False,
            )

        axes[i].set_title(country, fontsize=13, fontweight='bold')
        axes[i].tick_params(axis='both', which='major', labelsize=11)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=-45)



    fig.supxlabel('Year of observation',fontproperties = {'family':'monospace', 'weight':'bold', 'size':16})
    fig.supylabel('Average temperature [°C]', fontproperties = {'family':'monospace', 'weight':'bold', 'size':16})
    fig.suptitle('Average temperature trends by country and city',fontproperties = {'family':'monospace', 'weight':'bold', 'size':25})


    handles = [plt.Line2D([0], [0],color = city_palette[city],label=city, linewidth=2) 
        for city in all_cities]
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    legend = plt.figlegend(
        handles=handles, 
        title='City', 
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
        path = '../images/fig15.png'
        plt.savefig(path)
        print(f'Plot saved to {path}')
    else:
        print("Invalid argument. Use 0 to show the plot or 1 to save it.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 task5e.py [0/1]")
        sys.exit(1)
    
    try:
        option = int(sys.argv[1])
        plot_graph(option)
    except ValueError:
        print("Invalid input. Please provide 0 or 1.")
