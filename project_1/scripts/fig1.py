import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

'''1 a) 5 most populated countries (filter out groups like South Asia, OECD, etc.)
'''

def value_format(value, pos):
    if value >= 1e9:
        return f'{value*1e-9:.1f}B'
    elif value >= 1e6:
        return f'{value*1e-6:.1f}M'
    elif value >= 1e3:
        return f'{value*1e-3:.1f}K'
    else:
        return f'{value:.0f}'

#loading data
data = pd.read_csv("../data/dataset1.csv")
meta_con = pd.read_csv("../data/metadata1.csv")

codes_with_region = meta_con.loc[~pd.isna(meta_con['Region']), 'Country Code'].tolist()
countries = data[data['Country Code'].isin(codes_with_region)]
top_5_countries = countries.sort_values(by='2023', ascending=False).head(5)
years = data.columns[2:]

colors = plt.cm.tab20(np.linspace(0,1,5))
country_colors = {country: colors[i] for i, country in enumerate(top_5_countries['Country Name'])}

#plot
fig, ax = plt.subplots(figsize = (15,10))

def animate(year):
    ax.clear()

    # Title, labels, etc.
    plt.title(label="Population by year\nfor 5 most populated countries", fontsize=35, pad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    ax.set_xlabel('Country', fontsize=25, labelpad=10)
    ax.set_ylabel('Population size', fontsize=25, labelpad=10)
    max_population = 2*1e8+(top_5_countries['2023'].max())
    ax.set_ylim(0, max_population)
    ax.text(x=0.1, y=0.8, s=year, fontsize=45, transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=20)

    top_5_countries_sorted = top_5_countries.sort_values(by=year, ascending=True)
    bars = ax.bar(top_5_countries_sorted['Country Name'], top_5_countries_sorted[year], width=1, color=[country_colors[country] for country in top_5_countries_sorted['Country Name']])
    
    #country code
    for i, bar in enumerate(bars):
        country_code = top_5_countries_sorted['Country Code'].iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, country_code, 
                ha='center', va='bottom', fontsize=20, fontweight='bold')
    


ani = animation.FuncAnimation(fig, animate, frames=years, repeat=False)
ani.save('../images/fig1.gif', writer='imagemagick')
