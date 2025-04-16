import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

'''Line plot of
a) 5 most populated countries (filter out groups like South Asia, OECD, etc.)
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

# loading data
data = pd.read_csv("../data/dataset1.csv")
meta_con = pd.read_csv("../data/metadata1.csv")

# preparing data
codes_with_region = meta_con.loc[~pd.isna(meta_con['Region']), 'Country Code'].tolist()
countries = data[data['Country Code'].isin(codes_with_region)]
top_5_countries = countries.sort_values(by='2023', ascending=False).head(5)
years = data.columns[2:]

# color
colors = plt.cm.Set1(np.linspace(0,1,5))
country_colors = {country: colors[i] for i, country in enumerate(top_5_countries['Country Name'])}

# plot
fig, ax = plt.subplots(figsize=(15, 10))

# aimation function
def animate(year):
    ax.clear()
    
    # Set title, labels, etc.
    ax.set_title(f'Population by year\nfor 5 most populated countries', fontsize=30)
    ax.set_xlabel('Year', fontsize=25)
    ax.set_ylabel('Population Size', fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(-0.5, len(years)+1)
    x_ticks = range(0, len(years), 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(years[x_ticks], rotation=45)
    max_population = 2*1e8+(top_5_countries['2023'].max())
    ax.set_ylim(0, max_population)    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    ax.text(0.25, 0.8, f'{years[year]}', fontsize=45, transform=ax.transAxes, color='black')

    countries_sorted = top_5_countries.sort_values(by=years[year], ascending=False)
    for i, country in enumerate(countries_sorted['Country Name']):
        country_data = countries[countries['Country Name'] == country]
        year_data = country_data.iloc[0, 2:year+3]
        ax.plot(years[:year+1], year_data, label=country, color=country_colors[country], marker='' if year != 0 else '.')
        
        # country code + legend
        ax.text(years[year], year_data.iloc[year], country_data.iloc[0]['Country Code'], 
                color='black', fontsize=18, ha='center', va='bottom')
        ax.legend(loc = 2, fontsize=15)

ani = animation.FuncAnimation(fig, animate, frames=len(years), repeat=False)
ani.save('../images/fig7.gif', writer='imagemagick')
# plt.show()