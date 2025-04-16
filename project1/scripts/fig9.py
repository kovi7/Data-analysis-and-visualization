import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

"""Line plot of
c) the same as (b) but this time use Poland as "the centroid"
"""

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
countries = data[data['Country Code'].isin(codes_with_region)].copy()

random_year = '1985'
poland_value = countries.loc[countries['Country Name'] == 'Poland', random_year].values[0]
countries.loc[:, 'pop_diff_year'] = abs(countries[random_year] - poland_value)
closest_countries = countries.sort_values(by='pop_diff_year').head(5)  # Including the selected country
years = data.columns[2:]

# color
colors = plt.cm.Set1(np.linspace(0,1,5))
country_colors = {country: colors[i] for i, country in enumerate(closest_countries['Country Name'])}

# plot
fig, ax = plt.subplots(figsize=(15, 10))

def animate(year):
    ax.clear()
    
    # Set title, labels, etc.
    plt.title(label=f"Population by year for Poland\nand its closest countries by population in {random_year}", fontsize=30, pad=10)
    ax.set_xlabel('Year', fontsize=25)
    ax.set_ylabel('Population Size', fontsize=25)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlim(-0.5, len(years)+1)
    x_ticks = range(0, len(years), 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(years[x_ticks], rotation =45)
    max_population = 10000000+(closest_countries['2023'].max())
    ax.set_ylim(0, max_population)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    ax.text(0.25, 0.8, f'{years[year]}', fontsize=45, transform=ax.transAxes, color='black')

    closest_sorted = closest_countries.sort_values(by=years[year], ascending=True)
    for i, country in enumerate(closest_sorted['Country Name']):
        country_data = countries[countries['Country Name'] == country]
        year_data = country_data.iloc[0, 2:year+3]
        ax.plot(years[:year+1], year_data, label=country, color=country_colors[country], marker='' if year != 0 else '.')
        
        # country code
        offset = max_population * 0.01
        ax.text(years[year], year_data.iloc[year] + offset * (-1) ** i, country_data.iloc[0]['Country Code'], 
                color='black', fontsize=18, ha='center', va='bottom')
    ax.legend(labels = closest_countries.sort_values(by='1960', ascending=True)['Country Name'], loc = 2, fontsize = 15)

ani = animation.FuncAnimation(fig, animate, frames=len(years), repeat=False)
ani.save('../images/fig9.gif', writer='imagemagick')
