import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

"""Bubble plot of
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

#loading data
data = pd.read_csv("../data/dataset1.csv")
meta_con = pd.read_csv("../data/metadata1.csv")
land_area = pd.read_csv("../data/dataset2.csv")

#preparing data
codes_with_region = meta_con.loc[~pd.isna(meta_con['Region']), 'Country Code'].tolist()
countries = data[data['Country Code'].isin(codes_with_region)].copy()
x = pd.merge(countries, land_area[['Country Code', 'Land Area']], on='Country Code', how='left')
random_year = '1985'
poland_value = countries.loc[countries['Country Name'] == 'Poland', random_year].values[0]
x.loc[:,'pop_diff_year'] = abs(x[random_year] - poland_value)
closest_countries = x.sort_values(by='pop_diff_year').head(5).sort_values(by='2023', ascending=True)
years = x.columns[2:-2]

#colors
colors = plt.cm.Set1(np.linspace(0,1,5))
country_colors = {country: colors[i] for i, country in enumerate(closest_countries['Country Name'])}

# plot 
fig, ax = plt.subplots(figsize=(15, 10))
country_code_texts = []

def animate(year):
    global country_code_texts

    # title, labels, etc.
    plt.title(label=f"Population by year for Poland\nand its closest countries by population in {random_year}", fontsize=30, pad=10)
    ax.set_xlabel('Year', fontsize=25, labelpad=10)
    ax.set_ylabel('Population', fontsize=25, labelpad=10)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlim(-0.5, len(years) + 1)
    x_ticks = range(0, len(years), 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(years[x_ticks], rotation=45)
    max_population = 1e6+(closest_countries['2023'].max())
    ax.set_ylim(0, max_population) 
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    ax.text(0.25, 0.8, f'{years[year]}', fontsize=45, transform=ax.transAxes, color='black', backgroundcolor='white')
    
    for i, country in enumerate(closest_countries['Country Name']):
        country_data = x[x['Country Name'] == country]
        population = country_data.iloc[0, 2 + year]
        land_area = pd.to_numeric(country_data.iloc[0]['Land Area'])
        population_density = population / land_area
        ax.scatter(years[year], population, color=country_colors[country],
                   s=population_density, alpha=0.5)

        # country code
        country_code = country_data.iloc[0]['Country Code']
        offset = max_population * 0.01
        text =ax.text(years[year], (population + population_density**2)*1.03 + offset * (-1) ** i, country_code, 
                color='black', fontsize=18, ha='center', va='bottom')
        country_code_texts.append(text)

    # removing previous country codes
    for text in country_code_texts[:-5]: 
        text.remove()
    country_code_texts = country_code_texts[-5:]

    # legend
    if year == 0:
        handles, labels = [], []
        for country, color in country_colors.items():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.5))
            labels.append(country)
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=15)

ani = animation.FuncAnimation(fig, animate, frames=len(years), repeat=False)
ani.save('../images/fig12.gif', writer='imagemagick')
