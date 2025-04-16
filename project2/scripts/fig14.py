import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

'''Stacked plot of
b) pick one country and year at random and then find 4 other countries that are the closest by
population size (either + or -) in given year and do similar plot (e.g., Chile at 1985 and 4 other countries)
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

#preparing data
codes_with_region = meta_con.loc[~pd.isna(meta_con['Region']), 'Country Code'].tolist()
countries = data[data['Country Code'].isin(codes_with_region)].copy()
random_country_name = 'Chile'
random_year = '1985'
random_country_population_year = countries[countries['Country Name'] == random_country_name][random_year].values[0]
countries.loc[:,'pop_diff_year'] = abs(countries[random_year] - random_country_population_year)
closest_countries = countries.sort_values(by='pop_diff_year').head(5).sort_values(by='2023', ascending=True)
years = data.columns[2:-1]

#colors
colors = plt.cm.tab20c(np.linspace(0, 1, len(years)))

#plot
fig, ax = plt.subplots(figsize=(15, 10))
country_code_texts = []

def animate(year_index):    
    year = years[year_index]
    global country_code_texts

    # Title, labels, etc.
    plt.title(label=f"Population growth by year for {random_country_name}\nand its closest countries by population in {random_year}", fontsize=30, pad=10)
    ax.set_xlabel("Countries", fontsize=25, labelpad=10)
    ax.set_ylabel("Population size", fontsize=25, labelpad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    max_population = 2*1e6+(closest_countries['2023'].max())
    ax.set_ylim(0, max_population) 
    ax.text(0.09, 0.8, year, fontsize=45, transform=ax.transAxes, backgroundcolor='white')
    ax.tick_params(axis='both', labelsize=20)

    #stacking bars
    cumulative_heights = {country: 0 for country in closest_countries['Country Name']}
    for i, country in enumerate(closest_countries['Country Name']):
        var = closest_countries.loc[closest_countries['Country Name'] == country]
        if year_index == 0:
            difference = var[year].iloc[0]
        else:
            previous_year = years[year_index - 1]
            cumulative_heights[country] += var[previous_year].iloc[0]
            difference = var[year].iloc[0] - var[previous_year].iloc[0]
        
        prev_height = cumulative_heights[country]
        bar = ax.bar(country, difference, color=colors[year_index], bottom=prev_height)[0]
        
        # country code
        y = cumulative_heights[country] + 10 if year != '1960' else float(var[year].iloc[0])
        text = ax.text(x=bar.get_x() + bar.get_width() / 2, y=y, 
                    s=closest_countries.iloc[i]['Country Code'],
                    color='black', fontsize=20, ha='center', va='bottom', fontweight='bold')
        country_code_texts.append(text)

    #removing previous country codes
    for text in country_code_texts[:-5]: 
        text.remove()
    country_code_texts = country_code_texts[-5:]

ani = animation.FuncAnimation(fig, animate, frames=len(years), repeat=False)
ani.save('../images/fig14.gif', writer='imagemagick')

