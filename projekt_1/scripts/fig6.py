import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

'''c) the same as (b) but this time use Poland as "the centroid"
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
    
def add_texture(rect, texture_type='solid'):
    if texture_type == 'stripes':
        rect.set_hatch('//') 
    elif texture_type == 'dots':
        rect.set_hatch('...')
    elif texture_type == 'hatch':
        rect.set_hatch('x')
    elif texture_type == 'crosshatch':
        rect.set_hatch('++')
    return rect

#loading data
data = pd.read_csv("../data/dataset1.csv")
meta_con = pd.read_csv("../data/metadata1.csv")

#preparing data
codes_with_region = meta_con.loc[~pd.isna(meta_con['Region']), 'Country Code'].tolist()
countries = data[data['Country Code'].isin(codes_with_region)].copy()

random_year = '1985'
poland_value = countries.loc[countries['Country Name'] == 'Poland', random_year].values[0]
countries.loc[:, 'pop_diff_year'] = abs(countries[random_year] - poland_value)
closest_countries = countries.sort_values(by='pop_diff_year').head(5)  # Including the selected country
years = data.columns[2:]

#textures
texture_types = ['stripes', 'dots', 'solid', 'hatch', 'crosshatch']
country_texture = {country: texture_types[i] for i, country in enumerate(closest_countries['Country Name'])}

# plot
fig, ax = plt.subplots(figsize = (15,10))

def animate_poland(year):
    ax.clear()

    # Title, labels, etc.
    plt.title(label=f"Population by year for Poland\nand its closest countries by population in {random_year}", fontsize=31, pad=10)
    plt.ylabel('Population size', fontsize=25, labelpad=10)
    plt.xlabel('Country', fontsize=25, labelpad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    max_population = 6500000+(closest_countries['2023'].max())
    ax.set_ylim(0, max_population)
    ax.text(x=0.1, y=0.8, s=year, fontsize=45, transform=ax.transAxes)
    ax.tick_params(axis='both', labelsize=20)
    
    closest_sorted = closest_countries.sort_values(by=year, ascending=True)
    bars = ax.bar(closest_sorted['Country Name'], closest_sorted[year], color='gray')

    #country codes+textures
    for i, bar in enumerate(bars):
        country_code, country_name = closest_sorted['Country Code'].iloc[i],closest_sorted['Country Name'].iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, country_code, 
                ha='center', va='bottom', fontsize=20, fontweight='bold')
        add_texture(bar, texture_type=country_texture[country_name])


ani = animation.FuncAnimation(fig, animate_poland, frames=years, repeat=False)
ani.save('../images/fig6.gif', writer='imagemagick')
