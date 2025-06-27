import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

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
chosen = ['Cambodia', 'Switzerland', 'Malaysia','Lao PDR', 'Madagascar']

#preparing data
chosen_countries = data[data['Country Name'].isin(chosen)].copy()
years = chosen_countries.columns[2:]
cambodia_pop_1975 = float(chosen_countries[chosen_countries['Country Name']=='Cambodia']['1975'].values[0])

# color
colors = plt.cm.tab20(np.linspace(0,1,5))
country_colors = {country: colors[i] for i, country in enumerate(chosen_countries['Country Name'])}

#plot
fig, ax = plt.subplots(figsize=(15,10))

def animate(year):
    ax.clear()

    # Title, labels, etc.
    plt.title('Population growth dynamics by year\ncompared to Cambodia', fontsize= 35, pad=10)
    plt.ylabel('Population size', fontsize=25, labelpad=10)
    plt.xlabel("Country", fontsize=25, labelpad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    max_population = 5000000 + (chosen_countries['2023'].max())
    ax.set_ylim(0, max_population)
    ax.text(x=0.09, y=0.8, s=year, fontsize=45, transform=ax.transAxes, 
            color = 'darkred' if 1975 <= float(year) <= 1979 else 'black')
    ax.tick_params(axis='both', labelsize=20)
    ax.axhline(y=cambodia_pop_1975, linestyle='--', color = 'black')    

    bars = ax.bar(chosen_countries['Country Name'], chosen_countries[year].values, color=[country_colors[country] for country in chosen_countries['Country Name']])
    
    # Cambodian Genocide (1975-1979)
    if  1975 <= float(year) <= 1979:
        ax.text(0.53, 0.8, 'Cambodian Genocide', transform=ax.transAxes, 
                fontsize=45, color='darkred', ha='center')
    
    # country code
    for i, bar in enumerate(bars):
        population = chosen_countries[year].iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100000, round(population/1e6), 
                ha='center', va='bottom', fontsize=20)


# pause in 1975 
frames = list(years)
pause_duration = 20 
pause_frames = ['1975'] * pause_duration
pause_index = frames.index('1975')
frames = frames[:pause_index+1] + pause_frames + frames[pause_index+1:]

ani = FuncAnimation(fig, animate, frames=frames, repeat=False)
ani.save('../images/fig16.gif', writer='imagemagick')
