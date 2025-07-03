import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import pandas as pd

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
df_final = pd.read_csv('../../data/dataset1.csv')

#preparing data
df_filtered = df_final[(df_final['year'] >= 2005) & (df_final['year'] <= 2024)]
df_pivot_abs = df_filtered.pivot_table(index='year', columns='method', values='metric', aggfunc='sum').fillna(0)
methods_ordered=['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','NMR']

#plot
fig, ax = plt.subplots(figsize=(15, 10))
def animate(i):
    ax.clear()
    year = df_pivot_abs.index[i]

    # Title, labels, etc.
    ax.set_title(f'Distribution of structural data from different techniques\nin PDB (2005–2024) - absolute values', fontsize=30)
    ax.set_ylabel('Metric (n*1/r)', fontsize=25)
    ax.set_xlabel('Method', fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim(0, df_pivot_abs.values.max() * 1.1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(value_format))
    ax.text(x=0.7, y=0.8, s=year, fontsize=35, transform=ax.transAxes)

    values_yearly = df_pivot_abs.loc[year, methods_ordered]
    ax.bar(x=methods_ordered, height=values_yearly.values, color=['blue', 'orange', 'green'])
    

ani_abs = animation.FuncAnimation(fig, animate, frames=len(df_pivot_abs), repeat=False)
ani_abs.save('../../images/fig1.gif', writer='imagemagick')
