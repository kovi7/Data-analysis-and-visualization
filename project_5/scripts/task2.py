import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2 or sys.argv[1] not in ['0', '1']:
    print("Usage: python task2.py [0|1]")
    print("0 - show plots, 1 - save plots to files")
    sys.exit(1)
option = int(sys.argv[1])

countries=['Brazil','France','Japan','Poland','South_Africa','Sweden','Ukraine', 'New_Zealand']

for country in countries:
    path = f'data/{country.lower()}_yearly_avg.csv'
    df = pd.read_csv(path)
    plt.figure(figsize=(13, 8))
    plt.plot(df['Year'], df['AverageTemperature'], alpha=0.3)
    
    # add trend line?
    z = np.polyfit(df['Year'], df['AverageTemperature'], 1)
    p = np.poly1d(z)
    plt.plot(df['Year'], p(df['Year']), "r--", alpha=0.8)
    
    plt.title(f'Average Temperature Over Years - {country}', size = 23)
    plt.xlabel('Year', size = 18)
    plt.ylabel('Average Temperature (°C)', size = 18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize= 13)
    if option == 1:
        plt.savefig(f"images/{country.lower()[:3]}.png")
        print(f'Plot saved as {f"images/{country.lower()[:3]}.png"}')
    else:
        plt.show()
    plt.close()
 
