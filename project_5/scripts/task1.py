import pandas as pd

# data
data = pd.read_csv('data/temperature.csv')
filtered = data.dropna(subset=['City', 'Country'])
filtered = filtered.dropna(subset=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])
filtered = filtered.drop(columns=['day'])
filtered['AverageTemperatureCelsius'] = (filtered['AverageTemperatureFahr'] - 32) * 5/9
filtered['AverageTemperatureUncertaintyCelsius'] = (filtered['AverageTemperatureUncertaintyFahr'] - 32) * 5/9
filtered = filtered.drop(columns=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])

# filtered data
for country, country_data in filtered.groupby('Country'):
    country_yearly = country_data.groupby('year')['AverageTemperatureCelsius'].mean().reset_index()
    country_yearly.insert(0, 'Country', country)  
    country_yearly = country_yearly.rename(columns={'year': 'Year', 'AverageTemperatureCelsius': 'AverageTemperature'})
    filename = f"data/{country.lower().replace(' ', '_')}_yearly_avg.csv"
    country_yearly.to_csv(filename, index=False)

