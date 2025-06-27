
import pandas as pd

temp = pd.read_csv('../data/temperature.csv')

print(len(temp))

filtered = temp.copy().dropna(subset = ['City', 'Country'])
filtered = filtered.dropna(subset=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])

print(len(filtered))
filtered = filtered.drop(columns=['day'])



filtered['AverageTemperatureCelsius'] = (filtered['AverageTemperatureFahr'] - 32) *5/9
filtered['AverageTemperatureUncertaintyCelsius'] = (filtered['AverageTemperatureUncertaintyFahr']-32) *5/9

filtered = filtered.drop(columns=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])

print(filtered.head(10))
filtered.to_csv('../data/temperatures_clean.csv', index=False)
