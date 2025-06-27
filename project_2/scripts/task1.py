
import pandas as pd

temp = pd.read_csv('../data/temperature.csv')

print(len(temp))

#a) there are missing values for City and Country in some records, remove those

filtered = temp.copy().dropna(subset = ['City', 'Country'])

# b) Remove rows with missing values in AverageTemperatureFahr and AverageTemperatureUncertaintyFahr
filtered = filtered.dropna(subset=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])

print(len(filtered))

#c) remove empty 'City' and 'Country' fields
# filtered = filtered.drop(labels='')
print(len(filtered))


# d) Remove the 'day' field as it only contains 1
filtered = filtered.drop(columns=['day'])


# e) convert AverageTemperatureFahr and AverageTemperatureUncertaintyFahr into 
# AverageTemperatureCelsius and AverageTemperatureUncertaintyCelsius 
# (you can drop '*Fahr' columns)
filtered['AverageTemperatureCelsius'] = (filtered['AverageTemperatureFahr'] - 32) *5/9
filtered['AverageTemperatureUncertaintyCelsius'] = (filtered['AverageTemperatureUncertaintyFahr']-32) *5/9

filtered = filtered.drop(columns=['AverageTemperatureFahr', 'AverageTemperatureUncertaintyFahr'])

print(filtered.head(10))
# f) Save cleaned data to CSV
filtered.to_csv('../data/temperatures_clean.csv', index=False)
