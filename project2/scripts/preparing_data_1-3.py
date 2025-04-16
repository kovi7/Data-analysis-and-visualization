import pandas as pd

metadata = pd.read_csv("../data/raw/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_76253.csv", usecols=[0,1]).dropna()
metadata.to_csv('../data/metadata1.csv', index=False)

out = ['Indicator Name','Indicator Code', 'Unnamed: 68']
data1 = pd.read_csv("../data/raw/API_SP.POP.TOTL_DS2_en_csv_v2_76253.csv", skiprows = 4, usecols=lambda x: x not in out)
data1.dropna()
data1.to_csv('../data/dataset1.csv', index=False)


data2 = pd.read_csv("../data/raw/API_AG.LND.TOTL.K2_DS2_en_csv_v2_86.csv", skiprows = 4, usecols=lambda x: x not in out)
years = [col for col in data2.columns if col.isdigit()] 
data2["Land Area"] = data2[years].mean(axis=1, skipna=True)
data2 = data2[["Country Name", "Country Code", "Land Area"]]
data2.to_csv('../data/dataset2.csv', index=False)