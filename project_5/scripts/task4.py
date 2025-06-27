from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tabulate import tabulate
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

COUNTRY_PARAMS = {
    'Brazil': {
        'ar_lags': 19,
        'arima_order': (0, 2, 3),
        'hwes_params': {'trend': 'mul', 'seasonal': None, 'initialization_method':'estimated'}
    },
    'France': {
        'ar_lags': 30,
        'arima_order': (2, 2, 3),
        'hwes_params': {'trend': 'add', 'seasonal': None, 'initialization_method':'estimated'}
    }
}

def read_dataframe(path, country):
    df = pd.read_csv(path)
    df = df.loc[df['Country'] == country]
    df = df[['Year', 'AverageTemperature']]
    df = df.groupby(['Year'], as_index=False).mean()
    return df
    
def evaluate_model(model_type, train, test, country):
    params = COUNTRY_PARAMS.get(country, {
        'ar_lags': 5,
        'arima_order': (1, 1, 1),
        'hwes_params': {'trend': 'add', 'seasonal': None}
    })
    
    try:
        if model_type == 'ar':
            lags = params['ar_lags']
            model = AutoReg(train, lags=lags)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        elif model_type == 'arima':
            order = params['arima_order']
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
        elif model_type == 'hwes':
            hwes_params = params['hwes_params']
            model = ExponentialSmoothing(train, **hwes_params)
            model_fit = model.fit()
            predictions = model_fit.forecast(len(test))
        
        # MAE
        mae = mean_absolute_error(test, predictions)
        return mae
    except ValueError as e:
        return np.nan



def cross_validate_models(df, country, n_splits=5):
    data = df['AverageTemperature'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {
        'ar': [],
        'arima': [],
        'hwes': []
    }
    
    for train_index, test_index in tscv.split(data):
        train, test = data[train_index], data[test_index]
        
        # evaluate models
        for model_type in results.keys():
            mae = evaluate_model(model_type, train, test, country)
            if not np.isnan(mae):
                results[model_type].append(mae)
    
    # MAE
    for model_type in results.keys():
        if results[model_type]:
            results[model_type] = np.mean(results[model_type])
        else:
            results[model_type] = np.nan
    
    return results


def create_table(results_dict):
    table_data = []
    
    for country, results in results_dict.items():
        table_data.append([
            country, 
            f"{results['ar']:.4f}" if not np.isnan(results['ar']) else "N/A", 
            f"{results['arima']:.4f}" if not np.isnan(results['arima']) else "N/A", 
            f"{results['hwes']:.4f}" if not np.isnan(results['hwes']) else "N/A"
        ])
    
    headers = ['Country', 'AR', 'ARIMA', 'HWES']
    
    table_text = tabulate(table_data, headers=headers, tablefmt='grid')
    table_html = tabulate(table_data, headers=headers, tablefmt='html')
    
    return table_text, table_html

def main():
    countries = ['Brazil', 'France']
    results_dict = {}
    
    for country in countries:        
        try:
            file_path = f'data/{country.lower()}_yearly_avg.csv'
            if os.path.exists(file_path):
                df = read_dataframe(file_path, country)
            else:
                df = read_dataframe('data/temperature_clean.csv', country)
            
            results = cross_validate_models(df, country)
            results_dict[country] = results
        except Exception as e:
            print(f"Error processing {country}: {e}")
    
    table_text, table_html = create_table(results_dict)
    print("\nCross-validation results (MAE):")
    print(table_text)
    
    with open('table.txt', 'w') as f:
        f.write("Cross-validation results (MAE):\n")
        f.write(table_text)
    
    with open('table.html', 'w') as f:
        f.write("<html><body>\n")
        f.write("<h2>Cross-validation results (MAE)</h2>\n")
        f.write("<p>Model parameters:</p>\n")
        f.write("<ul>\n")
        for country, params in COUNTRY_PARAMS.items():
            f.write(f"<li><strong>{country}</strong>: AR lags={params['ar_lags']}, "
                   f"ARIMA order={params['arima_order']}</li>\n")
        f.write("</ul>\n")
        f.write(table_html)
        f.write("\n</body></html>")
    
    print("\nTable saved to files: table.txt and table.html")

if __name__ == '__main__':
    main()
