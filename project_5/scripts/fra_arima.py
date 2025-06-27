import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def read_dataframe(path):
    df = pd.read_csv(path)
    df = df.loc[df['Country'] == 'France']
    df = df[['Year', 'AverageTemperature']]
    df = df.groupby(['Year'], as_index=False).mean()
    return df

def create_model(series):
    model = ARIMA(series, order=(2,2,3), enforce_invertibility=False)
    results = model.fit()
    # print(results.summary())
    return results

def create_arima_plot(df, results, option):
    forecast_horizon = 250
    n_hist = len(df)

    #observed data
    pred_hist = results.get_prediction(start=0, end=n_hist-1)
    pred_hist_mean = pred_hist.predicted_mean
    pred_hist_ci = pred_hist.conf_int()

    #for future years
    pred_fut = results.get_forecast(steps=forecast_horizon)
    pred_fut_mean = pred_fut.predicted_mean
    pred_fut_ci = pred_fut.conf_int()

    years = df['Year'].tolist()
    last_year = years[-1]
    future_years = [last_year + i for i in range(1, forecast_horizon + 1)]

    plt.figure(figsize=(13, 8))
    plt.plot(years, pred_hist_mean, label='Prediction (historical)', color='blue')
    plt.fill_between(years, pred_hist_ci.iloc[:, 0], pred_hist_ci.iloc[:, 1], color='gray', alpha=0.3)
    plt.plot(future_years, pred_fut_mean, label='Forecast (future)', color='orange')
    plt.fill_between(future_years, pred_fut_ci.iloc[:, 0], pred_fut_ci.iloc[:, 1], color='orange', alpha=0.2, label='95% confidence interval')
    plt.scatter(years, df['AverageTemperature'], color='black', alpha=0.7, label='Observed data')

    plt.axvline(last_year, color='k', linestyle='--', alpha=0.7)
    plt.text(last_year+3, plt.ylim()[1]*0.95, 'Forecast start', va='top', size=18)
    plt.title('Average temperature by year in France\nARIMA model', size=26)
    plt.xlabel('Year', size=20)
    plt.ylabel('Average temperature [°C]', size=20)
    plt.legend(fontsize=15)
    plt.grid(True, 'both')
    plt.xticks(range(int(min(years)), int(max(future_years))+1, 50))
    plt.tick_params('both', labelsize=15)
    plt.ylim((7.5,20))
    plt.tight_layout()

    if option == 0:
        plt.show()
    else:
        path = 'images/fra_arima.png'
        plt.savefig(path)
        print(f'Plot saved as {path}')
    plt.close()

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['0', '1']:
        print("Usage: python fra_arima.py [0|1]")
        print("0 - show plot, 1 - save plot to file")
        sys.exit(1)
    option = int(sys.argv[1])

    df = read_dataframe('data/france_yearly_avg.csv')
    results = create_model(df['AverageTemperature'])
    create_arima_plot(df, results, option)

if __name__ == '__main__':
    main()
