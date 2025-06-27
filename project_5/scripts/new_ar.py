from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sys

warnings.filterwarnings('ignore')


def read_dataframe(path):
    df = pd.read_csv(path)
    df = df.loc[df['Country'] == 'New Zealand']
    df = df[['Year', 'AverageTemperature']]
    # df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.groupby(['Year'], as_index=False).mean()
    return df


def check_stationary(df):
    result = adfuller(df['AverageTemperature'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def create_model(df):
    model = AutoReg(df['AverageTemperature'], lags=19)
    results = model.fit()
    return results


def create_plot(df, results, option):
    forecast_horizon = 250
    pred = results.get_prediction(start=0, end=len(df)-1+forecast_horizon)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    years = df['Year'].tolist()
    last_year = years[-1]
    future_years = [last_year + i for i in range(1, forecast_horizon+1)]
    all_years = years + future_years

    plt.figure(figsize=(13, 8))
    plt.plot(all_years, pred_mean, label='Prediction')
    plt.fill_between(all_years[:], pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='gray', alpha=0.3, label='95% confidence interval')
    plt.scatter(df['Year'], df['AverageTemperature'], color = 'black', alpha =0.7, label='Observed data')
    plt.grid(True, 'both')
    plt.xticks(range(int(min(all_years)), int(max(all_years))+1, 50))
    plt.axvline(last_year, color='k', linestyle='--', alpha=0.7)
    plt.text(last_year+3, plt.ylim()[1]*0.95, 'Forecast start', va='top', size = 18)
    plt.title('Average temperature by year in New Zealand \n Autoregression model', size = 26)
    plt.xlabel('Year', size = 20)
    plt.ylabel('Average temperature[°C]', size = 20)
    plt.legend(fontsize = 15)
    plt.tick_params('both', labelsize = 15)
    plt.tight_layout()
    if option == 1:
        plt.savefig('images/new_ar.png')
    else:
        plt.show()
    plt.close()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['0', '1']:
        print("Usage: python new_ar.py [0|1]")
        print("0 - show plot interactively, 1 - save plot to file")
        sys.exit(1)
    option = int(sys.argv[1])
    df = read_dataframe('data/new_zealand_yearly_avg.csv')
    check_stationary(df)
    results = create_model(df)
    create_plot(df, results, option)


if __name__ == '__main__':
    main()