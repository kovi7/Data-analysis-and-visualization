import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import sys

def read_dataframe(path):
    df = pd.read_csv(path)
    df = df.loc[df['Country'] == 'Japan']
    df = df[['Year', 'AverageTemperature']]
    df = df.groupby(['Year'], as_index=False).mean()
    return df

def create_model(df):
    model = ExponentialSmoothing(df['AverageTemperature'], trend='mul', seasonal=None, initialization_method='estimated')
    results = model.fit()
    return results
    

def create_hwes_plot(df, model_results, option):
    forecast_horizon = 250
    n_sim = 1000

    #for observed data
    fitted = model_results.fittedvalues

    #to predict futture temperatures
    forecast = model_results.forecast(forecast_horizon)
    all_pred = np.concatenate([fitted, forecast])

    years = df['Year'].tolist()
    last_year = years[-1]
    future_years = [last_year + i for i in range(1, forecast_horizon + 1)]
    all_years = years + future_years

    #CI
    resid = df['AverageTemperature'].values - fitted.values
    sim_values = np.zeros((n_sim, len(all_pred)))
    for i in range(n_sim):
        noise = np.random.choice(resid, size=len(all_pred), replace=True)
        sim_values[i, :] = all_pred + noise

    lower = np.percentile(sim_values, 2.5, axis=0)
    upper = np.percentile(sim_values, 97.5, axis=0)

    #plott
    plt.figure(figsize=(13, 8))
    plt.plot(all_years, all_pred, label='Prediction (HWES)', color='tab:blue')
    # 95% CI
    plt.fill_between(all_years, lower, upper, color='gray', alpha=0.3, label='95% confidence interval')
    plt.scatter(df['Year'], df['AverageTemperature'], color='black', alpha=0.7, label='Observed data')

    plt.axvline(last_year, color='k', linestyle='--', alpha=0.7)
    plt.text(last_year+3, plt.ylim()[1]*0.95, 'Forecast start', va='top', size=18)
    plt.title('Average temperature by year in Japan\nHolt-Winters Exponential Smoothing', size=26)
    plt.xlabel('Year', size=20)
    plt.ylabel('Average temperature [°C]', size=20)
    plt.legend(fontsize=15)
    plt.grid(True, 'both')
    plt.xticks(range(int(min(all_years)), int(max(all_years))+1, 50))
    plt.tick_params('both', labelsize=15)
    plt.tight_layout()


    if option == 0:
        plt.show()
    else:
        path = 'images/jap_hwes.png'
        plt.savefig(path)
        print(f'Plot saved as {path}')
    plt.close()

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['0', '1']:
        print("Usage: python jap_hwes.py [0|1]")
        print("0 - show plot interactively, 1 - save plot to file")
        sys.exit(1)
    option = int(sys.argv[1])
    
    df = read_dataframe('data/japan_yearly_avg.csv')
    results = create_model(df)
    create_hwes_plot(df, results, option)

if __name__ == '__main__':
    main()
