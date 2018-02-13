# Time series analysis of BTC prices

import quandl
import pickle
import matplotlib.pyplot as plt
import logging
from dateutil.relativedelta import relativedelta
import datetime
import fbprophet
import numpy as np
from fbprophet.diagnostics import cross_validation
import pandas as pd

# Set project root directory
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'
# Configure Logging
fmt = '%(asctime)s -- %(levelname)s -- %(module)s %(lineno)d -- %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger('root')

def btc_quandl(id):
    '''Grab and store Quandal data for bitcoin value'''
    quandl.ApiConfig.api_key = 'ftosgLxbsFdzpqFzPCCH'
    # Try to grab a pickled version if it exists
    cache_path = '{}.pkl'.format(id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(id))
    # If it doesnt catch error and download
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(id))
        df = quandl.get(id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(id, cache_path))
    return df

# Grab bitcoin prices from quandl
btc_price = btc_quandl('BCHARTS/BITSTAMPUSD')
# Extract the last year of data
btc_price_year = btc_price[btc_price.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]

# Make a plot of the historic btc price and volume
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
fig.suptitle("Bitcoin Price and Volume ($)", fontsize=16)
btc_price['Weighted Price'].plot(grid=True, ax=axes[0,0], sharex=axes[0,1])
btc_price_year['Weighted Price'].plot(grid=True, ax=axes[0,1],sharey=axes[0,0])
btc_price['Volume (Currency)'].plot(grid=True, ax=axes[1,0])
btc_price_year['Volume (Currency)'].plot(grid=True, ax=axes[1,1],sharey=axes[1,0])
axes[0,0].set_ylabel('Weighted Price ($)')
axes[1,0].set_ylabel('Volume ($)')
axes[0,0].axvspan((datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d'),datetime.datetime.now().strftime('%Y-%m-%d') , alpha=0.5, color='red')
axes[1,0].axvspan((datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d'),datetime.datetime.now().strftime('%Y-%m-%d') , alpha=0.5, color='red')
fig.savefig(directory+'Images/btc.png')
fig.clf()
logger.info('Made BTC price and volume image: %s' % directory+'Images/btc.png')

# Grab some basic statistics from the past year and write to csv
btc_year_price_stats = btc_price_year.describe()
btc_year_price_stats.to_csv(directory+'Data/Bitcoin-analysis/btc_year_price.csv')

logger.info('Bitcoin Analysis\n')
logger.info('Time period: %s to %s' % (btc_price_year.index[0].strftime('%Y-%m-%d'), btc_price_year.index[-1].strftime('%Y-%m-%d')))
logger.info('Weight Price Stats')
logger.info('Max: %.2f, Min: %.2f, Mean: %.2f, Std: %.2f' % (btc_year_price_stats['Weighted Price']['max'], btc_year_price_stats['Weighted Price']['min'], btc_year_price_stats['Weighted Price']['mean'], btc_year_price_stats['Weighted Price']['std']))

# Perform basic timeseries analysis using fbprophet
btc_price.reset_index(level=0, inplace=True)
btc_price = btc_price.rename(columns={'Date': 'ds'})
# Drop zero values
btc_price = btc_price[btc_price['Weighted Price'] != 0]

# Fit to the log of the data
btc_price['y'] = np.log(btc_price['Weighted Price'])
btc_price_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True)
btc_price_prophet_log.fit(btc_price)
btc_price_forecast_log = btc_price_prophet_log.make_future_dataframe(periods=365*2, freq='D')
btc_price_forecast_log = btc_price_prophet_log.predict(btc_price_forecast_log)

btc_price_prophet_log.plot(btc_price_forecast_log, xlabel = 'Date', ylabel = 'LOG Weighted Price ($)')
plt.savefig(directory+'Images/btc_price_forecast.png')
plt.clf()
# Plot the components
btc_price_prophet_log.plot_components(btc_price_forecast_log)
plt.savefig(directory+'Images/btc_price_components.png')
plt.clf()

btc_price['y'] = np.log(btc_price['Volume (Currency)'])
btc_price_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True)
btc_price_prophet_log.fit(btc_price)
btc_price_forecast_log = btc_price_prophet_log.make_future_dataframe(periods=365*2, freq='D')
btc_price_forecast_log = btc_price_prophet_log.predict(btc_price_forecast_log)

btc_price_prophet_log.plot(btc_price_forecast_log, xlabel = 'Date', ylabel = 'LOG Weighted Price ($)')
plt.savefig(directory+'Images/btc_volume_forecast.png')
plt.clf()
# Plot the components
btc_price_prophet_log.plot_components(btc_price_forecast_log)
plt.savefig(directory+'Images/btc_volume_components.png')
plt.clf()


# Test changepoints
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    model = fbprophet.Prophet(daily_seasonality=False, changepoint_prior_scale=changepoint)
    model.fit(btc_price)

    future = model.make_future_dataframe(periods=365, freq='D')
    future = model.predict(future)

    btc_price[changepoint] = future['yhat']

# Create the plot
plt.figure(figsize=(10, 8))

# Actual observations
plt.plot(btc_price['ds'], btc_price['y'], 'ko', label='Observations')
colors = {0.001: 'b', 0.05: 'r', 0.1: 'grey', 0.5: 'gold'}

# Plot each of the changepoint predictions
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    plt.plot(btc_price['ds'], btc_price[changepoint], color=colors[changepoint], label='%.3f prior scale' % changepoint)

plt.legend(prop={'size': 14})
plt.xlabel('Date');
plt.ylabel('Market Cap (billions $)');
plt.title('Effect of Changepoint Prior Scale');
plt.show()


# Cross validate the model. Can it predict based on historical data
df_cv = cross_validation(btc_price_prophet_log, '365 days', initial='500 days', period='730 days')
cutoff = df_cv['cutoff'].unique()[0]
df_cv = df_cv[df_cv['cutoff'] == cutoff]

fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(btc_price_prophet_log.history['ds'].values, btc_price_prophet_log.history['y'], 'k.')
ax.plot(df_cv['ds'].values, df_cv['yhat'], ls='-', c='#0072B2')
ax.fill_between(df_cv['ds'].values, df_cv['yhat_lower'],
                df_cv['yhat_upper'], color='#0072B2',
                alpha=0.2)
ax.axvline(x=cutoff, c='gray', lw=4, alpha=0.5)
ax.set_ylabel('y')
ax.set_xlabel('ds')
#ax.text(x=pd.to_datetime('2010-01-01'),y=12, s='Initial', color='black',fontsize=16, fontweight='bold', alpha=0.8)
#ax.text(x=pd.to_datetime('2012-08-01'),y=12, s='Cutoff', color='black',fontsize=16, fontweight='bold', alpha=0.8)
#ax.axvline(x=cutoff + pd.Timedelta('365 days'), c='gray', lw=4,alpha=0.5, ls='--')
#ax.text(x=pd.to_datetime('2013-01-01'),y=6, s='Horizon', color='black',fontsize=16, fontweight='bold', alpha=0.8);

# Can we use the sentiment analysis to predict change points