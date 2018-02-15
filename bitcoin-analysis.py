# Time series analysis of BTC btc_data

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

# Configure Logging
fmt = '%(asctime)s -- %(levelname)s -- %(module)s %(lineno)d -- %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger('root')

# Set project root directory
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

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

# Grab bitcoin btc_data from quandl
btc_data = btc_quandl('BCHARTS/BITSTAMPUSD')
# Extract the last year of data
btc_data_year = btc_data[btc_data.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]

# Make a plot of the historic btc price and volume
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
fig.suptitle("Bitcoin Price and Volume ($)", fontsize=16)
btc_data['Weighted Price'].plot(grid=True, ax=axes[0,0], sharex=axes[0,1])
btc_data_year['Weighted Price'].plot(grid=True, ax=axes[0,1],sharey=axes[0,0])
btc_data['Volume (Currency)'].plot(grid=True, ax=axes[1,0])
btc_data_year['Volume (Currency)'].plot(grid=True, ax=axes[1,1],sharey=axes[1,0])
axes[0,0].set_ylabel('Weighted Price ($)')
axes[1,0].set_ylabel('Volume ($)')
axes[0,0].axvspan((datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d'),datetime.datetime.now().strftime('%Y-%m-%d') , alpha=0.5, color='red')
axes[1,0].axvspan((datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d'),datetime.datetime.now().strftime('%Y-%m-%d') , alpha=0.5, color='red')
fig.savefig(directory+'Images/btc.png')
fig.clf()
logger.info('Made BTC price and volume image: %s' % directory+'Images/btc.png')

# Grab some basic statistics from the past year and write to csv
btc_year_price_stats = btc_data_year.describe()
btc_year_price_stats.to_csv(directory+'Data/Bitcoin-analysis/btc_year_price.csv')

logger.info('Bitcoin Analysis\n')
logger.info('Time period: %s to %s' % (btc_data_year.index[0].strftime('%Y-%m-%d'), btc_data_year.index[-1].strftime('%Y-%m-%d')))
logger.info('Weight Price Stats')
logger.info('Max: %.2f, Min: %.2f, Mean: %.2f, Std: %.2f' % (btc_year_price_stats['Weighted Price']['max'], btc_year_price_stats['Weighted Price']['min'], btc_year_price_stats['Weighted Price']['mean'], btc_year_price_stats['Weighted Price']['std']))

# Perform basic timeseries analysis using fbprophet
btc_fbprohpet = btc_data.copy()
btc_fbprohpet.reset_index(level=0, inplace=True)
btc_fbprohpet = btc_fbprohpet.rename(columns={'Date': 'ds'})
# Drop zero values
btc_fbprohpet = btc_fbprohpet[btc_fbprohpet['Weighted Price'] != 0]

# Fit to the log of the data
btc_fbprohpet['y'] = np.log(btc_fbprohpet['Weighted Price'])
btc_data_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True)
btc_data_prophet_log.fit(btc_fbprohpet)
btc_data_forecast_log = btc_data_prophet_log.make_future_dataframe(periods=365*2, freq='D')
btc_data_forecast_log = btc_data_prophet_log.predict(btc_data_forecast_log)

btc_data_prophet_log.plot(btc_data_forecast_log, xlabel = 'Date', ylabel = 'LOG Weighted Price ($)')
plt.savefig(directory+'Images/btc_data_forecast.png')
plt.clf()
# Plot the components
btc_data_prophet_log.plot_components(btc_data_forecast_log)
plt.savefig(directory+'Images/btc_data_components.png')
plt.clf()

btc_fbprohpet['y'] = np.log(btc_fbprohpet['Volume (Currency)'])
btc_data_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True)
btc_data_prophet_log.fit(btc_fbprohpet)
btc_data_forecast_log = btc_data_prophet_log.make_future_dataframe(periods=365*2, freq='D')
btc_data_forecast_log = btc_data_prophet_log.predict(btc_data_forecast_log)

btc_data_prophet_log.plot(btc_data_forecast_log, xlabel = 'Date', ylabel = 'LOG Volume ($)')
plt.savefig(directory+'Images/btc_volume_forecast.png')
plt.clf()
# Plot the components
btc_data_prophet_log.plot_components(btc_data_forecast_log)
plt.savefig(directory+'Images/btc_volume_components.png')
plt.clf()
# The components / seasonality probably isnt reliable given the big jumps we see

# Volatility Histogram
#http://www.quantatrisk.com/2016/12/08/conditional-value-at-risk-normal-student-t-var-model-python/
# calculate daily logarithmic return
#btc_data_year['returns'] = (btc_data_year['Close']/btc_data_year['Close'].shift(-1)) - 1
#btc_hist = btc_data_year['returns'].copy()

btc_data['returns'] = (btc_data['Close']/btc_data['Close'].shift(-1)) - 1
btc_hist = btc_data['returns'].copy()
btc_hist = btc_hist.replace([np.inf, -np.inf, 0.0, -1.0], np.nan)
btc_hist = btc_hist.replace([np.inf, -np.inf], np.nan).dropna(how="all")

data = web.DataReader("IBM", data_source='google',
                  start='2010-12-01', end='2015-12-01')['Close']

cp = np.array(data.values)  # daily adj-close prices
ret = cp[1:]/cp[:-1] - 1    # compute daily returns

#ret = np.array(btc_hist)   # compute daily returns

# N(x; mu, sig) best fit (finding: mu, stdev)
mu_norm, sig_norm = norm.fit(ret)
dx = 0.0001  # resolution
x = np.arange(-1.0, 10, dx)
pdf = norm.pdf(x, mu_norm, sig_norm)
print("Sample mean  = %.5f" % mu_norm)
print("Sample stdev = %.5f" % sig_norm)
print()

# Student t best fit (finding: nu)
parm = t.fit(ret)
nu, mu_t, sig_t = parm
nu = np.round(nu)
pdf2 = t.pdf(x, nu, mu_t, sig_t)
print("nu = %.2f" % nu)
print()

# Compute VaRs and CVaRs
h = 1
alpha = 0.01  # significance level
lev = 100*(1-alpha)
xanu = t.ppf(alpha, nu)

CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm

VaR_t = np.sqrt((nu-2)/nu) * t.ppf(1-alpha, nu)*sig_norm  - h*mu_norm
CVaR_t = -1/alpha * (1-nu)**(-1) * (nu-2+xanu**2) * \
                t.pdf(xanu, nu)*sig_norm  - h*mu_norm

logger.info("%g%% %g-day Normal VaR     = %.2f%%" % (lev, h, VaR_n*100))
logger.info("%g%% %g-day Normal t CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
logger.info("%g%% %g-day Student t VaR  = %.2f%%" % (lev, h, VaR_t *100))
logger.info("%g%% %g-day Student t CVaR = %.2f%%" % (lev, h, CVaR_t*100))

plt.figure(num=1, figsize=(11, 6))
grey = .77, .77, .77
# main figure
plt.hist(ret, bins=100, normed=True, color=grey, edgecolor='none')
plt.hold(True)
plt.axis("tight")
plt.plot(x, pdf, 'b', label="Normal PDF fit")
plt.hold(True)
plt.axis("tight")
plt.plot(x, pdf2, 'g', label="Student t PDF fit")
plt.xlim([-0.2, 0.2])
plt.ylim([0, 20])
plt.legend(loc="best")
plt.xlabel("BTC Daily Returns")
plt.ylabel("Normalised Return Distribution")
plt.text(-0.18, 18, "%g%% %g-day Normal VaR       = %.2f%%" % (lev, h, VaR_n*100))
plt.text(-0.18, 17, "%g%% %g-day Normal t CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
plt.text(-0.18, 16, "%g%% %g-day Student t VaR   = %.2f%%" % (lev, h, VaR_t *100))
plt.text(-0.18, 15, "%g%% %g-day Student t CVaR = %.2f%%" % (lev, h, CVaR_t*100))

# inset
a = plt.axes([.60, .40, .25, .35])
plt.hist(ret, bins=100, normed=True, color=grey, edgecolor='none')
plt.hold(True)
plt.plot(x, pdf, 'b')
plt.hold(True)
plt.plot(x, pdf2, 'g')
plt.hold(True)
# Student VaR line
plt.plot([-CVaR_t, -CVaR_t], [0, 3], c='g')
# Normal VaR line
plt.plot([-CVaR_n, -CVaR_n], [0, 4], c='b')
#plt.text(-CVaR_n-0.015, 4.1, "Norm CVaR", color='b')
#plt.text(-CVaR_t-0.0171, 3.1, "Student t CVaR", color='r')
plt.xlim([-0.8, 0.1])
#plt.ylim([0, 5])
plt.savefig(directory+'Images/btc_volatility.png')
plt.clf()


# Short term and long term trends?

# Volume precedes price increase

# Rolling average and std

# Moving Average Convergence/Divergence
# https://www.investopedia.com/university/technical/techanalysis10.asp

# Machine learning prediction?
# https://github.com/sebastianheinz/stockprediction
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

# https://towardsdatascience.com/stock-analysis-in-python-a0054e2c1a4c

# Cross validate the model. Can it predict based on historical data
df_cv = cross_validation(btc_data_prophet_log, '365 days', initial='500 days', period='730 days')
cutoff = df_cv['cutoff'].unique()[0]
df_cv = df_cv[df_cv['cutoff'] == cutoff]

fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(btc_data_prophet_log.history['ds'].values, btc_data_prophet_log.history['y'], 'k.')
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



# Test changepoints
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    model = fbprophet.Prophet(daily_seasonality=False, changepoint_prior_scale=changepoint)
    model.fit(btc_data)

    future = model.make_future_dataframe(periods=365, freq='D')
    future = model.predict(future)

    btc_data[changepoint] = future['yhat']

# Create the plot
plt.figure(figsize=(10, 8))

# Actual observations
plt.plot(btc_data['ds'], btc_data['y'], 'ko', label='Observations')
colors = {0.001: 'b', 0.05: 'r', 0.1: 'grey', 0.5: 'gold'}

# Plot each of the changepoint predictions
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    plt.plot(btc_data['ds'], btc_data[changepoint], color=colors[changepoint], label='%.3f prior scale' % changepoint)

plt.legend(prop={'size': 14})
plt.xlabel('Date');
plt.ylabel('Market Cap (billions $)');
plt.title('Effect of Changepoint Prior Scale');
plt.show()


# Momentum / Volatility