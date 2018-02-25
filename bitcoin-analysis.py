'''
Time Series analysis of Bitcoin
'''

import quandl
import pickle
from scipy.stats import skew, kurtosis, kurtosistest
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import logging
from dateutil.relativedelta import relativedelta
import datetime
import fbprophet
import numpy as np
from fbprophet.diagnostics import cross_validation
import os

# Configure Logging
fmt = '%(asctime)s -- %(levelname)s -- %(module)s %(lineno)d -- %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger('root')

# Set project root directory
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'


def btc_quandl(id):
    '''
    Collect Quandal time series data as dataframe and pickle. If it exists already load it from pickle
    :param id: the quandal id
    :return: Quabdal dataframe
    '''
    # Grab and store Quandal data for bitcoin value
    quandl.ApiConfig.api_key = 'ftosgLxbsFdzpqFzPCCH'
    # Try to grab a pickled version if it exists
    cache_path = directory+'Data/'+'{}.pkl'.format(id).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(id))
    # If it doesnt catch error and download the data
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(id))
        df = quandl.get(id, returns='pandas')
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(id, cache_path))
    return df


# Grab bitcoin btc_data from quandl
btc_data = btc_quandl('BCHARTS/BITSTAMPUSD')

# Extract the last year of data
btc_data_year = btc_data[btc_data.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]

# Make a plot of the historic btc price and volume
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
fig.suptitle('Bitcoin Price and Volume ($)', fontsize=16)
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
btc_data_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.15)
# Pickle
if os.path.isfile(directory+'Data/Models/fbprophet_logweightprice.model.sav'):
    btc_data_prophet_log = pickle.load(open(directory+'Data/Models/fbprophet_logweightprice.model.sav', 'rb'))
else:
    btc_data_prophet_log.fit(btc_fbprohpet)
    pickle.dump(btc_data_prophet_log, open(directory+'Data/Models/fbprophet_logweightprice.model.sav', 'wb'))
btc_data_forecast_log = btc_data_prophet_log.make_future_dataframe(periods=365*2, freq='D')
btc_data_forecast_log = btc_data_prophet_log.predict(btc_data_forecast_log)

# Identify change points
btc_changepoints = btc_data_prophet_log.changepoints
# Work out if they are +ve or -ve
c_data = btc_fbprohpet.ix[btc_changepoints, :]
deltas = btc_data_prophet_log.params['delta'][0]
c_data['delta'] = deltas
c_data['abs_delta'] = abs(c_data['delta'])
# Sort the values by maximum change
c_data = c_data.sort_values(by='abs_delta', ascending=False)

# Limit to 10 largest changepoints
c_data = c_data[:10]

# Separate into negative and positive changepoints
cpos_data = c_data[c_data['delta'] > 0]
cpos_data = cpos_data['delta']
cneg_data = c_data[c_data['delta'] < 0]
cneg_data = cneg_data['delta']

# Write out to csv
cpos_data.to_csv(directory+'Data/Bitcoin-analysis/cpos.csv')
cneg_data.to_csv(directory+'Data/Bitcoin-analysis/cneg.csv')

# Make a plot
btc_data_prophet_log.plot(btc_data_forecast_log, xlabel = 'Date', ylabel = 'LOG Weighted Price ($)')
plt.plot(btc_data_forecast_log['ds'], btc_data_forecast_log['yhat'], color = 'navy', linewidth = 2.0, label = 'Modeled')
plt.vlines(cpos_data.index, ymin = 0, ymax= 10, colors = 'g', linewidth=1.2, linestyles = 'dashed', label = '+ve Change')
plt.vlines(cneg_data.index, ymin = 0, ymax= 10, colors = 'r', linewidth=1.2, linestyles = 'dashed', label = '-ve Change')
plt.legend(loc='upper left', prop={'size':10})
plt.savefig(directory+'Images/btc_data_forecast.png')
plt.clf()

# Plot the components
btc_data_prophet_log.plot_components(btc_data_forecast_log)
plt.savefig(directory+'Images/btc_data_components.png')
plt.clf()

# TODO Turn this into a function
# Do the same for the Volume
btc_fbprohpet['y'] = np.log(btc_fbprohpet['Volume (Currency)'])
btc_data_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=False)

# Pickle
if os.path.isfile(directory+'Data/Models/fbprophet_logvolume.model.sav'):
    btc_data_prophet_log = pickle.load(open(directory+'Data/Models/fbprophet_logvolume.model.sav', 'rb'))
else:
    btc_data_prophet_log.fit(btc_fbprohpet)
    pickle.dump(btc_data_prophet_log, open(directory+'Data/Models/fbprophet_logvolume.model.sav', 'wb'))

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

# Fit to the non-log data? Save the models using pickle
btc_fbprohpet['y'] = btc_fbprohpet['Weighted Price']
btc_data_prophet = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=False, changepoint_prior_scale=0.15)
# Pickle
if os.path.isfile(directory+'Data/Models/fbprophet_weightedprice.model.sav'):
    btc_data_prophet = pickle.load(open(directory+'Data/Models/fbprophet_weightedprice.model.sav', 'rb'))
else:
    btc_data_prophet.fit(btc_fbprohpet)
    pickle.dump(btc_data_prophet, open(directory+'Data/Models/fbprophet_weightedprice.model.sav', 'wb'))


btc_data_forecast = btc_data_prophet.make_future_dataframe(periods=365*2, freq='D')
btc_data_forecast = btc_data_prophet.predict(btc_data_forecast)
plt.plot(btc_data_forecast['ds'],btc_data_forecast['trend'])
btc_data_prophet.plot(btc_data_forecast, xlabel = 'Date', ylabel = 'Weighted Price ($)')

plt.savefig(directory+'Images/btc_weightedprice.png')
plt.clf()
# Plot the components
btc_data_prophet.plot_components(btc_data_forecast)
plt.savefig(directory+'Images/btc_weight_components.png')
plt.clf()


# Volatility Histogram
#http://www.quantatrisk.com/2016/12/08/conditional-value-at-risk-normal-student-t-var-model-python/
# calculate daily logarithmic return
#btc_data_year['returns'] = (btc_data_year['Close']/btc_data_year['Close'].shift(-1)) - 1
#btc_hist = btc_data_year['returns'].copy()

# Remove influence of missing days
btc_data = btc_data.replace([np.inf, -np.inf, 0.0, -1.0], np.nan)
btc_data = btc_data.replace([np.inf, -np.inf], np.nan).dropna(how='all')
btc_data['returns'] = (btc_data['Close']/btc_data['Close'].shift(-1)) - 1
btc_hist = btc_data['returns'].copy()
btc_hist = btc_hist.replace([np.inf, -np.inf], np.nan).dropna(how='all')

ret = np.array(btc_hist)

# Fit the normal distribution N(x; mu, sig) - best fit (finding: mu, stdev)
mu_norm, sig_norm = norm.fit(ret)
dx = 0.0001
x = np.arange(min(ret), max(ret), dx)
pdf = norm.pdf(x, mu_norm, sig_norm)
print 'Normal mean  = %.5f' % mu_norm
print 'Normal stdev = %.5f' % sig_norm
print

# Fit the t-distribution - best fit (finding: nu)
parm = t.fit(ret)
nu, mu_t, sig_t = parm
nu = np.round(nu)
pdf2 = t.pdf(x, nu, mu_t, sig_t)
print 'nu = %.2f' % nu
print

# Compute VaRs and CVaRs
h = 1.0
# significance 99%
alpha = 0.01
lev = 100.0*(1-alpha)
xanu = t.ppf(alpha, nu)

CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm

VaR_t = np.sqrt((nu-2)/nu) * t.ppf(1-alpha, nu)*sig_norm  - h*mu_norm
CVaR_t = -1/alpha * (1-nu)**(-1) * (nu-2+xanu**2) * \
                t.pdf(xanu, nu)*sig_t  - h*mu_t

print '%g%% %g-day Normal VaR     = %.2f%%' % (lev, h, VaR_n*100)
print '%g%% %g-day Normal t CVaR  = %.2f%%' % (lev, h, CVaR_n*100)
print '%g%% %g-day Student t VaR  = %.2f%%' % (lev, h, VaR_t *100)
print '%g%% %g-day Student t CVaR = %.2f%%' % (lev, h, CVaR_t*100)


plt.figure(num=1, figsize=(11, 6))
grey = .77, .77, .77
# main figure
plt.hist(ret, bins=200, normed=True, color=grey, edgecolor='none')
plt.hold(True)
plt.axis('tight')
plt.plot(x, pdf, 'b', label='Normal PDF fit')
plt.hold(True)
plt.axis('tight')
plt.plot(x, pdf2, 'g', label='Student t PDF fit')
plt.xlim([-0.2, 0.2])
plt.ylim([0, 20])
plt.legend(loc='best')
plt.xlabel('BTC Daily Returns')
plt.ylabel('Normalised Return Distribution')
plt.text(-0.18, 18, '%g%% %g-day Normal VaR       = %.2f%%' % (lev, h, VaR_n*100))
plt.text(-0.18, 17, '%g%% %g-day Normal t CVaR  = %.2f%%' % (lev, h, CVaR_n*100))
plt.text(-0.18, 16, '%g%% %g-day Student t VaR   = %.2f%%' % (lev, h, VaR_t *100))
plt.text(-0.18, 15, '%g%% %g-day Student t CVaR = %.2f%%' % (lev, h, CVaR_t*100))
plt.text(-0.18, 14, 'Skewness = %.2f' % skew(ret))
plt.text(-0.18, 13, 'Kurtosis = %.2f' % kurtosis(ret, fisher=False))

# inset
a = plt.axes([.60, .40, .25, .35])
plt.hist(ret, bins=200, normed=True, color=grey, edgecolor='none')
plt.hold(True)
plt.plot(x, pdf, 'b')
plt.hold(True)
plt.plot(x, pdf2, 'g')
plt.hold(True)
# Student VaR line
plt.plot([-CVaR_t, -CVaR_t], [0, 3], c='g')
# Normal VaR line
plt.plot([-CVaR_n, -CVaR_n], [0, 4], c='b')
#plt.text(-CVaR_n-0.015, 4.1, 'Norm CVaR', color='b')
#plt.text(-CVaR_t-0.0171, 3.1, 'Student t CVaR', color='r')
plt.xlim([-0.35, 0.1])
#plt.ylim([0, 5])
plt.savefig(directory+'Images/btc_volatility.png')
plt.clf()

# Show residuals
dx = 0.0065353275048835
x = np.arange(min(ret), max(ret), dx)
pdf = norm.pdf(x, mu_norm, sig_norm)
pdf2 = t.pdf(x, nu, mu_t, sig_t)

(n, bins, patches) = plt.hist(ret, bins=200, normed=True)
residual_ret_norm = pdf-n
residual_ret_t = pdf2-n

plt.figure(figsize=(4, 2))
plt.plot(x,residual_ret_t, color='g')
plt.plot(x,residual_ret_norm, color='b')
plt.ylabel('Residuals')
plt.xlim([-0.2, 0.2])
plt.savefig(directory+'Images/btc_volatility_residuals.png')
plt.clf()


# Moving Average Convergence/Divergence
# Accumulation distribution
# https://www.investopedia.com/university/technical/techanalysis10.asp


# Cross validate the model. Can it predict based on historical data
df_cv = cross_validation(btc_data_prophet_log, '365 days', initial='1825 days', period='365 days')
#df_cv2 = cross_validation(btc_data_prophet_log, horizon='365 days')
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
plt.savefig(directory+'Images/btc_crossvalidation5.png')
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


