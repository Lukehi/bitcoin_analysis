# Simple timeseries analysis of bitcoin prices

import quandl
import pickle
import matplotlib.pyplot as plt
import logging
from dateutil.relativedelta import relativedelta
import datetimefrom matplotlib.finance import candlestick_ohlc

# Set project root directory
directory = '/Users/lukehindson/Documents/Jobs/Jobs/Bitcoin/'
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
logger.info('Made BTC price and volume image: %s') % (directory+'Images/btc.png')

# Grab some basic statistics from the past year and write to csv
btc_price_year = btc_price[btc_price.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]
btc_year_price_stats = btc_year_price.describe()
btc_year_price_stats.to_csv(directory+'Data/Bitcoin-analysis/btc_year_price.csv')

logger.info('Bitcoin Analysis\n')
print ('Time period: %s to %s') % (btc_year_price.index[0].strftime('%Y-%m-%d'),btc_year_price.index[-1].strftime('%Y-%m-%d'))
print ('Weight Price Stats')
print ('Max: %.2f, Min: %.2f, Mean: %.2f, Std: %.2f') % (btc_year_price_stats['Weighted Price']['max'], btc_year_price_stats['Weighted Price']['min'], btc_year_price_stats['Weighted Price']['mean'], btc_year_price_stats['Weighted Price']['std'])
print
logger.info('Time period: %s\n')

# Perform basic timeseries analysis