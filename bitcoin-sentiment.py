# Combine the results of bitcoin-analysis.py, twitter-sentiment.py, and google-trends.py
# Search for correlation between the sentiment trends and BTC price / volume.

import pandas as pd
import quandl

# Set project root directory
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

def btc_quandl(id):
    '''Grab and store Quandal data for bitcoin value'''
    quandl.ApiConfig.api_key = 'ftosgLxbsFdzpqFzPCCH'
    # Try to grab a pickled version if it exists
    cache_path = directory+'Data/'+'{}.pkl'.format(id).replace('/','-')
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

# Grab the Google trends data
google_data = pd.read_csv(directory+'Data/Google/btc_googletrends.csv')

# Grab the Twitter sentiment data
#google_data = pd.read_csv(directory+'Data/Twitter/btc_twittersentiment.csv')


# Plot trends vs btc price


# Analyse and search for trends. Can the