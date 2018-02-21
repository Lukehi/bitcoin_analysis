# Combine the results of bitcoin-analysis.py, twitter-sentiment.py, and google-trends.py
# Search for correlation between the sentiment trends and BTC price / volume.

# TODO: Tidy up code

import pandas as pd
import quandl
import datetime
from dateutil.relativedelta import relativedelta
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors,svm
import optunity
import optunity.metrics
import fbprophet
import os


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
btc_data['Weighted Price_norm'] = btc_data['Weighted Price'] / max(btc_data['Weighted Price'])
btc_cpos = pd.read_csv(directory+'Data/Bitcoin-analysis/cpos.csv', names=['date','delta'])
btc_cpos = btc_cpos.set_index('date')
btc_cpos.index = pd.to_datetime(btc_cpos.index)
btc_cneg = pd.read_csv(directory+'Data/Bitcoin-analysis/cneg.csv', names=['date','delta'])
btc_cneg = btc_cneg.set_index('date')
btc_cneg.index = pd.to_datetime(btc_cneg.index)
# Extract the last year of data
btc_data_year = btc_data[btc_data.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]

# Grab the Google trends data
google_data = pd.read_csv(directory+'Data/Google/btc_googletrends.csv')
google_data = google_data.set_index('date')
google_data.index = pd.to_datetime(google_data.index)

# Normalise
google_data['bitcoin_norm'] = google_data['bitcoin'] / max(google_data['bitcoin'])
google_data_year = google_data[google_data.index > (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')]

# Grab the Twitter sentiment data
twitter_data = pd.read_csv(directory+'Data/Twitter/sentiment_vader_all_1d.csv')
twitter_data['compound_norm'] = twitter_data['compound'] / max(twitter_data['compound'])
#twitter_data = twitter_data.set_index('date')
twitter_data = twitter_data.set_index('date')
twitter_data.index = pd.to_datetime(twitter_data.index)
twitter_data = twitter_data[twitter_data['neu'] != 1]

# Plot trends vs btc price
# Make a plot of the historic btc price and volume with google trend
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
fig.suptitle('Bitcoin Price vs. Trends', fontsize=16)
btc_data['Weighted Price_norm'].plot(linestyle='-',color='b', label = 'BTC', grid=True, ax=axes[0], sharex=axes[1])
google_data['bitcoin_norm'].plot(linestyle='-',color='g', label = 'Google', grid=True, ax=axes[0])
btc_data['Weighted Price_norm'].plot(linestyle='-',color='b', label = 'BTC', grid=True, ax=axes[1])
twitter_data['compound_norm'].plot(linestyle='-',color='r', label = 'Twitter', grid=True, ax=axes[1])
plt.xlabel('Date')
axes[0].set_ylabel('Weighted Price ($)')
axes[1].set_ylabel('Sentiment Score')
plt.legend()
plt.savefig(directory+'Images/btc_google_twitter.png')
plt.clf()
# Analyse and search for trends. In the relationships
# Read in the change points for the weighted price and compare to trends identify change points in trends

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
fig.suptitle('Bitcoin Price vs. Trends', fontsize=16)
google_data['bitcoin'].plot(linestyle='-',color='g', label = 'Google', grid=True,ax=axes[0], sharex=axes[1])
#twitter_data['neg'].plot(linestyle='-',color='r', label = 'Twitter', grid=True, ax=axes[1])
twitter_data['pos'].plot(linestyle='-',color='r', label = 'Twitter', grid=True, ax=axes[1])
plt.xlabel('Date')
axes[0].set_ylabel('Google Interest')
axes[1].set_ylabel('Twitter Sentiment Score')
plt.legend()
plt.savefig(directory+'Images/google_twitter_btcchange.png')
plt.clf()

plt.legend()
plt.savefig(directory+'Images/google_twitter_btcchange.png')


# Measure correlation
# Note - need to probably make sure that the time series are stationary by first subtracting the overall trend
btc_data_3d = btc_data[(btc_data.index >= '2011-10-12') & (btc_data.index <= '2016-12-18')]
btc_data_3d = btc_data_3d.resample('3d').mean()

btc_data_3d['Weighted Price'].corr(btc_data_3d['Weighted Price'].shift(1))
btc_data_3d['Weighted Price'].corr(btc_data_3d['Weighted Price'].shift(100))

google_data_3d = google_data[(google_data.index >= '2011-10-12') & (google_data.index <= '2016-12-18')]
google_data_3d = google_data_3d.resample('3d').mean()

btc_data_3d['Weighted Price_norm'].corr(google_data_3d['bitcoin_norm'].shift(1))

twitter_data_3d = twitter_data[(twitter_data.index >= '2011-10-12') & (twitter_data.index <= '2016-12-18')]
twitter_data_3d = twitter_data_3d.resample('3d').mean()

btc_data_3d['Weighted Price_norm'].corr(twitter_data_3d['compound'].shift(3))

# The correlation needs to be measured for time series that have the overall trend removed
# Load the model for the log price model
btc_data_prophet_log = pickle.load(open(directory + 'Data/Models/fbprophet_logweightprice.model.sav', 'rb'))
# subtract the trend from the data
btc_fbprohpet = btc_data.copy()
btc_fbprohpet = btc_fbprohpet.reset_index(level=0, inplace=False)
btc_fbprohpet = btc_fbprohpet.rename(columns={'Date': 'ds'})
# Drop zero values
btc_fbprohpet = btc_fbprohpet[btc_fbprohpet['Weighted Price'] != 0]

btc_fbprohpet['y'] = np.log(btc_fbprohpet['Weighted Price'])
btc_data_forecast_log = btc_data_prophet_log.make_future_dataframe(periods=1, freq='D')
btc_data_forecast_log = btc_data_prophet_log.predict(btc_data_forecast_log)

btc_data_forecast_log = btc_data_forecast_log.set_index('ds')
btc_data_forecast_log.index = pd.to_datetime(btc_data_forecast_log.index)

btc_data_notrend = np.log(btc_data['Weighted Price']) - btc_data_forecast_log['trend']
btc_data_notrend = btc_data_notrend.replace([np.inf, -np.inf, 0.0, -1.0], np.nan)
btc_data_notrend = btc_data_notrend.replace([np.inf, -np.inf], np.nan).dropna(how='all')

# Make a model for Google
google_fbprohpet = google_data.copy()
google_fbprohpet.reset_index(level=0, inplace=True)
google_fbprohpet = google_fbprohpet.rename(columns={'date': 'ds'})
# Drop zero values
google_fbprohpet = google_fbprohpet[google_fbprohpet['bitcoin'] != 0]

# Fit to the log of the data
google_fbprohpet['y'] = np.log(google_fbprohpet['bitcoin'])
google_data_prophet_log = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.15)
# Pickle
if os.path.isfile(directory+'Data/Models/fbprophet_loggoogle.model.sav'):
    google_data_prophet_log = pickle.load(open(directory+'Data/Models/fbprophet_loggoogle.model.sav', 'rb'))
else:
    google_data_prophet_log.fit(google_fbprohpet)
    pickle.dump(google_data_prophet_log, open(directory+'Data/Models/fbprophet_loggoogle.model.sav', 'wb'))
google_data_forecast_log = google_data_prophet_log.make_future_dataframe(periods=365*2, freq='D')
google_data_forecast_log = google_data_prophet_log.predict(google_data_forecast_log)
google_data_forecast_log = google_data_forecast_log.set_index('ds')
google_data_forecast_log.index = pd.to_datetime(google_data_forecast_log.index)

google_data_notrend = np.log(google_data['bitcoin']) - google_data_forecast_log['trend']
google_data_notrend = google_data_notrend.replace([np.inf, -np.inf, 0.0, -1.0], np.nan)
google_data_notrend = google_data_notrend.replace([np.inf, -np.inf], np.nan).dropna(how='all')

# Make a plot of the detrended data.
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
fig.suptitle('Static BTC and Google Trends', fontsize=16)
btc_data_notrend.plot(linestyle='-',color='b', label = 'BTC', grid=True, ax=axes[0], sharex=axes[1])
google_data_notrend.shift(-20).plot(linestyle='-',color='g', label = 'Google', grid=True, ax=axes[1])
plt.xlabel('Date')
axes[1].set_ylabel('Log Google Interest')
axes[0].set_ylabel('Log BTC Weighted Price ($)')
axes[1].legend(loc='upper left')
axes[0].legend(loc='upper left')
plt.savefig(directory+'Images/btc_google_static.png')
plt.clf()


# Measure correlation between detrended google and bitcoin
pearson = []
for shift in range(0,200):
    pearson.append(btc_data_notrend.corr(google_data_notrend.shift(-1*shift)))

plt.plot(range(0,200), pearson)

# Subtract the trend

# Twitter data is too noisy. Can SVM be used?











# Perform analytical tests to look for correlation
# Could make a prophet model of the trends and see if the change positions are associated with the btc price changes?
# Can we perform some machine learning that is trained to classify whether the btc price will rise or fall based on the twitter sentiment / google trend.

# This can be defined as a classification problem https://arxiv.org/pdf/1610.09225.pdf
# sentiment as features, need to have the three features +ve neutral -ve
# output is 1 / 0 depending on if btc price rises or falls

twitter_data_3d = pd.read_csv(directory+'Data/Twitter/sentiment_vader_all_3d.csv')
twitter_data_3d = twitter_data_3d.set_index('date')
twitter_data_3d.index = pd.to_datetime(twitter_data_3d.index)

X = df_sentiment3d[list(['neg','neu','pos'])].values

# Try the google trends
google_data_3d = google_data[(google_data.index >= '2011-10-12') & (google_data.index <= '2016-12-18')]
google_data_3d = google_data_3d.resample('3d').mean()
google_data_3d['label'] = ''
# Check this
google_data_3d['label'][google_data_3d['bitcoin'] >= google_data_3d['bitcoin'].shift(-1)] = -1
google_data_3d['label'][google_data_3d['bitcoin'] <= google_data_3d['bitcoin'].shift(-1)] = 1
X = google_data_3d['label']
X[-1] = -1
X = np.asarray(X.astype('int'))

df_features = df_sentiment3d[['neg','neu','pos']]
df_features['google'] = X
df_features['google'] = df_features['google'].shift
X = df_features[list(['neg','neu','pos','google'])].shift(-1).values
X = X[:-1]


# Labels are whether the btc price increased or decreased
# Grab the same time range as the sentiment analysis
btc_data_3d = btc_data[(btc_data.index >= '2011-10-12') & (btc_data.index <= '2016-12-18')]
btc_data_3d = btc_data_3d.resample('3d').mean()
btc_data_3d['label'] = ''
# Check this
btc_data_3d['label'][btc_data_3d['Weighted Price'] >= btc_data_3d['Weighted Price'].shift(-1)] = -1
btc_data_3d['label'][btc_data_3d['Weighted Price'] <= btc_data_3d['Weighted Price'].shift(-1)] = 1
y = btc_data_3d['label']
y[-1] = -1
y = np.asarray(y.astype('int'))
y = y[:-1]
# Split training and testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# Optimize parameters
#http://optunity.readthedocs.io/en/latest/examples/python/sklearn/svc.html

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=data, y=labels, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(data, labels)

# Train the classifier
#clf = svm.SVC(kernel='rbf', C=10 ** hps['logC'],gamma=10 ** hps['logGamma'], degree=3,coef0=0.0,tol=0.001)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy

example_measures = np.array([[1.0,0.0,1.0,1.0]])
#example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print prediction

# Given the model can we correctly predict the btc change points?

# Make a simple example to demonstrate.

# Replicate one of the points

# Visualise 2D
X1 = X[:,0][y == 1]
X0 = X[:,0][y == -1]
Y1 = X[:,1][y == 1]
Y0 = X[:,1][y == -1]
Z1 = X[:,2][y == 1]
Z0 = X[:,2][y == -1]

plt.scatter(X0,Y0, color='b')
plt.scatter(X1,Y1, color='r')
plt.show()

# Visualise 3D
X1 = X[:,0][y == 1]
X0 = X[:,0][y == 0]
Y1 = X[:,1][y == 1]
Y0 = X[:,1][y == 0]
Z1 = X[:,2][y == 1]
Z0 = X[:,2][y == 0]
Xs = X[:,0]
Ys = X[:,1]
Zs = X[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, Y1, Z1, color='r')
ax.scatter(X0, Y0, Z0, color='b')
ax.scatter(Xs, Ys, Zs, color='black')
# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

w = clf.coef_[0]
print(w)
a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
# Run a test

# Maybe make a 3D plot of sentiment (neg neu pos) colored by 1 / 0 stock increase

# Could potentially run the google trend? The labels could be 0 1 for if the google trend increases
# This might work better because it's a lot simpler

# Machine learning prediction?
# https://github.com/sebastianheinz/stockprediction
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
# https://towardsdatascience.com/stock-analysis-in-python-a0054e2c1a4c
