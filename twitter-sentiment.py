# Twitter Bitcoin sentiment analysis
import got
from nltk.sentiment.util import *
import pandas as pd
from nltk.corpus import twitter_samples
import preprocessor as p
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
import matplotlib.pyplot as plt
from dateutil import rrule
from datetime import datetime, timedelta
import unicodedata
import glob, os
import re


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# root directory
directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

# Setup blank dataframe to hold tweet output
columns = ['date','id', 'permalink', 'username', 'text','clean_text', 'retweets', 'favorites', 'mentions', 'hashtags',
           'geo', 'sentiment_vader', 'sentiment_afinn']
df = pd.DataFrame(columns=columns)

# Define a specific time period
start = datetime.strptime('2013-10-19', '%Y-%m-%d')
now = datetime.now()
# Testing pars
#now = start+timedelta(days=1)
#start = now-timedelta(days=5)

# Define the NLTK approach
sid = SentimentIntensityAnalyzer()
afinn = Afinn()

#  For each day  grab 1000 tweets. Put this into its own function
count = 765
for dt in rrule.rrule(rrule.DAILY, dtstart=start, until=now):
	print dt
	# Testing pars
	#tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince('2011-10-19').setUntil('2013-10-20').setMaxTweets(1000)
	#tweets = got.manager.TweetManager.getTweets(tweetCriteria)

	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince(dt.strftime('%Y-%m-%d'))\
		.setUntil((dt+timedelta(days=1)).strftime('%Y-%m-%d')).setMaxTweets(1000)
	# We may miss days for a wide range of reasons. For now just continue
	try:
		tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	except:
		print 'missed: ', dt
		continue

	# Write the output to dataframe
	for tweet in tweets:
		if type(tweet.text) == unicode:
			tweet_text = unicodedata.normalize('NFKD', tweet.text).encode('ascii','ignore')
		else:
			tweet_text = tweet.text
		# Clean up the text
		tweet_cleantext = p.clean(tweet_text).replace('/', '').replace('https', '').replace('http', '')
		# Calculate sentiment
		sentiment_vader = sid.polarity_scores(tweet_cleantext)['compound']
		sentiment_afinn = afinn.score(str(tweet_cleantext))
		# Enter results to dataframe
		df = df.append({'date': tweet.date, 'id': tweet.id, 'permalink':tweet.permalink, 'username': tweet.username,
		                'text': tweet.text, 'clean_text':tweet_cleantext, 'sentiment_vader':sentiment_vader,
		                'sentiment_afinn':sentiment_afinn,'retweets': tweet.retweets, 'favorites': tweet.favorites, 'mentions': tweet.mentions,
		                'hashtags': tweet.hashtags, 'geo': tweet.geo}, ignore_index=True)

	count += 1
	# Save each 15 days to a csv and clear the df
	if count % 15 == 0:
		df.to_csv(directory + 'Tweets/tweets_'+str(count)+'.csv', encoding='utf-8')
		df = pd.DataFrame(columns=columns)

# Collate the data frames into data and sentiment
# Setup blank dataframe to hold tweet output
csv_files = []
for file in glob.glob(directory + 'Tweets/*.csv'):
    csv_files.append(file)

# Reorder based on count index
csv_files = sorted_nicely(csv_files)

# Read them in and append results to a combined df
df = pd.read_csv(csv_files[0], usecols=['date','sentiment_vader','sentiment_afinn'])
# Drop all the rows with 0 sentiment
df = df[df.sentiment_vader != 0]
# Need to really do some additional filtering like identifying bots.
for file in csv_files:
	print file
	df_temp = pd.read_csv(file, usecols=['date','sentiment_vader','sentiment_afinn'])
	df_temp = df_temp[df_temp.sentiment_vader != 0]
	df = df.append(df_temp)
	print len(df)

df = df.set_index(pd.to_datetime(df.date))
# Resample to per day
df_day_vader = df['sentiment_vader']
df_day_afinn = df['sentiment_afinn']
df_day_vader = (df_day_vader.resample('1d').mean() * 10)+10
df_day_afinn = df_day_afinn.resample('1d').mean()

# Write out the results to a csv
df_day_vader.to_csv(directory+'Data/Twitter/sentiment_vader.csv')

# Make a plot
plt.figure(figsize=(11, 6))
plt.gcf().subplots_adjust(bottom=0.15)
df_day_vader.plot(label='VADER')
df_day_afinn.plot(label='Afinn')
plt.axis("tight")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend(loc="upper right")
plt.savefig(directory+'Images/twitter_sentiment.png')
plt.clf()


# Rerun the VADER sentiment because we need all three features
df_sentiment = pd.DataFrame(columns=['date','clean_text'])

for file in csv_files:
	sentiment_vader_list = []
	print file
	df_temp = pd.read_csv(file, usecols=['date','clean_text'])
	for index, row in df_temp.iterrows():
		try:
			sentiment_vader = sid.polarity_scores(row.clean_text)
		except:
			sentiment_vader = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

		sentiment_vader_list.append(sentiment_vader)

	df_temp_sentiment = pd.DataFrame.from_dict(sentiment_vader_list)
	df_temp = df_temp.join(df_temp_sentiment)
	df_sentiment = df_sentiment.append(df_temp)

df_sentiment = df_sentiment.set_index(pd.to_datetime(df_sentiment.date))
df_sentiment.to_csv(directory+'Data/Twitter/sentiment_vader_all.csv')

# Resample every 3 days
df_sentiment3d = df_sentiment.resample('3d').mean()

# Write out the results to a csv
df_sentiment3d.to_csv(directory+'Data/Twitter/sentiment_vader_all_3d.csv')




# Test the NLTK VADER approach on the twitter corpus. Do we accurately retrieve +ve or -ve reviews?
# http://www.nltk.org/api/nltk.sentiment.html
false_detections_ss = 0
false_detections_af = 0

tweets_neg = twitter_samples.strings('negative_tweets.json')
for string in tweets_neg:
	ss = sid.polarity_scores(string)
	af = afinn.score(string)
	if ss['compound'] > 0.0:
		false_detections_ss += 1
	if af > 0.0:
		false_detections_af += 1

tweets_pos = twitter_samples.strings('positive_tweets.json')
for string in tweets_pos:
	ss = sid.polarity_scores(string)
	af = afinn.score(string)
	if ss['compound'] < 0.0:
		false_detections_ss += 1
	if af < 0.0:
		false_detections_af += 1

print 'accuracy VADER: ', 100 - (float(false_detections_ss) / (len(tweets_pos) +len(tweets_neg))*100)
print 'accuracy Afinn: ', 100 - (float(false_detections_af) / (len(tweets_pos) +len(tweets_neg))*100)

