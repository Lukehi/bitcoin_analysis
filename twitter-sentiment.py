# Twitter Bitcoin sentiment analysis

import got
from nltk.sentiment.util import *
import pandas as pd
from nltk.corpus import twitter_samples
import preprocessor as p
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from dateutil import rrule
from datetime import datetime, timedelta

directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

# Setup blank dataframe to hold tweet output
columns = ['date','id', 'permalink', 'username', 'text','clean_text', 'retweets', 'favorites', 'mentions', 'hashtags',
           'geo', 'sentiment']
df = pd.DataFrame(columns=columns)

# Extract top 50 tweets for each day from start to now. Store results in df
now = datetime.now()
#now = start+timedelta(days=10)
start = datetime.strptime('2011-09-13', '%Y-%m-%d')

sid = SentimentIntensityAnalyzer()

for dt in rrule.rrule(rrule.DAILY, dtstart=start, until=now):
	print dt
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince(dt.strftime('%Y-%m-%d'))\
		.setUntil((dt+timedelta(days=1)).strftime('%Y-%m-%d')).setTopTweets(True).setMaxTweets(50)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	# write the output to pandas
	for tweet in tweets:
		print tweet.date
		tweet_cleantext = p.clean(tweet.text.encode('utf-8')).replace('/', '').replace('https', '').replace('http', '')
		sentiment = sid.polarity_scores(tweet_cleantext)['compound']
		df = df.append({'date': tweet.date, 'id': tweet.id, 'permalink':tweet.permalink, 'username': tweet.username,
		                'text': tweet.text, 'clean_text':tweet_cleantext, 'sentiment':sentiment,
		                'retweets': tweet.retweets, 'favorites': tweet.favorites, 'mentions': tweet.mentions,
		                'hashtags': tweet.hashtags, 'geo': tweet.geo}, ignore_index=True)
		# Pickle it regularly just incase it falls over
		df.to_pickle(directory+'tweets.pickle')

# Test the NLTK VADER approach on the twitter corpus. Do we accurately retrieve +ve or -ve reviews?
# http://www.nltk.org/api/nltk.sentiment.html

false_detections = 0

tweets_neg = twitter_samples.strings('negative_tweets.json')
for string in tweets_neg:
	ss = sid.polarity_scores(string)
	if ss['compound'] > 0.0:
		false_detections += 1

tweets_pos = twitter_samples.strings('positive_tweets.json')
for string in tweets_pos:
    ss = sid.polarity_scores(string)
    if ss['compound'] < 0.0:
	    false_detections += 1

print 'accuracy: ', 100 - (float(false_detections) / (len(tweets_pos) +len(tweets_neg))*100)