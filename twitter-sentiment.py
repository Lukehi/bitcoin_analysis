# Twitter Bitcoin sentiment analysis

import logging
import got
from nltk.sentiment.util import *
import datetime
import pandas as pd
import regex
from textblob import TextBlob
from nltk.corpus import twitter_samples

from dateutil import rrule
from datetime import datetime, timedelta


def remove_by_regex(tweets, regexp):
	tweets.loc[:, "Tweets"].replace(regexp, "", inplace=True)
	return tweets


now = datetime.now()
start = now - timedelta(days=6*365)

columns = ['date','id', 'permalink', 'username', 'text', 'retweets', 'favorites', 'mentions', 'hashtags', 'geo']
df = pd.DataFrame(columns=columns)
# Extract top 50 tweets for each day from start to now store results in df
for dt in rrule.rrule(rrule.DAILY, dtstart=start, until=now):
	print dt
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince(dt.strftime('%Y-%m-%d'))\
		.setUntil((dt+timedelta(days=1)).strftime('%Y-%m-%d')).setTopTweets(True).setMaxTweets(10)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	# write the output to pandas
	for tweet in tweets:
		print tweet.date
		df = df.append({'date': tweet.date, 'id': tweet.id, 'permalink':tweet.permalink, 'username': tweet.username,
		                'text': tweet.text, 'retweets': tweet.retweets, 'favorites': tweet.favorites,
		                'mentions': tweet.mentions, 'hashtags': tweet.hashtags, 'geo': tweet.geo}, ignore_index=True)


# Cleanup the text


# Run NLTK VADER

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