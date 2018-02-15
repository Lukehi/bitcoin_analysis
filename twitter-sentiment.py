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
from pandas import Series, DataFrame, Panel


from random import randrange


def random_date(start,l):
#Randomly get a start time
   current = start
   while l >= 0:
      curr = current + timedelta(minutes=randrange(60))
      yield curr
      l-=1


directory = '/Users/lukehindson/PycharmProjects/Bitcoin/'

# Setup blank dataframe to hold tweet output
columns = ['date','id', 'permalink', 'username', 'text','clean_text', 'retweets', 'favorites', 'mentions', 'hashtags',
           'geo', 'sentiment_vader', 'sentiment_afinn']
df = pd.DataFrame(columns=columns)

# Read in a list of top financial and BTC based twitter usernames


# Extract 50 random tweets for each day from start to now that have been retweet or favourited. Store results in df
# Might need to be a bit smarter than this and extract at random time of the day because selecting the top doesnt seem to work

# Must be able to select random tweets.... Need to probably alter the TweetManager.py code

# For test define a specific time period
start = datetime.strptime('2011-09-13', '%Y-%m-%d')
now = start+timedelta(days=10)

#start = now-timedelta(days=5)
#now = datetime.now()


sid = SentimentIntensityAnalyzer()
afinn = Afinn()

for dt in rrule.rrule(rrule.DAILY, dtstart=start, until=now):
	print dt
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince('2017-11-13').setUntil('2017-11-14').setMaxTweets(40).setTopTweets(True)

	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince('2016-02-01').setUntil('2016-02-02').setMaxTweets(100)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)


	tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setSince(dt.strftime('%Y-%m-%d'))\
		.setUntil((dt+timedelta(days=1)).strftime('%Y-%m-%d')).setTopTweets(True).setMaxTweets(50)
	# write the output to pandas
	for tweet in tweets:
		print tweet.date
		if type(tweet.text) == unicode:
			tweet_text = unicodedata.normalize('NFKD', tweet.text).encode('ascii','ignore')
		else:
			tweet_text = tweet.text
		tweet_cleantext = p.clean(tweet_text).replace('/', '').replace('https', '').replace('http', '')
		sentiment_vader = sid.polarity_scores(tweet_cleantext)['compound']
		sentiment_afinn = afinn.score(str(tweet_cleantext))
		df = df.append({'date': tweet.date, 'id': tweet.id, 'permalink':tweet.permalink, 'username': tweet.username,
		                'text': tweet.text, 'clean_text':tweet_cleantext, 'sentiment_vader':sentiment_vader,
		                'sentiment_afinn':sentiment_afinn,'retweets': tweet.retweets, 'favorites': tweet.favorites, 'mentions': tweet.mentions,
		                'hashtags': tweet.hashtags, 'geo': tweet.geo}, ignore_index=True)
		# Pickle it regularly just incase it falls over
		df.to_pickle(directory+'tweets.pickle')

df.to_csv(directory+'tweets.csv')
# Resample to per day
df_day = df['sentiment_vader']
df_day.index = df['date']
df_day.sort_index()
df_day = df_day.resample('1d').mean()
df_day.plot()
plt.show()


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

# Check we are actually getting the top tweets
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('Bitcoin').setSince("2017-01-01").setUntil(
	"2017-12-31").setMaxTweets(1).setTopTweets(True)
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("Bitcoin").setTopTweets(True).setMaxTweets(10).setSince("2017-01-01").setUntil(
	"2017-12-31")
# first one
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
