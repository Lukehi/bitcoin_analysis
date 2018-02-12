# Twitter Bitcoin sentiment analysis
#https://www.data-blogger.com/2017/02/22/getting-rich-using-bitcoin-stockprices-and-twitter/
#https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
#https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/


import tweepy
from tweepy import OAuthHandler

consumer_key = 'YOUR-CONSUMER-KEY'
consumer_secret = 'YOUR-CONSUMER-SECRET'
access_token = 'YOUR-ACCESS-TOKEN'
access_secret = 'YOUR-ACCESS-SECRET'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text)