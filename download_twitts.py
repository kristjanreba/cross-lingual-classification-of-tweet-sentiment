import pandas as pd
import tweepy
import time


# You must define your own Twitter api keys
#CONSUMER_KEY = ''
#CONSUMER_SECRET = ''
#OAUTH_TOKEN = ''
#OAUTH_TOKEN_SECRET = ''

# connect to twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, retry_count=100, retry_delay=60, wait_on_rate_limit=True)

def download_tweets(lang):
    id_file = 'data/twitter_sentiment/{}_Twitter_sentiment.csv'.format(lang)
    new_file = 'data/tweets/{}.csv'.format(lang)

    # read tweet ids
    df = pd.read_csv(id_file)
    ids = list(df['TweetID'])

    f = open(new_file, 'a')

    num_chunks = len(ids) // batch_size + 1
    for i in range(num_chunks): # loop over batches of tweets
        ids_batch = ids[i*batch_size:(i+1)*batch_size]
        try:
            tweets = api.statuses_lookup(id_=ids_batch, include_entities=False, trim_user=True) # fetch as little metadata as possible
        except:
            time.sleep(60)
            i = i - 1
            continue

        if len(ids_batch) != len(tweets): print('{} unexpected response size {}, expected {}'.format(i, len(tweets), len(ids_batch)))

        # write to file
        for tweet in tweets:
            text = tweet.text.replace('"', '\'') # replace quotation
            text = text.replace('\n', '') # remove new line
            line = '{},"{}"\n'.format(tweet.id, text)
            f.write(line)

    f.close()


# batch size depends on Twitter limit, 100 at this time
batch_size = 100
languages = ['Slovenian', 'Slovak', 'Serbian', 'Bosnian', 'Croatian', 'Russian', 'Polish', 'German', 'English', 'Swedish', 'Hungarian', 'Bulgarian', 'Portuguese']

for lang in languages:
    print('Downaloding Tweets in {}'.format(lang))
    download_tweets(lang)
