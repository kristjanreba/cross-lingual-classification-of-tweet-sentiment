'''
Clean twitter data.
- merge sentiment with twitter text data on TweetID
- remove links
- remove mention words
- remove reserved words (RT,FV)
- duplicate tweets (some tweets have 2 different sentiments)
'''

import pandas as pd
import preprocessor as p

def merge(lang):
    '''
    Merge sentiment with actual Tweet on TweetID.
    Discard lines where Tweet does not exist.
    '''
    sentiment_file = 'data/twitter_sentiment/{}_Twitter_sentiment.csv'.format(lang)
    tweet_file = 'data/tweets/{}.csv'.format(lang)
    merged_file = 'data/merged/{}.csv'.format(lang)

    sent = pd.read_csv(sentiment_file)
    tweet = pd.read_csv(tweet_file, names=['TweetID', 'Text'])

    merged = pd.merge(tweet, sent, on='TweetID')
    merged.drop(['AnnotatorID', 'TweetID'], axis=1, inplace=True)

    merged.to_csv(merged_file)

def clean(lang):
    merged_file = 'data/merged/{}.csv'.format(lang)
    clean_file = 'data/merged_clean/{}.csv'.format(lang)
    merged = pd.read_csv(merged_file, engine='python')
    merged.drop_duplicates(['Text'], inplace=True)

    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)

    text = []
    labels = []
    for index, row in merged.iterrows():
        try:
            label = row['HandLabel']
            clean_row = p.clean(row['Text'])
            text.append(clean_row)
            labels.append(label)
            if index % 1000 == 0: print(index)
        except:
            continue
    cleaned = pd.DataFrame({'Text':text, 'HandLabels':labels})
    cleaned.to_csv(clean_file)

languages = ['Croatian', 'Polish', 'English']
for lang in languages:
    print('Merging Tweets in {}'.format(lang))
    merge(lang)
    print('Cleaning Tweets in {}'.format(lang))
    clean(lang)
