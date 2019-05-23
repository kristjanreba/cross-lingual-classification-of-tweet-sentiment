'''
Clean twitter data.
- merge sentiment with twitter text data on TweetID
- remove links
- remove duplicate Tweets
- remove mention words (eg. @username)
- remove reserved words (RT, FV)
- duplicate tweets (some tweets have 2 different sentiments) -> removed the
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
    '''
    Create new folder for cleaned data.
    Remove duplicate Tweets and do some tweet cleaning.
    '''
    merged_file = 'data/TweetText_Label/{}_tweet_label.csv'.format(lang)
    clean_file = 'data/clean/{}.csv'.format(lang)
    merged = pd.read_csv(merged_file, engine='python')
    #merged.drop_duplicates(['Tweet text'], inplace=True)

    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)

    text = []
    labels = []
    for index, row in merged.iterrows():
        try:
            label = row['SentLabel']
            clean_row = p.clean(row['Tweet text'])
            text.append(clean_row)
            labels.append(label)
            if index % 1000 == 0: print(index)
        except:
            continue
    cleaned = pd.DataFrame({'Text':text, 'HandLabels':labels})
    cleaned.to_csv(clean_file)



'''
Merge with sentiment and clean for languages in list
'''
languages = ['Albanian', 'Bosnian', 'Bulgarian', 'Croatian', 'English', 'German', 'Hungarian', 'Polish', 'Portuguese', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Spanish', 'Swedish']
for lang in languages:
    #print('Merging Tweets in {}'.format(lang))
    #merge(lang)
    print('Cleaning Tweets in {}'.format(lang))
    clean(lang)
