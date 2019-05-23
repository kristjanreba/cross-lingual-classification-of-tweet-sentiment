import pandas as pd
import csv

languages = ['Albanian', 'Bosnian', 'Bulgarian', 'Croatian', 'English', 'German', 'Hungarian', 'Polish', 'Portuguese', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Spanish', 'Swedish']

for lang in languages:
    df = pd.read_csv('data/clean/{}.csv'.format(lang))
    text = df['Text']
    labels = df['HandLabels']

    file_text = 'data/clean/text/{}.csv'.format(lang)
    file_labels = 'data/clean/labels/{}.csv'.format(lang)

    text.to_csv(file_text, header=None, index=None)
    labels.to_csv(file_labels, header=None, index=None)
