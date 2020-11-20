import pandas as pd
import numpy as np
import subprocess

# list of languages
#languages = ['Albanian', 'Bosnian', 'Bulgarian', 'Croatian', 'English', 'German', 'Hungarian', 'Polish', 'Portuguese', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Swedish']
languages = ['Slovenian', 'Swedish']

# list of language abbreviations
#abbreviations = ['sq', 'bs', 'bg', 'hr', 'en', 'de', 'hu', 'pl', 'pt', 'ru', 'sr', 'sk', 'sl', 'sv']
abbreviations = ['sl', 'sv']

for lang, abrev in zip(languages, abbreviations):
    # encode text and write to file
    infile = '../data/clean/text/{}.csv'.format(lang)
    outfile = '../data/embed/{}.raw'.format(lang)
    subprocess.call('./embed.sh %s %s %s' % (infile, abrev, outfile), shell=True)
