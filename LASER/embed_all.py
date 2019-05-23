import pandas as pd
import numpy as np
import subprocess

# list of languages
languages = ['Russian', 'Serbian', 'Slovak', 'Swedish']

# list of language abbreviations
abbreviations = ['ru', 'sr', 'sk', 'sv']

for lang, abrev in zip(languages, abbreviations):
    # encode text and write to file
    infile = '../data/clean/text/{}.csv'.format(lang)
    outfile = '../data/embed/{}.raw'.format(lang)
    subprocess.call('./embed.sh %s %s %s' % (infile, abrev, outfile), shell=True)
