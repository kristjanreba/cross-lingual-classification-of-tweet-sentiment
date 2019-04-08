import pandas as pd
import numpy as np
import os
import sys
import argparse

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

def metrices(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    '''
    acc1 = 1
    avg_f1 = f1_score(y_true, y_pred, average='micro')
    alpha = 1
    scores = {
        'Acc': acc,
        'Acc+1': acc1,
        'Avg_F1':avg_f1,
        'Alpha': alpha
    }
    '''
    print(acc)
    return acc

def cross_validation(X, y, folds=10, shuffle=False):
    skf = StratifiedKFold(n_splits=folds, shuffle=shuffle)
    for train_index, test_index in skf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        '''
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrices(y_true, y_pred)
        '''






# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder, EncodeLoad, EncodeFile
from text_processing import Token, BPEfastApply, SplitLines, JoinEmbed

###############################################################################

parser = argparse.ArgumentParser('LASER: calculate embeddings for MLDoc')
parser.add_argument(
    '--data_dir', type=str, default='embed',
    help='Base directory for created files')

# options for encoder
parser.add_argument(
    '--encoder', type=str, required=True,
    help='Encoder to be used')
parser.add_argument(
    '--bpe_codes', type=str, required=True,
    help='Directory of the tokenized data')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help="List of languages to test on")
parser.add_argument(
    '--buffer-size', type=int, default=10000,
    help='Buffer size (sentences)')
parser.add_argument(
    '--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument(
    '--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument(
    '--cpu', action='store_true',
    help='Use CPU instead of GPU')
parser.add_argument(
    '--verbose', action='store_true',
    help='Detailed output')
args = parser.parse_args()

print('LASER: calculate embeddings')

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

enc = EncodeLoad(args)


languages = ['Slovenian']

print('\nProcessing:')
for part in ('train1000', 'dev', 'test'):
    # for lang in "en" if part == 'train1000' else args.lang:
    for lang in languages:
        cfname = os.path.join(args.data_dir, 'mldoc.' + part)
        Token(cfname + '.txt.' + lang,
              cfname + '.tok.' + lang,
              lang=lang,
              romanize=(True if lang == 'el' else False),
              lower_case=True, gzip=False,
              verbose=args.verbose, over_write=False)
        SplitLines(cfname + '.tok.' + lang,
                   cfname + '.split.' + lang,
                   cfname + '.sid.' + lang)
        BPEfastApply(cfname + '.split.' + lang,
                     cfname + '.split.bpe.' + lang,
                     args.bpe_codes,
                     verbose=args.verbose, over_write=False)
        EncodeFile(enc,
                   cfname + '.split.bpe.' + lang,
                   cfname + '.split.enc.' + lang,
                   verbose=args.verbose, over_write=False,
                   buffer_size=args.buffer_size)
        JoinEmbed(cfname + '.split.enc.' + lang,
                  cfname + '.sid.' + lang,
                  cfname + '.enc.' + lang)
