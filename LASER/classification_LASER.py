import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from collections import Counter


#-------------------------------------------------------------------------------
dim = 1024
epochs = 10
batch_size = 32


def get_X(lang):
    X = np.fromfile('../data/embed/{}.raw'.format(lang), dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    X = X[1:,:]
    return X

def get_Y(lang):
    df = pd.read_csv('../data/clean/labels/{}.csv'.format(lang))
    y_text = df.values
    y_text = np.ravel(y_text)
    return y_text

def avg_f1_score(y_true, y_pred, encoder_dict):
    #score = f1_score(y_true, y_pred, labels=[0,1,2], average=None)
    #print(score)
    #return score
    # get average F1 for postive and negative F1 scores
    scores = f1_score(y_true, y_pred, average=None)
    f1_negative = scores[encoder_dict['Negative']]
    f1_positive = scores[encoder_dict['Positive']]
    return (f1_negative + f1_positive) / 2.0

def create_model():
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_data(train_lang, test_lang, use_test_lang):
    # create X_train matrix for training languages
    #print('Loading X_train.')
    X_train = np.array([])
    for lang in train_lang:
        X = get_X(lang)
        X_train = np.vstack([X_train, X]) if X_train.size else X
    #print(X_train.shape)

    #print('Loading X_test.')
    X_test = get_X(test_lang)
    len_test = X_test.shape[0]

    # create Y matrix that stacks train and test languages
    #print('Loading Y_train and Y_test.')
    len_train = 0
    y = np.array([])
    for lang in train_lang:
        # read the hand labels for a language
        y_text = get_Y(lang)
        y = np.concatenate([y, y_text], axis=0) if y.size else y_text
        len_train += y_text.shape[0]
        #print('Shape y_text: ', y_text.shape)

    y_text = get_Y(test_lang)
    #print(y_text.shape)
    #print(y.shape)
    y = np.concatenate([y, y_text], axis=0)


    print('len_train: ', len_train)
    print('len_test: ', len_test)
    print(y.shape)

    #X, y = shuffle(X, y, random_state=0)

    # encode class names as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # create dictionary for mapping
    encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    #print(encoder_dict)

    # convert integers to one hot encoded
    Y_one_hot = np_utils.to_categorical(encoded_Y, num_classes=3)

    # if we use the test language training option than we have to adjust how the dataset is split
    if use_test_lang:
        X_train = np.vstack([X_train, X_test[:int(len_test*0.7),:]])
        X_test = X_test[int(len_test*0.7):,:]
        len_train += int(len_test*0.7)

    # split encoded labels back to train and test
    Y_train = Y_one_hot[:len_train,:]
    Y_test = Y_one_hot[len_train:,:]

    #print('X_train.shape:', X_train.shape)
    #print('Y_train.shape:', Y_train.shape)
    #print('X_test.shape:', X_test.shape)
    #print('Y_test.shape:', Y_test.shape)
    return X_train, Y_train, X_test, Y_test, encoder_dict

def load_single(lang):
    X = np.fromfile('../data/embed/{}.raw'.format(lang), dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    X = X[1:,:]
    #print('Language: ', lang)
    #print('Shape X: ', X.shape)

    # read the hand labels for a language
    df = pd.read_csv('../data/clean/labels/{}.csv'.format(lang))
    y_text = df.values
    y_text = np.ravel(y_text)
    #print('Shape y_text: ', y_text.shape)

    # encode class names as integers
    encoder = LabelEncoder()
    encoder.fit(y_text)
    encoded_Y = encoder.transform(y_text)
    # create dictionary for mapping
    encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    # convert integers to one hot encoded
    Y_one_hot = np_utils.to_categorical(encoded_Y, num_classes=3)
    #print('Shape Y_train: ', Y_one_hot.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.3, random_state=1)

    #print('X_train.shape:', X_train.shape)
    #print('Y_train.shape:', Y_train.shape)
    #print('X_test.shape:', X_test.shape)
    #print('Y_test.shape:', Y_test.shape)
    return X_train, Y_train, X_test, Y_test, encoder_dict

def experiment_single_lang(lang):
    # load data
    X_train, Y_train, X_test, Y_test, encoder_dict = load_single(lang)

    # create model
    model = create_model()

    # train Classifier
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # make predictions
    Y_pred = model.predict(X_test)

    # convert from one-hot encoding
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    # evaluate classifier
    print('Language: ', lang)
    print('avg F1 = ', avg_f1_score(y_true, y_pred, encoder_dict))
    print('acc = ', accuracy_score(y_true, y_pred))
    print('-------------------------------------------------------------------')

def experiment(train_langs, test_lang, use_test_lang=False):
    # load data
    X_train, Y_train, X_test, Y_test, encoder_dict = load_data(train_langs, test_lang, use_test_lang)

    # create model
    model = create_model()

    # train Classifier
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # make predictions
    Y_pred = model.predict(X_test)

    # convert from one-hot encoding
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    avg_f1 = avg_f1_score(y_true, y_pred, encoder_dict)
    acc = accuracy_score(y_true, y_pred)

    # evaluate classifier
    print('Training on languages: ', train_langs)
    print('Evaluatng on language: ', test_lang)
    print('avg F1 = ', avg_f1)
    print('acc = ', acc)
    print('-------------------------------------------------------------------')
    return acc, avg_f1

def majority_classifier_experiment(lang):
    y = list(get_Y(lang))
    c = Counter(y)
    acc = max(c.values()) / len(y)

    print('MC acc = ', acc)
    return acc

if __name__ == "__main__":

    results_file = '../results.txt'

    languages = ['Albanian', 'Bosnian', 'Bulgarian', 'Croatian', 'English', 'German', 'Hungarian', 'Polish', 'Portuguese', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Swedish']

    experiments_same_fam = [
        (['English'], 'German'),
        (['Serbian'], 'Slovenian'),
        (['Serbian'], 'Croatian'),
        (['Serbian'], 'Bosnian'),
        (['Polish'], 'Slovenian'),
        (['Slovak'], 'Slovenian'),
        (['Croatian'], 'Slovenian'),
        (['Croatian'], 'Serbian'),
        (['Croatian'], 'Bosnian'),
        (['Slovenian'], 'Croatian'),
        (['Slovenian'], 'Serbian'),
        (['Slovenian'], 'Bosnian'),
    ]

    experiments_diff_lang_fam = [
        (['German'], 'Slovenian'),
        (['English'], 'Slovenian'),
        (['Swedish'], 'Slovenian'),
        (['Hungarian'], 'Slovenian'),
        (['Portuguese'], 'Slovenian'),
    ]

    experiments_large_train_dataset = [
        (['Croatian', 'Serbian', 'Bosnian'], 'Slovenian'),
        (['English', 'Swedish'], 'German'),
    ]

    with open(results_file, 'a+') as f:

        '''
        f.write('\nmajority classifier accuracy\n')
        for l in languages:
            acc = majority_classifier_experiment(l)
            f.write("{} acc:{}\n".format(l, acc))
        '''
        '''
        f.write('\nexperiments_same_fam\n')
        for train_langs, test_lang in experiments_same_fam:
            acc, f1 = experiment(train_langs, test_lang, use_test_lang=True)
            f.write("{} {} acc:{:.2f}, f1:{:.2f}\n".format(train_langs, test_lang, acc, f1))
        
        f.write('\nexperiments_diff_lang_fam\n')
        for train_langs, test_lang in experiments_diff_lang_fam:
            acc, f1 = experiment(train_langs, test_lang, use_test_lang=True)
            f.write("{} {} acc:{:2f}, f1:{:.2f}\n".format(train_langs, test_lang, acc, f1))

        f.write('\nexperiments_large_train_dataset\n')
        for train_langs, test_lang in experiments_large_train_dataset:
            acc, f1 = experiment(train_langs, test_lang, use_test_lang=True)
            f.write("{} {} acc:{:2f}, f1:{:.2f}\n".format(train_langs, test_lang, acc, f1))
        '''

    '''
    # Experiments to use every language available to train and use the test language also
    print("\n\nFULL DATASET\n\n")

    for lang in languages:
        train_lang = languages.copy()
        train_lang.remove(lang)
        test_lang = lang
        experiment(train_lang, test_lang, use_test_lang=True)
    '''

    
