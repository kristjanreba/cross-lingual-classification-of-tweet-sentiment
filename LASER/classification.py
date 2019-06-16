import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


#-------------------------------------------------------------------------------
# load data
# read embedded text data
lang = 'Slovenian'
dim = 1024
X = np.fromfile('../data/embed/Slovenian.raw', dtype=np.float32, count=-1)
X.resize(X.shape[0] // dim, dim)
X = X[1:,:]
print('Language: ', lang)
print('Shape X: ', X.shape)

# read the hand labels for a language
df = pd.read_csv('../data/clean/labels/Slovenian.csv')
y_text = df.values
y_text = np.ravel(y_text)
print('Shape y_text: ', y_text.shape)

# encode class names as integers
encoder = LabelEncoder()
encoder.fit(y_text)
encoded_Y = encoder.transform(y_text)
# create dictionary for mapping
encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# convert integers to one hot encoded
Y_one_hot = np_utils.to_categorical(encoded_Y, num_classes=3)
print('Shape Y_train: ', Y_one_hot.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.3, random_state=1)

#-------------------------------------------------------------------------------
# create model
model = Sequential()
model.add(Dense(8, input_dim=dim, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#-------------------------------------------------------------------------------
# train Classifier
model.fit(X_train, Y_train, epochs=1, batch_size=32)

#-------------------------------------------------------------------------------
# evaluate Classifier

# make predictions
Y_pred = model.predict(X_test)

# convert from one-hot encoding
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

def avg_f1_score(y_true, y_pred, encoder_dict):
    scores = f1_score(y_true, y_pred, average=None)
    # get average F1 for postive and negative F1 scores
    f1_negative = scores[encoder_dict['negative']]
    f1_positive = scores[encoder_dict['positive']]
    return (f1_negative * f1_positive) / 2.0

print('avg F1 = ', avg_f1_score(y_true, y_pred, encoder_dict))
print('acc = ', accuracy_score(y_true, y_pred))
#confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
