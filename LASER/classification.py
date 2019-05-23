import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense



dim = 1024
X = np.fromfile("../data/embed/Slovenian.raw", dtype=np.float32, count=-1)
X.resize(X.shape[0] // dim, dim)
print(X.shape)
print(X[:2])

#-------------------------------------------------------------------------------
# load data

pd.read_csv()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#-------------------------------------------------------------------------------
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#-------------------------------------------------------------------------------
# train Classifier

model.fit(X_train, y_train, epochs=150, batch_size=10)


#-------------------------------------------------------------------------------
# evaluate Classifier

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
