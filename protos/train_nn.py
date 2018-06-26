import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

df = pd.read_csv('../input/creditcard.csv')

train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['Class'])

X_train = train.drop(['Class'], axis=1)
X_test = test.drop(['Class'], axis=1)
Y_train = train['Class']
Y_test = test['Class']

print('X_train: {}'.format(X_train.shape))
print('Y_train: {}'.format(Y_train.shape))
print('X_test: {}'.format(X_test.shape))
print('Y_test: {}'.format(Y_test.shape))

print(Y_train.head())

Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

np.random.seed(0)

model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training and test
epoch = 10
batch_size = 2048
model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size)

score, acc = model.evaluate(X_test, Y_test)
print('Test score:', score)
print('Test accuracy:', acc)

history = model.fit(X_train, Y_train, epochs=20, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)
history.history.keys()

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Testing loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Testing accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

