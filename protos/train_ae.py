import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

df = pd.read_csv('../input/creditcard.csv')
#count_classes = pd.DataFrame(pd.value_counts(df['Class'], sort = True).sort_index())
#print(count_classes)

train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['Class'])

X_train = train.drop(['Class'], axis=1)
X_test = test.drop(['Class'], axis=1)
y_train = train['Class']
y_test = test['Class']

print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))
