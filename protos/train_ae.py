import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../input/creditcard.csv')

X_train, X_test = train_test_split(df, test_size=0.2, random_state=2)
# semi supervised training
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
Y_train = X_train['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values


