"""
This is my practices in Python script to test out time series
pip freeze >config/requirements.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load
df = read_csv('src/data/AirPassengers.csv', usecols=[1])
plt.plot(df)

# Convert to array
df_ar = df.values
df_ar = df_ar.astype('float32')

# Normalization
transform = MinMaxScaler(feature_range=(0,1))
df_ar = transform.fit_transform(df_ar)

# Train/Test
train_size = int(len(df_ar) * 0.7)
test_size = len(df_ar) - train_size
train, test = df_ar[0:train_size,:], df_ar [train_size:len(df_ar),:]

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)

# Size to look back
seq_size = 5

trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)
