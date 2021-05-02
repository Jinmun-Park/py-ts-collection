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

print("Shape of training set : {}".format(trainX.shape))
print("Shape of test set : {}".format(testX.shape))
print(trainX.shape[1])

# Modelling (Large Dens can cause overfitting)

model = Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu')) #12
#model.add(Dense(32, activation='relu')) #8
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
print(model.summary())

# Fit & Prediction
model.fit(trainX, trainY, validation_data=(testX, testY), verbose=2, epochs=100)

trainpredict = model.predict(trainX)
testpredict = model.predict(testX)

TrainPredict = transform.inverse_transform(trainpredict)
TrainY_inverse = transform.inverse_transform([trainY])
TestPredict = transform.inverse_transform(testpredict)
TestY_inverse = transform.inverse_transform([testY])

# RMSE
trainscore = math.sqrt(mean_squared_error(TrainY_inverse[0], TrainPredict[:,0]))
print('Train Score : %.2f RMSE' % (trainscore))
testscore = math.sqrt(mean_squared_error(TestY_inverse[0], TestPredict[:,0]))
print('Test Score : %.2f RMSE' % (testscore))

# Plot
trainpredictplot = np.empty_like(df_ar)
trainpredictplot[:, :] = np.nan
trainpredictplot[seq_size:len(TrainPredict)+seq_size, :] = TrainPredict

testpredictplot = np.empty_like(df_ar)
testpredictplot[:, :] = np.nan
testpredictplot[len(TrainPredict)+(seq_size*2)+1:len(df_ar)-1,:] = TestPredict

plt.plot(transform.inverse_transform(df_ar))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()