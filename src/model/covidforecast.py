"""
LSTM (Many Approaches) :
https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/

keras time series generator ;
https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/

multi step LSTM :
https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

Limitation of LSTM :
https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/


Next Step-> CNN
https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/

Creating package after studying codes from URL and Youtube ;
https://www.youtube.com/watch?v=97bZKO6cJfg&t=870s&ab_channel=DigitalSreeni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler #Step 3
from keras.preprocessing.sequence import TimeseriesGenerator #Step 4
from keras.models import Sequential #Step 5
from keras.layers import Dense, LSTM

pd.set_option('display.max_rows', None)

df_confirmed = pd.read_csv('src/data/time_series_covid19_confirmed_global.csv')
print(df_confirmed.columns)
print(df_confirmed['Country/Region'].unique)
country = "US"

## Step 1 : Data Sreucture

df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),columns=["confirmed"])
df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index,format='%m/%d/%y')

df_confirmed_country.plot(figsize=(10,5),title="COVID confirmed cases")

df_confirmed_country.head(10)
df_confirmed_country.tail(10)
print("First 10 days", df_confirmed_country.head(10))
print("Last 10 days", df_confirmed_country.tail(10))
print("Total days in the dataset", len(df_confirmed_country))

## Step 2 : Training & Test (14 days interval)
x = len(df_confirmed_country)-14

train = df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

## Step 3 : Normalization
scaler = MinMaxScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

## Step 4 : TS conversion
seq_size = 7  ## number of steps (lookback)
n_features = 1 ## number of features. This dataset is univariate so it is 1

train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(train))
print("Total number of samples in the generated data = ", len(train_generator))

test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(test)) # 14 as we're using last 14 days for test
print("Total number of samples in the generated data = ", len(test_generator)) # 7

## Step 5 : Modelling
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(seq_size, n_features)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

## Step 6 : Fitting
'''
fit_generator will be depreciated, check this out ;
https://stackoverflow.com/questions/60585948/convert-model-fit-generator-to-model-fit
'''

history = model.fit(train_generator,
                              validation_data=test_generator,
                              epochs=50, steps_per_epoch=10)


loss = history.history['loss']
val_loss = history.history['val_loss']
print('This is training loss', len(loss))
print('This is validation loss', len(val_loss))
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
1) Underfitting

This is the only case where loss > validation_loss, but only slightly, if loss is far higher than validation_loss, please post your code and data so that we can have a look at

2) Overfitting

loss << validation_loss

This means that your model is fitting very nicely the training data but not at all the validation data, in other words it's not generalizing correctly to unseen data

3) Perfect fitting

loss == validation_loss

If both values end up to be roughly the same and also if the values are converging (plot the loss over time) then chances are very high that you are doing it right
"""

## Step 7 : Forecast

prediction = []

current_batch = train_scaled[-seq_size:] #Final data points in train
current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

# Predict future, beyond test dates
future = 7 #Days
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) #np.append

# Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index  #Get dates for test data

# Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data
df_forecast = pd.DataFrame(columns=["actual_confirmed","predicted"], index=time_series_array)

df_forecast.loc[:,"predicted"] = rescaled_prediction[:,0]
df_forecast.loc[:,"actual_confirmed"] = test["confirmed"]

# Plot
df_forecast.plot(title="Predictions for next 7 days")