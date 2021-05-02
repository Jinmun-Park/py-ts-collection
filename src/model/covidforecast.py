import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler #Step 3
from keras.preprocessing.sequence import TimeseriesGenerator #Step 4
from keras.models import Sequential, Model #Step 5
from keras.layers import LSTM, Input, Dropout #Step 5
from keras.layers import Dense #Step 5
from keras.layers import RepeatVector #Step 5
from keras.layers import TimeDistributed #Step 5
import seaborn as sns

# data_import_setup
dir_path = 'src/data/time_series_covid19_confirmed_global.csv'
country = "US"

def data_import(dir_path, country, case_plot):

    print('Your directory path is :', dir_path)
    print('Selected country is :', country)
    df_confirmed = pd.read_csv(dir_path)
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),columns=["confirmed"])
    df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index, format='%m/%d/%y')

    if case_plot == 1:
        print('Completed showing your selected country confirmed Covid19 cases')
        df_confirmed_country.plot(figsize=(10,5),title="COVID confirmed cases")
    else:
        print('You set default in plot. Please key in 1 to show confirmed covid 19 cases in your selected countries')

    print("First 10 days", df_confirmed_country.head(10))
    df_confirmed_country.head(10)
    print("Last 10 days", df_confirmed_country.tail(10))
    df_confirmed_country.tail(10)

    print("This is total days in the dataset", len(df_confirmed_country))

    return df_confirmed_country

##################################################################### data import #####################################################################
df_confirmed_country = data_import(dir_path = dir_path, country = country, case_plot = 1)
#######################################################################################################################################################

data = df_confirmed_country
seq_size = 10
neurons = 128
anomaly = 1
validation_plot = 1
fcst = 1

#https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

def data_modelling(data, seq_size, neurons, validation_plot, fcst):

    ### Data Preprocessing

    ## Step 2 : Training & Test (14 days interval)
    x = len(data) - 14

    train = data.iloc[:x]
    test = data.iloc[x:]

    ## Step 3 : Normalization
    scaler = MinMaxScaler()
    # scaler = StandardScalaer() #Removing mean standardize variance
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    ## Step 4 : TS conversion
    print('Your number of steps (lookback) set up in your TSgenerator is  :', seq_size)
    print('This will lookback 7 times in your datasets')
    print('Your batch_size will be 1 by default (Time Series LSTM)')

    train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=seq_size, batch_size=1)
    print("Total number of samples in the original training data = ", len(train))
    print("Total number of samples in the generated data = ", len(train_generator))

    test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
    print("Total number of samples in the original training data = ",
          len(test))  # 14 as we're using last 14 days for test
    print("Total number of samples in the generated data = ", len(test_generator))  # 7

    first_neurons = neurons
    print('Your first neurons selected in your first layer in LSTM is :', first_neurons)
    second_neurons = int(neurons / 2)
    print('Your first neurons selected in your first layer in LSTM is :', second_neurons)
    first_dense = int((neurons / 2) / 2)
    print('Your first dense selected in your first layer in LSTM is :', first_dense)
    second_dense = 1
    print('Your second and last dense selected in your first layer in LSTM is :', second_dense)

    ### Data Modelling

    ## Step 5 : Modelling

    # Batch Size and Epoch explanation : https://www.youtube.com/watch?v=Y-zswp6Yxf0&ab_channel=DigitalSreeni
    n_features = 1
    print('By default, your number of features set up in your modelling is  :', n_features)

    model = Sequential()
    model.add(LSTM(first_neurons, activation='relu', return_sequences=True, input_shape=(seq_size, n_features)))
    model.add(LSTM(second_neurons, activation='relu'))
    model.add(Dense(first_dense))
    model.add(Dense(second_dense))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    ## Step 6 : Fitting
    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=50, steps_per_epoch=10)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    if validation_plot == 1:
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        print('You set default in plot. Please key in 1 to show your training & validation loss')

    ### Model Forecasting

    ## Step 7 : Forecast
    if fcst == 1:

        prediction = []

        current_batch = train_scaled[-seq_size:]  # Final data points in train
        current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

        # Predict future, beyond test dates

        future = int(input("You have selected to produce forecast. Please type in your forecast period. Example) 7"))
        print("Successfully completed key in your forecast period :" + str(future))

        for i in range(len(test) + future):
            current_pred = model.predict(current_batch)[0]
            prediction.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)  # np.append

        # Inverse transform to before scaling so we get actual numbers
        rescaled_prediction = scaler.inverse_transform(prediction)

        time_series_array = test.index  # Get dates for test data

        # Add new dates for the forecast period
        for k in range(0, future):
            time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

        # Create a dataframe to capture the forecast data
        df_forecast = pd.DataFrame(columns=["actual_confirmed", "predicted"], index=time_series_array)

        df_forecast.loc[:, "predicted"] = rescaled_prediction[:, 0]
        df_forecast.loc[:, "actual_confirmed"] = test["confirmed"]

        # Plot
        df_forecast.plot(title="Predictions for next 7 days")
        plt.show()
    else:
        print('You set default in plot. Please key in 1 to perform forecasting and plot')

##################################################################### data modelling #####################################################################
data_modelling(data=df_confirmed_country, seq_size = 10, neurons = 128, validation_plot = 1, fcst = 1)
##########################################################################################################################################################


def anomaly_modelling(data, seq_size, neurons, validation_plot, fcst):

    x = len(data) - 14

    train = data.iloc[:x]
    test = data.iloc[x:]

    scaler = MinMaxScaler()
    scaler.fit(data)

    train['confirmed'] = scaler.transform(train[['confirmed']])
    test['confirmed'] = scaler.transform(test[['confirmed']])

    def to_sequences(x, y, seq_size=1):
        x_values = []
        y_values = []

        for i in range(len(x) - seq_size):
            # print(i)
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])

        return np.array(x_values), np.array(y_values)

    trainX, trainY = to_sequences(train, train, seq_size)
    testX, testY = to_sequences(test, test, seq_size)

    model = Sequential()
    model.add(LSTM(neurons, input_shape=(trainX.shape[1], trainX.shape[2]))) #seq_size, n_features
    model.add(Dropout(rate=0.2))

    model.add(RepeatVector(trainX.shape[1])) #seq_size

    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(trainX.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    #history = model.fit(trainX, trainY, epochs=50, steps_per_epoch=10)
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    plot1 = plt.figure(1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=1)

    plot2 = plt.figure(2)
    plt.hist(trainMAE, bins=30)
    plt.title('Train MAE')
    plot3 = plt.figure(3)
    plt.hist(testMAE, bins=30)
    plt.title('Test MAE')
    plt.show(block=True)

    max_trainMAE = float(input("Select your max MAE. Example) 0.3"))
    print("Successfully completed key in your forecast period : " + str(max_trainMAE))

    anomaly_df = pd.DataFrame(test[seq_size:])
    anomaly_df['testMAE'] = testMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    anomaly_df['confirmed'] = test[seq_size:]

    plot4 = plt.figure(4)
    sns.lineplot(x=anomaly_df.index, y=anomaly_df['testMAE'], label = 'testMAE')
    sns.lineplot(x=anomaly_df.index, y=anomaly_df['max_trainMAE'], label = 'max_trainMAE')
    plt.title('train MAE and test MAE')
    plt.legend()

    anomaly_check = anomaly_df['anomaly'].values.sum()
    if anomaly_check == 0:
        print("You do not have anomaly within the range you have selected. You have selected your anomaly range as : ", max_trainMAE)

    else:
        anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
        anomalies = pd.DataFrame(anomalies)

        inverse = pd.DataFrame(scaler.inverse_transform(anomaly_df[['confirmed']]), columns=['confirmed'])
        time = pd.DataFrame(anomaly_df.index, columns=['date'])

        plot5 = plt.figure(5)
        sns.lineplot(x=time['date'], y=inverse['confirmed'])
        sns.scatterplot(x=time['date'], y=scaler.inverse_transform(anomalies['confirmed']), color='r')
        plt.show(block=True)
