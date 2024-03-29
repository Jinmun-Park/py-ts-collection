import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import scipy.stats as st

# Multivariate
stationary = 1
split_size = 0.8
seq_size = 10 # Size to look back
neurons = 64
epochs = 20 #10
batch_size = 16 #32

# Forecast components
n_future = 90 #n_future dates

# data_import_setup
dir_path = 'src/data/GE.csv'
df = pd.read_csv(dir_path)

def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output

def forecast_period_date(n_future):
    date_select = str(input("INPUT : Please write your date column to specify the date in your model. "
                        "Example) Date"))
    global train_date
    train_date = pd.to_datetime(df[date_select])
    global forecast_period_dates
    forecast_period_dates = pd.date_range(list(train_date)[-1], periods=n_future, freq='1d').tolist()


forecast_period_date(n_future=n_future)

def data_analysis(dir_path, histogram):

    # STEP 1 : BRIEF INFORMATION
    print('MESSAGE : This is length of your datasets :', len(df))
    print('MESSAGE : These are names of columns :', df.columns)
    print(df.head())

    # STEP 2 : PLOT TIME SERIES
    df.plot(figsize=(12,5))
    plt.title('Figure 1 : Plot time series using all variables')
    plt.show(block=True)

    # STEP 3 : SELECT COLUMNS AND PLOT
    column_select = input("INPUT : Please select your columns to plot graph. Example) Date, Close").split()
    print("MESSAGE : Successfully completed key in your column fields : " + str(column_select))
    df_plot = df[df.columns.intersection(column_select)]
    df_plot.plot(figsize=(12, 5))
    plt.title('Figure 2 :Plot time series using selected variables')
    plt.show(block=True)

    if histogram == 1:

        # STEP 4 : Freedman–Diaconis
        column_select = input("INPUT : Please select your columns to plot graph. Example) Date, Close").split()
        print("MESSAGE : Successfully completed key in your forecast period : " + str(column_select))

        x = df[df.columns.intersection(column_select)].values

        q25, q75 = np.percentile(x, [.25, .75])
        bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
        bins = round((x.max() - x.min()) / bin_width)
        print("MESSAGE : Freedman–Diaconis number of bins:", bins)
        plt.hist(x, bins=int(bins))
        plt.title("Figure 3 : Freedman–Diaconis Plot")
        plt.show(block=True)

        # STEP 5 : HISTOGRAM AFTER F-D
        plt.hist(x, density=True, bins=int(bins), label="Data")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(clean(x))
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title("Figure 4 : Histogram on your selected columns")
        plt.show(block=True)
    else:
        print('MESSAGE : You set default in plot. Please key in 1 to show your histogram plot')

data_analysis(dir_path=dir_path, histogram=1)

def model_multivariate(stationary, seq_size, neurons, epochs, batch_size):

    # STEP 0 : DATA SELECTION
    print('MESSAGE : These are names of columns :', df.columns)
    print('MESSAGE : df.Head(5) : ')
    print(df.head(5))

    column_select = input("INPUT : Please write column names to build your multivariate model. "
                          "Do not write your date field. Example) Close, Open, Volume").split()
    print("MESSAGE : Successfully completed key in multivariate features : " + str(column_select))
    df_select = df[df.columns.intersection(column_select)].astype(float)

    # STEP 1 : SCALER
    if stationary == 1:
        print("MESSAGE : You selected stationary. Scaler will be selected StandardScaler().")
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    scaler = scaler.fit(df_select)
    df_for_training_scaled = scaler.transform(df_select)

    # STEP 2 : TRAN & TEST SPLIT
    x = []
    y = []

    n_future = 1  # Number of days we want to predict into the future. This is default setting.

    for i in range(seq_size, len(df_for_training_scaled) - n_future + 1):  # 14:251
        x.append(df_for_training_scaled[i - seq_size:i, 0:df_select.shape[1]])
        y.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(x), np.array(y)

    print('MESSAGE : trainX shape == {}.'.format(trainX.shape))
    print('MESSAGE : trainY shape == {}.'.format(trainY.shape))

    # STEP 3 : PARAMETER SETTING
    first_neurons = neurons
    print('MESSAGE : Your first neurons selected in your first layer in LSTM is :', first_neurons)
    second_neurons = int(neurons / 2)
    print('MESSAGE : Your first neurons selected in your first layer in LSTM is :', second_neurons)
    first_dense = int((neurons / 2) / 2)
    print('MESSAGE : Your first dense selected in your first layer in LSTM is :', first_dense)
    second_dense = 1
    print('MESSAGE : Your second and last dense selected in your first layer in LSTM is :', second_dense)

    # STEP 4 : BUILDING MODEL
    model = Sequential()
    model.add(LSTM(first_neurons, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(second_neurons, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # STEP 5 : MODEL FIT
    history = model.fit(trainX, trainY, epochs = epochs, batch_size = batch_size, validation_split = 0.1, verbose = 1)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show(block=True)

    # STEP 6 : MODEL FORECAST

    return trainX, trainY, model

##################################################################### data modelling #####################################################################
trainX, trainY, model = model_multivariate(stationary = stationary, seq_size = seq_size, neurons = neurons, epochs = epochs, batch_size = batch_size)
##########################################################################################################################################################

def model_forecasst(model, n_future, ):

    train_date, forecast_period_dates = forecast_period_date(n_future = n_future)

    forecast = model.predict(trainX[-n_future:])  # forecast

    # Perform inverse transformation to rescale back to original range
    # Since we used 5 variables for transform, the inverse expects same dimensions
    # Therefore, let us copy our values 5 times and discard them after inverse transform
    forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

    # Convert timestamp to date
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    original = df[['Date', 'Open']]
    original['Date'] = pd.to_datetime(original['Date'])
    original = original.loc[original['Date'] >= '2020-5-1']

    sns.lineplot(original['Date'], original['Open'])
    sns.lineplot(df_forecast['Date'], df_forecast['Open'])


'''
# STEP 2 : TRAN & TEST SPLIT
scaler.fit(df_select)

train_size = int(len(df_select) * split_size)
train = df_select.iloc[:train_size]
test = df_select.iloc[train_size:]

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

# STEP 3_1 : DENSE SETTING
def to_sequences_dense(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)

trainX, trainY = to_sequences_dense(train_scaled, seq_size)
testX, testY = to_sequences_dense(test_scaled, seq_size)

# STEP 3_2 : LSTM SETTING
def to_sequences_lstm(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        # print(i)
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])

    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences_lstm(train, train, seq_size)
testX, testY = to_sequences_lstm(test, test, seq_size)

'''