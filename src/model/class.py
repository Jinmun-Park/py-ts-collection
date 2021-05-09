import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
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

class TimeSeries:
    def __init__(self, dir_path, n_future):
        self. dir_path = dir_path
        self.n_future = n_future

    def data_preprocess(self):

        # STEP 1 : DATAFRAME SETUP
        print('MESSAGE : Your directory path is :', self.dir_path)
        df = pd.read_csv(self.dir_path)
        print('MESSAGE : This is length of your datasets :', len(df))
        print('MESSAGE : These are names of columns :', df.columns)
        print('MESSAGE : df.Head(5) : ')
        print(df.head())

        # STEP 2 : SELECT COLUMNS FOR MULTIVARIATE MODEL
        print("MESSAGE : THIS IS YOUR INTIAL SETUP TO BUILD MULTIVARIATE MODEL")
        column_select = input("INPUT : Please write column names to build your multivariate model. "
                              "Do not write your date field. Example) Close, Open, Volume").split()
        print("MESSAGE : Successfully completed key in multivariate features : " + str(column_select))
        df_select = df[df.columns.intersection(column_select)].astype(float)

        # STEP 3 : SELECT 'DATE' COLUMN TO BUILD MULTIVARIATE MODEL
        date_select = str(input("INPUT : Please write your date column to specify the date in your model. "
                                "Example) Date"))
        train_date = pd.to_datetime(df[date_select])
        forecast_period_dates = pd.date_range(list(train_date)[-1], periods=self.n_future, freq='1d').tolist()

        return df, df_select, train_date, forecast_period_dates

    def data_analysis(self, histogram):

        # STEP 1 : DF TIME SERIES
        df.plot(figsize=(12, 5))
        plt.title('Figure 1 : Plot time series using all variables')
        plt.show(block=True)
        print('MESSAGE : Successfully completed plotting Figure 1')

        # STEP 3 : SELECT COLUMNS AND PLOT
        column_plot = input("INPUT : Please select your columns to plot graph. This includes DATE field. Example) Date, Close").split()
        print("MESSAGE : Successfully completed key in your forecast period : " + str(column_plot))
        df_plot = df[df.columns.intersection(column_plot)]
        df_plot.plot(figsize=(12, 5))
        plt.title('Figure 2 :Plot time series using selected variables')
        plt.show(block=True)

        if histogram == 1:

            # STEP 4 : Freedman–Diaconis
            x = df[df.columns.intersection(column_plot)].values

            q25, q75 = np.percentile(x, [.25, .75])
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = round((x.max() - x.min()) / bin_width)
            print("MESSAGE : Freedman–Diaconis number of bins:", bins)
            plt.hist(x, bins=int(bins))
            plt.title("Figure 3 : Freedman–Diaconis Plot")
            plt.show(block=True)

            # STEP 5 : HISTOGRAM AFTER F-D
            def clean(serie):
                output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
                return output

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
            print('You set default in plot. Please key in 1 to show your histogram plot')

class Preprocessing(TimeSeries):
    def __init__(self):
        TimeSeries.__init__(self, dir,)

raw = TimeSeries(dir_path = 'src/data/GE.csv')
raw.data_analysis(histogram=1)