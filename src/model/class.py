import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as st

class TimeSeries:
    def __init__(self, dir_path):
        self. dir_path = dir_path
    
    def data_analysis(self, histogram):

        # STEP 1 : BRIEF INFORMATION
        print('I1 : Your directory path is :', self.dir_path)
        df = pd.read_csv(self.dir_path)
        print('I2 : This is length of your datasets :', len(df))
        print('I3 : These are names of columns :', df.columns)
        print(df.head())

        # STEP 2 : PLOT TIME SERIES
        df.plot(figsize=(12, 5))
        plt.title('Figure 1 : Plot time series using all variables')
        plt.show(block=True)

        # STEP 3 : SELECT COLUMNS AND PLOT
        column_select = input("INPUT : Please select your columns to plot graph. Example) Date, Close").split()
        print("MESSAGE : Successfully completed key in your forecast period : " + str(column_select))
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

test = TimeSeries(dir_path = 'src/data/GE.csv')
test.data_analysis(histogram=1)