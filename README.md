# Python-Time series modelling collection

## Project Overview
<p align="justify">This repository contains my studies in both trational time series modelling and machine learning to creat class function. My final objective is to publish a package that can transform the data to build univariate or multivariate model. My aim is to test out many modelling techniques and melt them down into my package. Kaggle contests are good education materils and i am using several datasets from Kaggle to test out accuracy. </p>

***

## Data Sources
1. Airpassenger
2. [Yahoo Finance - GE][GE]
3. [Covid19][Covid]
4. [CreditFraud][Kaggle-Credit]

***

## Data Model 

### 1. Recursive Nerual Network

### 2. LSTM 
- Univariate
```
raw = TsModelling(dir_path = 'src/data/GE.csv', n_future = 90)
raw.data_preprocess_uni()
#raw.data_analysis(histogram=1)
trainX, trainY, model = raw.univariate(stationary = 1, seq_size = 14, neurons = 128, epochs = 50, batch_size = 32)
raw.univariate_forecast()
```
- Multivariate
```
raw = TsModelling(dir_path = 'src/data/GE.csv', n_future = 60)
raw.data_preprocess_multi()
#raw.data_analysis(histogram=1)
trainX, trainY, model = raw.multivariate(stationary = 1, seq_size = 14, neurons = 64, epochs = 100, batch_size = 14)
raw.multivariate_forecast()
```
### 3. AutoTs

### 4. Imbalance Modelling (SMOTE)


[GE]:https://finance.yahoo.com/quote/GE?p=GE
[Covid]:https://github.com/CSSEGISandData/COVID-19
[Kaggle-Credit]:https://www.kaggle.com/mlg-ulb/creditcardfraud

