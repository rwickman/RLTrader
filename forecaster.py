import numpy as np
from data_loader import *
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#class Forecaster:

    #def forecast():
        #stock_dataset = load_data(/dta/MSFT.csv) // make it data_csv later like in trainer.py & add self stuff
        #stock_dataset.train_df.head()

# stock_dataset = load_data("data/MSFT.csv") # make it data_csv later like in trainer.py & add self stuff
# print(stock_dataset.train_df.head())

df = pd.read_csv("data/MSFT.csv")
print(df.head())
prophet_df = df[['Date','Adj Close']]
print(prophet_df.head())
m = Prophet()
prophet_df = prophet_df.rename(columns = {'Date':'ds'})
prophet_df = prophet_df.rename(columns = {'Adj Close':'y'})
train_mask = (prophet_df["ds"] < "2019-01-01")
test_mask = (prophet_df["ds"] >= "2019-01-01") & (prophet_df["ds"] < "2020-01-01")
train_set = prophet_df.loc[train_mask]
test_set = prophet_df.loc[test_mask]

m.fit(train_set)

future = m.make_future_dataframe(periods=365) # change to 365 later
future.tail()
forecast = m.predict(future)
print("predicted:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# rms = mean_squared_error(train_set[['y']], forecast[['yhat']], squared=False)
# print(rms)
# fig1 = m.plot(forecast)


