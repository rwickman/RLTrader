import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# class DataLoader:
#     def __init__(self, args):
#         self.args = args

@dataclass
class StockDataset:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_mean: float
    train_std: float

def load_data(data_filename, standardize=True):
    df = pd.read_csv(data_filename)

    # Convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Masks for splits
    train_mask = (df["Date"] < "2018-01-01")
    val_mask = (df["Date"] >= "2018-01-01") & (df["Date"] < "2019-01-01")
    test_mask = (df["Date"] >= "2019-01-01") & (df["Date"] < "2020-01-01")

    # Split the data
    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask]
    test_df = df.loc[test_mask]


    if standardize:
        # Standardize adjusted closing prices based only on training date
        train_mean = train_df["Adj Close"].mean()
        train_std = train_df["Adj Close"].std()
        print(train_mean)
        
        train_df.loc["Adj Close"] = (train_df["Adj Close"] - train_mean) / train_std
        val_df.loc["Adj Close"] = (val_df["Adj Close"] - train_mean) / train_std
        test_df.loc["Adj Close"] = (test_df["Adj Close"] - train_mean) / train_std


    return StockDataset(train_df, val_df, test_df, train_mean, train_std)


def compute_macd(df):
    """Compute the MACD and signal line.
    
    Returns:
        MACD and signal line tuple
    """
    ema_12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26

    sig = macd.ewm(span=9, adjust=False).mean()
    return macd, sig
    # plt.plot(macd, label="MACD")
    # plt.plot(sig, label="Signal Line")
    # plt.legend(loc='upper left')
    # plt.show()


def window_data(data, window_size):
    num_windows = len(data) - window_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))
    for i in range(len(data) - window_size + 1):
        windows[i] = data[i:window_size+i]
    return windows

# def create_dataset(data_filename):
#     train_df, val_df, test_df = load_data(data_filename)
#     train_data = np.vstack((macd, sig, train_df["Adj Close"])).T

# data_filename = "data/MSFT.csv"
# train_df, val_df, test_df = load_data(data_filename)
# compute_macd(val_df)
#x = pd.plotting.autocorrelation_plot(test_df["Adj Close"])
#plt.show()