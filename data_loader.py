import pandas as pd

data_filename = "data/MSFT.csv"

def load_data(data_filename):
    df = pd.read_csv(data_file)

    # Convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Masks for splits
    train_mask = (df["Date"] < "2018-01-01")
    val_mask = (df["Date"] >= "2018-01-01") & (df["Date"] < "2019-01-01")
    test_mask = (df["Date"] >= "2019-01-01") & (df["Date"] < "2020-01-01")

    # Split the data
    train_data = df.loc[train_mask]
    val_data = df.loc[val_mask]
    test_data = df.loc[test_mask]

    return train_data, val_data, test_data