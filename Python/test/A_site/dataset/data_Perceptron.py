import pandas as pd

def load_data(str):
    data = pd.read_csv(str)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y

