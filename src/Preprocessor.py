import pandas as pd


def load_data():
    df = pd.read_csv("src/loan_data.csv")
    return df
