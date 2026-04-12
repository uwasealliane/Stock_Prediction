import pandas as pd

def load_data():
    data = pd.read_csv("bk_stock_data.csv")
    print("Data loaded successfully")
    return data