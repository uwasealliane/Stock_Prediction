import pandas as pd

def load_data():

    # =========================
    # LOAD CSV DATASET
    # =========================
    data = pd.read_csv("data.csv")

    # Convert Date column
    data['Date'] = pd.to_datetime(data['Date'])

    print("CSV data loaded successfully ✅")

    return data