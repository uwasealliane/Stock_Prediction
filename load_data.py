import pandas as pd

def load_data():

    # =========================
    # LOAD CSV DATASET
    # =========================
    ACTIVE_DATASET = "AAPL.csv"

    data = pd.read_csv(
    f"datasets/{ACTIVE_DATASET}"
)

    # Convert Date column
    data['Date'] = pd.to_datetime(data['Date'])

    print("CSV data loaded successfully ✅")

    return data