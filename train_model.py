from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocess import preprocess_data
import pandas as pd
import os

# =========================
# DATASETS TO TRAIN
# =========================

datasets = [
    "AAPL.csv",
    "TSLA.csv",
    "MSFT.csv",
    "NVDA.csv"
]

# Create models folder if missing
os.makedirs("models", exist_ok=True)

# =========================
# TRAIN EACH DATASET
# =========================

for dataset in datasets:

    print(f"\nTraining {dataset} ...")

    data = pd.read_csv(
        f"datasets/{dataset}"
    )

    X, y, scaler = preprocess_data(data)

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(X.shape[1], X.shape[2])
        )
    )

    model.add(
        LSTM(50)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=1
    )

    model_name = dataset.replace(
        ".csv",
        ".keras"
    )

    model.save(
        f"models/{model_name}"
    )

    print(
        f"{model_name} saved successfully ✅"
    )

print("\nALL MODELS TRAINED SUCCESSFULLY ✅")