import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):

    # =========================
    # USE 5 FEATURES
    # =========================
    features = data[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ]

    # =========================
    # SCALE DATA
    # =========================
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(features)

    X = []
    y = []

    # =========================
    # CREATE 60 DAY SEQUENCES
    # =========================
    for i in range(60, len(scaled_data)):

        X.append(
            scaled_data[i-60:i]
        )

        # CLOSE PRICE
        y.append(
            scaled_data[i, 3]
        )

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler