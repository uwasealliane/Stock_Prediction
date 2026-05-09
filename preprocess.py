import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):

    # Use 5 features
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    # Create 60-day sequences
    for i in range(60, len(scaled_data)):

        # Last 60 days
        X.append(scaled_data[i-60:i])

        # Predict Close price
        y.append(scaled_data[i, 3])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler