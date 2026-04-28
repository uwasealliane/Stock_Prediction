from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(data):
    data = data[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])   # FIXED HERE

    X = np.array(X)
    y = np.array(y)

    # reshape y for LSTM compatibility
    y = y.reshape(-1, 1)

    return X, y, scaler