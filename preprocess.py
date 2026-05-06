import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):

    data = data[['Close']]  # ONLY CLOSE PRICE

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler