from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(X_train, y_train):

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(64))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.save("model.h5")

    print("Model saved ✅")

    return model