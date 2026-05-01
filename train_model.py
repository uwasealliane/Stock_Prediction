from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(X_train, y_train):
    model = Sequential()
    
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # ✅ SAVE MODEL
    model.save("model.h5")

    print("Model saved as model.h5 ✅")

    return model