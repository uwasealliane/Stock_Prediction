from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocess import preprocess_data
import pandas as pd

# Load data
data = pd.read_csv("data.csv")

# Preprocess
X, y, scaler = preprocess_data(data)

# Build model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=5, batch_size=32)

# Save
model.save("model.h5")

print("Model trained and saved successfully ✅")