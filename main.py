from load_data import load_data
from preprocess import preprocess_data
from train_model import train_model
from predict import make_predictions

# Step 1: Load
data = load_data()

# Step 2: Preprocess
X, y, scaler = preprocess_data(data)

# Step 3: Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 4: Train
model = train_model(X_train, y_train)

# Step 5: Predict
predictions = make_predictions(model, X_test, scaler)

print(predictions[:5])