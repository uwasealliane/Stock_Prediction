from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from predict import predict_future
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load data
data = pd.read_csv("data.csv")

# preprocess
X, y, scaler = preprocess_data(data)

# split (important for evaluation)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# load model
model = load_model("model.h5")
print("Model loaded ✅")


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    mae = None
    rmse = None

    if request.method == "POST":
        days = int(request.form["days"])

        # future prediction
        predictions = predict_future(model, X, scaler, days)

        # 🔥 evaluate model on test data
        y_pred = model.predict(X_test)

        # inverse transform (VERY IMPORTANT)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        # metrics
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    return render_template(
        "index.html",
        predictions=predictions,
        mae=mae,
        rmse=rmse
    )


if __name__ == "__main__":
    app.run(debug=True)