from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from predict import predict_future
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

app = Flask(__name__)

# LOAD DATA ONCE
data = pd.read_csv("data.csv")

X, y, scaler = preprocess_data(data)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = load_model("model.h5")
print("Model loaded ✅")


@app.route("/", methods=["GET", "POST"])
def index():

    predictions = []
    mae = None
    rmse = None
    error_message = None
    days = 0

    if request.method == "POST":

        days = int(request.form["days"])

        # ✅ VALIDATION (IMPORTANT FIX)
        if days < 1:
            error_message = "Days must be at least 1"
        elif days > 30:
            error_message = "Maximum allowed days is 30"
            days = 30

        if not error_message:

            # prediction
            predictions = predict_future(model, X, scaler, days)
            predictions = [float(p) for p in predictions]

            # evaluation
            y_pred = model.predict(X_test)

            y_test_inv = scaler.inverse_transform(y_test)
            y_pred_inv = scaler.inverse_transform(y_pred)

            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    return render_template(
        "index.html",
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        days=days,
        error=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)