from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# CREATE FLASK APP
# =========================
app = Flask(__name__)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
# =========================
# LOAD AAPL DATA
# =========================
data = pd.read_csv("data.csv")

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Dataset range
MIN_DATE = data['Date'].min().strftime("%Y-%m-%d")
MAX_DATE = data['Date'].max().strftime("%Y-%m-%d")

# =========================
# PREPROCESS DATA
# =========================
X, y, scaler = preprocess_data(data)

# =========================
# SPLIT DATA
# =========================
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# LOAD MODEL
# =========================
model = load_model("model.h5")

print("AAPL Model loaded successfully ✅")
print(f"Data range: {MIN_DATE} to {MAX_DATE}")

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    predictions = None
    mae = None
    rmse = None
    error = None
    trend = None
    next_date = None

    selected_date = MAX_DATE

    historical_prices = None
    historical_dates = None
    prediction_dates = None

    # =========================
    # WHEN USER SUBMITS
    # =========================
    if request.method == "POST":

        try:

            # =========================
            # GET DATE FROM CALENDAR
            # =========================
            selected_date = request.form.get("date")

            # Convert selected date
            selected_dt = pd.to_datetime(selected_date)

            # =========================
            # FIND SELECTED ROW
            # =========================
            matched_rows = data[data['Date'] == selected_dt]

            if len(matched_rows) == 0:

                error = "Selected date not found"

            else:

                row_index = matched_rows.index[0]

                # =========================
                # NEED 60 PREVIOUS DAYS
                # =========================
                if row_index < 60:

                    error = "Not enough previous data"

                else:

                    # =========================
                    # GET LAST 60 DAYS
                    # =========================
                    sequence_data = data.iloc[row_index-60:row_index]

                    # Select features
                    sequence_data = sequence_data[
                        ['Open', 'High', 'Low', 'Close', 'Volume']
                    ]

                    # Scale data
                    scaled_sequence = scaler.transform(sequence_data)

                    # Reshape for LSTM
                    current_input = scaled_sequence.reshape(1, 60, 5)

                    # =========================
                    # PREDICT NEXT DAY
                    # =========================
                    pred = model.predict(
                        current_input,
                        verbose=0
                    )[0][0]

                    # =========================
                    # INVERSE TRANSFORM
                    # =========================
                    dummy = np.zeros((1, 5))

                    # Close price column
                    dummy[0, 3] = pred

                    real_prediction = scaler.inverse_transform(dummy)[0][3]

                    predictions = [float(real_prediction)]

                    # =========================
                    # NEXT DATE
                    # =========================
                    next_date = (
                        selected_dt + timedelta(days=1)
                    ).strftime("%Y-%m-%d")

                    # =========================
                    # TREND
                    # =========================
                    # Calculate trend
                    last_real_price = sequence_data['Close'].iloc[-1]

                    if predictions[0] > last_real_price:
                     trend = "UP"
                    else:
                      trend = "DOWN"

                    # =========================
                    # HISTORICAL CHART DATA
                    # =========================
                    historical_df = data.iloc[row_index-30:row_index]

                    historical_prices = historical_df[
                        'Close'
                    ].tolist()

                    historical_dates = historical_df[
                        'Date'
                    ].dt.strftime("%Y-%m-%d").tolist()

                    prediction_dates = [next_date]

                    # =========================
                    # MODEL EVALUATION
                    # =========================
                    y_pred = model.predict(
                        X_test,
                        verbose=0
                    )

                    dummy_test = np.zeros((len(y_test), 5))
                    dummy_pred = np.zeros((len(y_pred), 5))

                    dummy_test[:, 3] = y_test.flatten()
                    dummy_pred[:, 3] = y_pred.flatten()

                    y_test_inv = scaler.inverse_transform(
                        dummy_test
                    )[:, 3]

                    y_pred_inv = scaler.inverse_transform(
                        dummy_pred
                    )[:, 3]

                    mae = mean_absolute_error(
                        y_test_inv,
                        y_pred_inv
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test_inv,
                            y_pred_inv
                        )
                    )

        except Exception as e:

            error = str(e)

    # =========================
    # RETURN PAGE
    # =========================
    return render_template(
        "index.html",
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        error=error,
        trend=trend,
        next_date=next_date,
        selected_date=selected_date,
        min_date=MIN_DATE,
        max_date=MAX_DATE,
        historical_prices=historical_prices,
        historical_dates=historical_dates,
        prediction_dates=prediction_dates
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)