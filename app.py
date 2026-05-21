from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from preprocess import preprocess_data

import pandas as pd
import numpy as np

from datetime import timedelta

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# =========================
# CREATE APP
# =========================
app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("data.csv")

data['Date'] = pd.to_datetime(data['Date'])

# =========================
# DATE RANGE
# =========================
MIN_DATE = data['Date'].min().strftime("%Y-%m-%d")
MAX_DATE = data['Date'].max().strftime("%Y-%m-%d")

# =========================
# PREPROCESS
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

print("MODEL LOADED SUCCESSFULLY ✅")

# =========================
# MODEL EVALUATION
# =========================
y_pred = model.predict(X_test, verbose=0)

dummy_test = np.zeros((len(y_test), 5))
dummy_pred = np.zeros((len(y_pred), 5))

dummy_test[:, 3] = y_test
dummy_pred[:, 3] = y_pred.flatten()

y_test_inv = scaler.inverse_transform(dummy_test)[:, 3]
y_pred_inv = scaler.inverse_transform(dummy_pred)[:, 3]

mae_global = mean_absolute_error(
    y_test_inv,
    y_pred_inv
)

rmse_global = np.sqrt(
    mean_squared_error(
        y_test_inv,
        y_pred_inv
    )
)

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    # =========================
    # DEFAULT VALUES
    # =========================
    selected_date = MAX_DATE

    actual_price = 0.00
    predicted_price = 0.00

    next_date = "None"

    trend = "WAITING"

    confidence = 0.0

    mae = round(mae_global, 2)
    rmse = round(rmse_global, 2)

    chart_labels = []
    actual_chart = []
    predicted_chart = []

    error = None

    # =========================
    # WHEN USER PREDICTS
    # =========================
    if request.method == "POST":

        try:

            # =========================
            # GET DATE
            # =========================
            selected_date = request.form.get("date")

            selected_dt = pd.to_datetime(
                selected_date
            )

            # =========================
            # FIND DATE ROW
            # =========================
            matched_rows = data[
                data['Date'] == selected_dt
            ]

            if len(matched_rows) == 0:

                error = "Selected date not found"

            else:

                row_index = matched_rows.index[0]

                # =========================
                # NEED 60 DAYS
                # =========================
                if row_index < 60:

                    error = "Need at least 60 previous days"

                else:

                    # =========================
                    # ACTUAL PRICE
                    # =========================
                    actual_price = float(
                        data.iloc[row_index]['Close']
                    )

                    # =========================
                    # GET LAST 60 DAYS
                    # =========================
                    sequence_data = data.iloc[
                        row_index-60:row_index
                    ][
                        ['Open',
                         'High',
                         'Low',
                         'Close',
                         'Volume']
                    ]

                    # =========================
                    # SCALE
                    # =========================
                    scaled_sequence = scaler.transform(
                        sequence_data
                    )

                    # =========================
                    # RESHAPE
                    # =========================
                    current_input = scaled_sequence.reshape(
                        1,
                        60,
                        5
                    )

                    # =========================
                    # PREDICT
                    # =========================
                    pred = model.predict(
                        current_input,
                        verbose=0
                    )[0][0]

                    # =========================
                    # INVERSE TRANSFORM
                    # =========================
                    dummy = np.zeros((1, 5))

                    dummy[0, 3] = pred

                    predicted_price = scaler.inverse_transform(
                        dummy
                    )[0][3]

                    predicted_price = float(
                        predicted_price
                    )

                    # =========================
                    # NEXT DATE
                    # =========================
                    next_date = (
                        selected_dt + timedelta(days=1)
                    ).strftime("%Y-%m-%d")

                    # =========================
                    # TREND
                    # =========================
                    if predicted_price > actual_price:

                        trend = "BULLISH"

                    else:

                        trend = "BEARISH"

                    # =========================
                    # CONFIDENCE
                    # =========================
                    confidence = round(
                        (1 - (mae / predicted_price)) * 100,
                        1
                    )

                    # =========================
                    # CHART DATA
                    # =========================
                    historical_df = data.iloc[
                        row_index-7:row_index
                    ]

                    chart_labels = historical_df[
                        'Date'
                    ].dt.strftime(
                        "%d %b"
                    ).tolist()

                    actual_chart = historical_df[
                        'Close'
                    ].tolist()

                    # ADD PREDICTION LABEL
                    chart_labels.append(
                        "Prediction"
                    )

                    # CONTINUE ACTUAL LINE
                    actual_chart.append(None)

                    # PREDICTION LINE
                    predicted_chart = [None] * (
                        len(actual_chart) - 2
                    )

                    predicted_chart.append(
                        actual_chart[-2]
                    )

                    predicted_chart.append(
                        predicted_price
                    )

        except Exception as e:

            error = str(e)

    # =========================
    # RETURN TEMPLATE
    # =========================
    return render_template(

        "index.html",

        selected_date=selected_date,

        actual_price=actual_price,

        predicted_price=predicted_price,

        next_date=next_date,

        trend=trend,

        confidence=confidence,

        mae=mae,

        rmse=rmse,

        min_date=MIN_DATE,

        max_date=MAX_DATE,

        chart_labels=chart_labels,

        actual_chart=actual_chart,

        predicted_chart=predicted_chart,

        error=error
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":

    app.run(debug=True)