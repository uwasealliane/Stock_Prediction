from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file
)

from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from tensorflow.keras.models import load_model

from preprocess import preprocess_data

import pandas as pd
import numpy as np

from datetime import timedelta

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet


# =========================
# CREATE APP
# =========================
app = Flask(__name__)

app.secret_key = "stock_secret_key"

# =========================
# DATABASE CONFIG
# =========================
app.config['SQLALCHEMY_DATABASE_URI'] = \
'mysql+pymysql://root:@localhost/stock_prediction_db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================
# USER MODEL
# =========================
class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("data.csv")

data['Date'] = pd.to_datetime(data['Date'])

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

print("MODEL LOADED SUCCESSFULLY ✅")

# =========================
# GLOBAL VARIABLES
# =========================
mae_global = 0
rmse_global = 0

# =========================
# SIGNUP
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get("username")

        email = request.form.get("email")

        password = request.form.get("password")

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash("Email already exists")

            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        new_user = User(

            username=username,

            email=email,

            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        flash("Account created successfully")

        return redirect(url_for("login"))

    return render_template("signup.html")

# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")

        password = request.form.get("password")

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session['username'] = user.username

            flash("Login successful")

            return redirect(url_for("index"))

        else:

            flash("Invalid email or password")

    return render_template("login.html")

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():

    session.pop("username", None)

    flash("Logged out successfully")

    return redirect(url_for("login"))

# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    global mae_global
    global rmse_global

    if 'username' not in session:

        return redirect(url_for("login"))

    selected_date = MAX_DATE

    actual_price = 0.0
    predicted_price = 0.0

    next_date = "None"

    trend = "WAITING"

    confidence = 0.0

    mae = 0.0
    rmse = 0.0

    chart_labels = []
    actual_chart = []
    predicted_chart = []

    error = None

    # =========================
    # MODEL EVALUATION
    # =========================
    try:

        y_pred = model.predict(X_test, verbose=0)

        # CREATE DUMMY ARRAYS
        y_test_dummy = np.zeros((len(y_test), 5))
        y_pred_dummy = np.zeros((len(y_pred), 5))

        # PUT CLOSE PRICE
        y_test_dummy[:, 3] = y_test
        y_pred_dummy[:, 3] = y_pred.flatten()

        # INVERSE TRANSFORM
        y_test_inv = scaler.inverse_transform(
            y_test_dummy
        )[:, 3]

        y_pred_inv = scaler.inverse_transform(
            y_pred_dummy
        )[:, 3]

        # METRICS
        mae = float(
            mean_absolute_error(
                y_test_inv,
                y_pred_inv
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test_inv,
                    y_pred_inv
                )
            )
        )

        mae_global = mae
        rmse_global = rmse

    except Exception as e:

        error = str(e)

    # =========================
    # PREDICT
    # =========================
    if request.method == "POST":

        try:

            selected_date = request.form.get("date")

            selected_dt = pd.to_datetime(
                selected_date
            )

            matched_rows = data[
                data['Date'] == selected_dt
            ]

            if len(matched_rows) == 0:

                error = "Selected date not found"

            else:

                row_index = matched_rows.index[0]

                if row_index < 60:

                    error = "Not enough previous data"

                else:

                    # =========================
                    # ACTUAL PRICE
                    # =========================
                    actual_price = float(
                        data.iloc[row_index]['Close']
                    )

                    # =========================
                    # LAST 60 DAYS
                    # =========================
                    sequence_data = data.iloc[
                        row_index-60:row_index
                    ][[
                        'Open',
                        'High',
                        'Low',
                        'Close',
                        'Volume'
                    ]]

                    scaled_sequence = scaler.transform(
                        sequence_data
                    )

                    current_input = scaled_sequence.reshape(
                        1,
                        60,
                        5
                    )

                    pred = model.predict(
                        current_input,
                        verbose=0
                    )[0][0]

                    # =========================
                    # INVERSE TRANSFORM
                    # =========================
                    dummy_array = np.zeros((1, 5))

                    dummy_array[0, 3] = pred

                    predicted_price = scaler.inverse_transform(
                        dummy_array
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

                        trend = "BULLISH 📈"

                    else:

                        trend = "BEARISH 📉"

                    # =========================
                    # CONFIDENCE
                    # =========================
                    confidence = max(
                        0,
                        round(
                            100 - (
                                mae / predicted_price
                            ) * 100,
                            1
                        )
                    )

                    # =========================
                    # CHART
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

                    actual_chart.append(None)

                    predicted_chart = [None] * (
                        len(actual_chart) - 2
                    )

                    predicted_chart.append(
                        actual_chart[-2]
                    )

                    predicted_chart.append(
                        predicted_price
                    )

                    chart_labels.append(
                        "Prediction"
                    )

        except Exception as e:

            error = str(e)

    return render_template(

        "index.html",

        username=session.get("username"),

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
# DOWNLOAD REPORT
# =========================
@app.route("/download-report")
def download_report():

    pdf_file = "prediction_report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "StockPrice AI Prediction Report",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    report_lines = [

        f"MAE: ${round(mae_global,2)}",

        f"RMSE: ${round(rmse_global,2)}",

        "Model: LSTM Deep Learning",

        "Stock: AAPL"
    ]

    for line in report_lines:

        paragraph = Paragraph(
            line,
            styles['BodyText']
        )

        elements.append(paragraph)

        elements.append(Spacer(1, 12))

    doc.build(elements)

    return send_file(
        pdf_file,
        as_attachment=True
    )

# =========================
# CREATE TABLES
# =========================
with app.app_context():

    db.create_all()

# =========================
# RUN APP
# =========================
if __name__ == "__main__":

    app.run(debug=True)