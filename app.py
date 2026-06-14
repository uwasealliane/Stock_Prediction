
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from flask import jsonify, request, session
from werkzeug.security import generate_password_hash
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

mae_global = 0
rmse_global = 0
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
from datetime import datetime
from datetime import datetime

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )

    is_admin = db.Column(
        db.Boolean,
        default=False
    )

    status = db.Column(
        db.String(20),
        default="Active"
    )

    last_login = db.Column(
        db.DateTime,
        nullable=True
    )

# =========================
# PREDICTION MODEL
# =========================
class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id')
    )

    stock = db.Column(
        db.String(50)
    )

    actual_price = db.Column(
        db.Float
    )

    predicted_price = db.Column(
        db.Float
    )

    trend = db.Column(
        db.String(50)
    )

    created_at = db.Column(
        db.DateTime,
        default=db.func.now()
    )

    selected_date = db.Column(
    db.String(20)
)
    
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(
        db.String(200),
        nullable=False,
        unique=True
    )

    is_active = db.Column(
        db.Boolean,
        default=False
    )

    uploaded_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# =========================
# LOAD DATASET
# =========================
# =========================
# LOAD DATASET
# =========================

ACTIVE_DATASET = "AAPL.csv"

data = pd.read_csv(
    f"datasets/{ACTIVE_DATASET}"
)


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

            # Update last login
            user.last_login = db.func.now()
            db.session.commit()

            # Clear flash messages
            session.pop('_flashes', None)

            # Save session
            session['username'] = user.username
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin

            # Admin
            if user.is_admin:
                flash(
                    f"Welcome Admin {user.username}!",
                    "success"
                )
                return redirect(
                    url_for("admin_dashboard")
                )

            # Normal user
            flash(
                f"Welcome back {user.username}!",
                "success"
            )

            return redirect(
                url_for("index")
            )

        else:

            flash(
                "Invalid email or password",
                "danger"
            )

    if 'username' in session:

        if session.get("is_admin"):
            return redirect(
                url_for("admin_dashboard")
            )

        return redirect(
            url_for("index")
        )

    return render_template("login.html")
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    global mae_global, rmse_global

    if 'username' not in session:
        return redirect(url_for("login"))

    error = None
    confidence = 0.0  # Initialize confidence

    # USE LATEST DATE AUTOMATICALLY
    latest_row = data.iloc[-1]
    selected_dt = latest_row['Date']
    selected_date = selected_dt.strftime("%Y-%m-%d")
    actual_price = float(latest_row['Close'])
    predicted_price = 0.0
    next_date = (selected_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    trend = "WAITING"
    mae = 0.0
    rmse = 0.0

    current_mae = 0.0
    current_rmse = 0.0
    chart_labels = []
    actual_chart = []
    predicted_chart = []

    try:
        # MODEL EVALUATION (keep as is)
        y_pred = model.predict(X_test, verbose=0)
        y_test_dummy = np.zeros((len(y_test), 5))
        y_pred_dummy = np.zeros((len(y_pred), 5))
        y_test_dummy[:, 3] = y_test
        y_pred_dummy[:, 3] = y_pred.flatten()
        y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 3]
        y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, 3]
        mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
        rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
        global mae_global, rmse_global
        mae_global = mae
        rmse_global = rmse

        # DEFAULT PREDICTION
        row_index = len(data) - 1
        sequence_data = data.iloc[row_index-60:row_index][['Open', 'High', 'Low', 'Close', 'Volume']]
        scaled_sequence = scaler.transform(sequence_data)
        current_input = scaled_sequence.reshape(1, 60, 5)
        pred = model.predict(current_input, verbose=0)[0][0]
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 3] = pred
        predicted_price = float(scaler.inverse_transform(dummy_array)[0][3])

        

        # TREND
        if predicted_price > actual_price:
            trend = "BULLISH 📈"
        else:
            trend = "BEARISH 📉"

        # =========================
        # CONFIDENCE CALCULATION (NEW)
        # =========================
        try:
            confidence = round(100 - abs(predicted_price - actual_price) / actual_price * 100, 1)
            confidence = max(0, min(100, confidence))
        except:
            confidence = 0.0

        # =========================
        # SAVE PREDICTION (NEW with selected_date)
        # =========================
        user = User.query.filter_by(username=session['username']).first()
        if user:
            existing = Prediction.query.filter_by(user_id=user.id, selected_date=selected_date).first()
            if not existing:
                new_prediction = Prediction(
                    user_id=user.id,
                    stock="AAPL",
                    actual_price=actual_price,
                    predicted_price=predicted_price,
                    trend=trend,
                    selected_date=selected_date
                )
                db.session.add(new_prediction)
                db.session.commit()

        # CHART DATA
        historical_df = data.iloc[row_index-7:row_index]
        chart_labels = historical_df['Date'].dt.strftime("%d %b").tolist()
        actual_chart = historical_df['Close'].tolist()
        actual_chart.append(None)
        predicted_chart = [None] * (len(actual_chart) - 2)
        predicted_chart.append(actual_chart[-2])
        predicted_chart.append(predicted_price)
        chart_labels.append("Prediction")

    except Exception as e:
        error = str(e)

    # =========================
    # USER PREDICTION (POST)
    # =========================
    if request.method == "POST":
        try:
            selected_date = request.form.get("date")
            selected_dt = pd.to_datetime(selected_date)
            matched_rows = data[data['Date'] == selected_dt]

            if len(matched_rows) == 0:
                error = "Selected date not found"
            else:
                row_index = matched_rows.index[0]
                if row_index < 60:
                    error = "Not enough previous data"
                else:
                    actual_price = float(data.iloc[row_index]['Close'])
                    sequence_data = data.iloc[row_index-60:row_index][['Open', 'High', 'Low', 'Close', 'Volume']]
                    scaled_sequence = scaler.transform(sequence_data)
                    current_input = scaled_sequence.reshape(1, 60, 5)
                    pred = model.predict(current_input, verbose=0)[0][0]
                    dummy_array = np.zeros((1, 5))
                    dummy_array[0, 3] = pred
                    predicted_price = float(scaler.inverse_transform(dummy_array)[0][3])
                    next_date = (selected_dt + timedelta(days=1)).strftime("%Y-%m-%d")

                    current_mae = round(
                    abs(actual_price - predicted_price),
                    2
                    )

                    current_rmse = round(
                    np.sqrt(
                    (actual_price - predicted_price) ** 2
                    ),
                    2
                    )

                    if predicted_price > actual_price:
                        trend = "BULLISH 📈"
                    else:
                        trend = "BEARISH 📉"

                    # =========================
                    # CONFIDENCE CALCULATION (NEW)
                    # =========================
                    try:
                        confidence = round(100 - abs(predicted_price - actual_price) / actual_price * 100, 1)
                        confidence = max(0, min(100, confidence))
                    except:
                        confidence = 0.0

                    # =========================
                    # SAVE PREDICTION (NEW with selected_date)
                    # =========================
                    user = User.query.filter_by(username=session['username']).first()
                    if user:
                        existing = Prediction.query.filter_by(user_id=user.id, selected_date=selected_date).first()
                        if not existing:
                            new_prediction = Prediction(
                                user_id=user.id,
                                stock="AAPL",
                                actual_price=actual_price,
                                predicted_price=predicted_price,
                                trend=trend,
                                selected_date=selected_date
                            )
                            db.session.add(new_prediction)
                            db.session.commit()

                    # CHART DATA
                    historical_df = data.iloc[row_index-7:row_index]
                    chart_labels = historical_df['Date'].dt.strftime("%d %b").tolist()
                    actual_chart = historical_df['Close'].tolist()
                    actual_chart.append(None)
                    predicted_chart = [None] * (len(actual_chart) - 2)
                    predicted_chart.append(actual_chart[-2])
                    predicted_chart.append(predicted_price)
                    chart_labels.append("Prediction")

        except Exception as e:
            error = str(e)

            print("MAE =", mae)
            print("RMSE =", rmse)

    return render_template(
        "index.html",
        username=session.get("username"),
        selected_date=selected_date,
        actual_price=actual_price,
        predicted_price=predicted_price,
        next_date=next_date,
        trend=trend,
        confidence=confidence,
        mae=current_mae if request.method == "POST" else mae,
        rmse=current_rmse if request.method == "POST" else rmse,
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

    datasets = [
        "AAPL.csv",
        "TSLA.csv",
        "MSFT.csv",
        "NVDA.csv"
    ]

    for file in datasets:

        existing = Dataset.query.filter_by(
            filename=file
        ).first()

        if not existing:

            new_dataset = Dataset(
                filename=file,
                is_active=(file == "AAPL.csv")
            )

            db.session.add(new_dataset)

    db.session.commit()

    active_dataset = Dataset.query.filter_by(
        is_active=True
    ).first()

    if active_dataset:
        ACTIVE_DATASET = active_dataset.filename
    else:
        ACTIVE_DATASET = "AAPL.csv"
    # =========================
# CREATE DEFAULT ADMIN
# =========================
with app.app_context():

    db.create_all()

    admin = User.query.filter_by(
        email="alliane@gmail.com"
    ).first()

    if not admin:

        admin_user = User(

            username="Admin",

            email="alliane@gmail.com",

            password=generate_password_hash(
                "admin123"
            ),

            is_admin=True,

        )

        db.session.add(admin_user)

        db.session.commit()

        print("ADMIN CREATED SUCCESSFULLY ✅")

        @app.route('/admin')
        def admin_dashboard():
         return render_template('admin_dashboard.html')


@app.route('/admin/users')
def admin_users():
    users = User.query.all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/predictions')
def admin_predictions():
    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        'admin_predictions.html',
        predictions=predictions
    )


@app.route('/admin/dataset')
def admin_dataset():
    return render_template('admin_dataset.html')


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('login'))

        # =========================
# ADMIN DASHBOARD
# =========================

@app.route("/admin")
def admin_dashboard():
    global mae_global, rmse_global
    if 'username' not in session:
        return redirect(url_for("login"))
    

    if not session.get("is_admin"):
        flash("Access denied")
        return redirect(url_for("index"))

    total_users = User.query.count()

    total_predictions = Prediction.query.count()

    predictions = Prediction.query.order_by(
    Prediction.id.desc()
    ).limit(10).all()

    users = User.query.all()

    user_data = []

    for user in users:

        prediction_count = Prediction.query.filter_by(
            user_id=user.id
        ).count()

        user_data.append({

            "id": user.id,

            "username": user.username,

            "email": user.email,

            "is_admin": user.is_admin,

            "predictions": prediction_count,

            "last_login":
                user.last_login.strftime("%Y-%m-%d %H:%M:%S")
                if user.last_login else "Never"

        })
    import os

    data = pd.read_csv(
    f"datasets/{ACTIVE_DATASET}"
)

    dataset_rows = len(data)

    dataset_columns = len(data.columns)

    date_range = (
    str(data['Date'].min())[:10]
    + " - " +
    str(data['Date'].max())[:10]
    )

    last_updated = datetime.fromtimestamp(
    os.path.getmtime(
        f"datasets/{ACTIVE_DATASET}"
    )
).strftime("%B %d, %Y")

    print("MAE:", mae_global)
    print("RMSE:", rmse_global)

    return render_template(
    "admin.html",

    username=session.get("username"),

    total_users=total_users,
    total_predictions=total_predictions,

    users=user_data,
    predictions=predictions,

    mae=round(mae_global, 2),
    rmse=round(rmse_global, 2),

    dataset_rows=dataset_rows,
    dataset_columns=dataset_columns,
    date_range=date_range,
    last_updated=last_updated,
    datasets=datasets
)





@app.route("/search-user")
def search_user():

    keyword = request.args.get("keyword", "")

    users = User.query.filter(
        User.username.like(f"%{keyword}%")
    ).all()

    predictions = Prediction.query.all()

    total_users = User.query.count()

    total_predictions = Prediction.query.count()

    return render_template(
        "admin.html",
        username=session.get("username"),
        users=users,
        predictions=predictions,
        total_users=total_users,
        total_predictions=total_predictions
    )

@app.route('/add-user', methods=['POST'])
def add_user():

    if not session.get('is_admin'):
        return jsonify({
            "success": False,
            "error": "Unauthorized"
        }), 403

    data = request.get_json(force=True)

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    is_admin = data.get("is_admin", False)

    if not username or not email or not password:
        return jsonify({
            "success": False,
            "error": "All fields are required"
        }), 400

    existing_user = User.query.filter(
        (User.email == email) |
        (User.username == username)
    ).first()

    if existing_user:
        return jsonify({
            "success": False,
            "error": "User already exists"
        }), 409

    new_user = User(
        username=username,
        email=email,
        password=generate_password_hash(password),
        is_admin=is_admin,
        status="Active"
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "User added successfully"
    }), 200


@app.route("/retrain-model", methods=["POST"])
def retrain_model():

    global model

    try:

        X, y, scaler = preprocess_data(data)

        split = int(len(X) * 0.8)

        X_train = X[:split]
        y_train = y[:split]

        model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            verbose=1
        )

        model.save("model.h5")

        flash(
            "Model retrained successfully!",
            "success"
        )

    except Exception as e:

        flash(
            f"Retraining failed: {e}",
            "danger"
        )

    return redirect(url_for("admin_dashboard"))

# =========================
# DELETE SINGLE PREDICTION RECORD
# =========================
@app.route("/delete-prediction/<int:prediction_id>", methods=["DELETE"])
def delete_prediction(prediction_id):

    if not session.get("is_admin"):
        return jsonify({
            "success": False,
            "error": "Unauthorized"
        })

    prediction = Prediction.query.get(prediction_id)

    if not prediction:
        return jsonify({
            "success": False,
            "error": "Prediction not found"
        })

    db.session.delete(prediction)
    db.session.commit()

    return jsonify({
        "success": True
    })

# =========================
# CLEAR ALL PREDICTION HISTORY
# =========================
@app.route("/clear-prediction-history", methods=["DELETE"])
def clear_prediction_history():

    if not session.get("is_admin"):
        return jsonify({
            "success": False,
            "error": "Unauthorized"
        })

    Prediction.query.delete()

    db.session.commit()

    return jsonify({
        "success": True
    })


# =========================
# GET PREDICTION STATS (AJAX)
# =========================
@app.route("/api/prediction-stats")
def prediction_stats():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 401
    
    total = Prediction.query.count()
    bullish = Prediction.query.filter(Prediction.trend.like("%BULLISH%")).count()
    bearish = Prediction.query.filter(Prediction.trend.like("%BEARISH%")).count()
    
    return jsonify({
        "total": total,
        "bullish": bullish,
        "bearish": bearish
    })

@app.route("/toggle-user/<int:user_id>")
def toggle_user(user_id):

    user = User.query.get(user_id)

    if not user:
        return jsonify({
            "success": False,
            "error": "User not found"
        })

    if user.is_admin:
        return jsonify({
            "success": False,
            "error": "Cannot modify admin"
        })

    user.status = (
        "Suspended"
        if user.status == "Active"
        else "Active"
    )

    db.session.commit()

    return jsonify({
        "success": True
    })

@app.route("/delete-user/<int:user_id>")
def delete_user(user_id):

    user = User.query.get(user_id)

    if not user:
        return jsonify({
            "success": False,
            "error": "User not found"
        })

    if user.is_admin:
        return jsonify({
            "success": False,
            "error": "Admin cannot be deleted"
        })

    Prediction.query.filter_by(
        user_id=user.id
    ).delete()

    db.session.delete(user)

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "User deleted successfully"
    })

# =========================
# RUN APP
# =========================
if __name__ == "__main__":

    app.run(debug=True)