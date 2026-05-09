from load_data import load_data
from preprocess import preprocess_data
from train_model import train_model

# =========================
# LOAD DATA
# =========================
data = load_data()

# =========================
# PREPROCESS
# =========================
X, y, scaler = preprocess_data(data)

# =========================
# SPLIT DATA
# =========================
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

# =========================
# TRAIN MODEL
# =========================
model = train_model(X_train, y_train)

print("Training completed ✅")