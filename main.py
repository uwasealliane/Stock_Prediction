from load_data import load_data
from preprocess import clean_data
from train_model import train_model
from predict import make_prediction
#from visualize import plot_results

# Step 1: Load data
data = load_data()

# Step 2: Clean data
data = clean_data(data)

# Step 3: Train model
model, X_test, y_test = train_model(data)

# Step 4: Predict
predictions = make_prediction(model, X_test)

# Step 5: Visualize
#plot_results(y_test, predictions)

from predict import predict_new_data
predict_new_data(model)

from predict import make_prediction,predict_new_data