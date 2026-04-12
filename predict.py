import pandas as pd

# OLD function (for test predictions)
def make_prediction(model, X_test):
    predictions = model.predict(X_test)
    print("Prediction done")
    return predictions


# NEW function (for user input prediction)
def predict_new_data(model):
    print("\nEnter new stock data:")

    open_price = float(input("Open price: "))
    high_price = float(input("High price: "))
    low_price = float(input("Low price: "))
    volume = float(input("Volume: "))

    new_data = pd.DataFrame([[open_price, high_price, low_price, volume]],
                            columns=['Open', 'High', 'Low', 'Volume'])

    prediction = model.predict(new_data)

    print("Predicted Close Price:", prediction[0])