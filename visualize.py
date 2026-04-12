import matplotlib.pyplot as plt

def plot_results(y_test, predictions):
    plt.figure()

    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")

    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Index")
    plt.ylabel("Price")

    plt.legend()
    plt.show()