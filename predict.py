import numpy as np

def predict_future(model, X, scaler, days):

    predictions = []

    current_input = X[-1]
    current_input = current_input.reshape(1, 60, 1)

    for _ in range(days):

        pred = model.predict(current_input, verbose=0)[0][0]

        predictions.append(pred)

        new_input = np.append(
            current_input[:, 1:, :],
            [[[pred]]],
            axis=1
        )

        current_input = new_input

    predictions = np.array(predictions).reshape(-1, 1)

    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten()