import numpy as np

def predict_future(model, X, scaler, days):
    predictions = []

    # last 60 days input
    current_input = X[-1:]  # shape: (1, 60, 1)

    for _ in range(days):

        pred = model.predict(current_input, verbose=0)
        pred_value = pred[0][0]

        predictions.append(pred_value)

        # IMPORTANT: keep shape (1,1,1)
        pred_reshaped = np.array(pred_value).reshape(1, 1, 1)

        # slide window
        current_input = np.concatenate(
            (current_input[:, 1:, :], pred_reshaped),
            axis=1
        )

    # convert back to real prices
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions