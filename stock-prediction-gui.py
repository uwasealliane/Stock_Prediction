import tkinter as tk
from tkinter import ttk, PhotoImage
import pandas as pd
from train_model import train_model
from load_data import load_data
from preprocess import clean_data

# Load model
data = load_data()
data = clean_data(data)
model, X_test, y_test = train_model(data)

# Prediction function
def predict_price():
    try:
        days = int(entry_days.get())

        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        low_price = float(entry_low.get())
        volume = float(entry_volume.get())

        results = []

        for i in range(days):
            new_data = pd.DataFrame([[open_price, high_price, low_price, volume]],
                                    columns=['Open', 'High', 'Low', 'Volume'])

            prediction = model.predict(new_data)[0]
            results.append(f"Day {i+1}: {round(prediction,4)}")

            # simulate next day
            open_price = prediction
            high_price = prediction + 0.01
            low_price = prediction - 0.01

        result_label.config(text="\n".join(results), fg="#00ffcc")

    except ValueError:
        result_label.config(text="Invalid input!", fg="red")

# Window
window = tk.Tk()
window.title("AI Stock Predictor")
window.geometry("650x550")
window.configure(bg="#0f172a")

frame = tk.Frame(window, bg="#1e293b", padx=20, pady=20)
frame.pack(fill="both", expand=True, padx=20, pady=20)

# ===== TITLE =====
tk.Label(frame, text="AI Stock Price Predictor",
         font=("Arial", 18, "bold"),
         bg="#1e293b", fg="white").pack(pady=10)

# ===== COMPANY SELECTOR =====
tk.Label(frame, text="Select Company",
         bg="#1e293b", fg="#cbd5f5").pack()

company_box = ttk.Combobox(frame, values=["AAPL", "TSLA", "MSFT", "AMZN"])
company_box.set("AAPL")
company_box.pack(pady=5)

# ===== INPUTS =====
def create_entry(label):
    tk.Label(frame, text=label, bg="#1e293b", fg="white").pack()
    entry = tk.Entry(frame, bg="#334155", fg="white")
    entry.pack(pady=5, fill="x")
    return entry

entry_open = create_entry("Open Price")
entry_high = create_entry("High Price")
entry_low = create_entry("Low Price")
entry_volume = create_entry("Volume")

# ===== DAYS INPUT =====
entry_days = create_entry("Number of Days to Predict")

# ===== BUTTON =====
tk.Button(frame, text="Predict",
          font=("Arial", 13, "bold"),
          bg="#3b82f6", fg="white",
          command=predict_price).pack(pady=15)

# ===== RESULT =====
result_label = tk.Label(frame, text="", bg="#1e293b", fg="#00ffcc")
result_label.pack()

window.mainloop()