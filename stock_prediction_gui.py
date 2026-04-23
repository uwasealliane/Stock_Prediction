import tkinter as tk
from tkinter import ttk
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
            results.append(f"Day {i+1}: {round(prediction, 4)}")

            # simulate next day
            open_price = prediction
            high_price = prediction + 0.01
            low_price = prediction - 0.01

        # Clear previous results and display new ones
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        # Create grid of results (2 columns for better display)
        for i, result in enumerate(results):
            row = i // 2
            col = i % 2
            tk.Label(result_frame, text=result, 
                     font=("Arial", 10), 
                     bg="#0f172a", fg="#00ffcc",
                     anchor="w").grid(row=row, column=col, padx=10, pady=5, sticky="w")
        
        result_count_label.config(text=f"Showing {len(results)} predictions")

    except ValueError:
        result_count_label.config(text="Invalid input!", fg="red")
        for widget in result_frame.winfo_children():
            widget.destroy()
        tk.Label(result_frame, text="Please enter valid numbers", 
                 bg="#0f172a", fg="red").pack()

# Window
window = tk.Tk()
window.title("AI Stock Predictor")
window.geometry("900x600")
window.configure(bg="#0f172a")

# Main container frame
main_frame = tk.Frame(window, bg="#0f172a")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# ===== LEFT PANEL (Inputs) =====
left_panel = tk.Frame(main_frame, bg="#1e293b", padx=20, pady=20, relief="ridge", bd=2)
left_panel.pack(side="left", fill="both", expand=False, padx=(0, 10))

# Title
tk.Label(left_panel, text="AI Stock Price Predictor",
         font=("Arial", 16, "bold"),
         bg="#1e293b", fg="white").pack(pady=(0, 15))

# Company Selector
tk.Label(left_panel, text="Select Company", font=("Arial", 11),
         bg="#1e293b", fg="#cbd5f5", anchor="w").pack(fill="x", pady=(0, 5))
company_box = ttk.Combobox(left_panel, values=["AAPL", "TSLA", "MSFT", "AMZN"], font=("Arial", 10))
company_box.set("AAPL")
company_box.pack(fill="x", pady=(0, 15))

# Input fields with labels on left side (grid layout within left panel)
input_frame = tk.Frame(left_panel, bg="#1e293b")
input_frame.pack(fill="x", pady=10)

# Create input fields using grid for proper alignment
labels = ["Open Price:", "High Price:", "Low Price:", "Volume:", "Number of Days to Predict:"]
entries = []

for i, label_text in enumerate(labels):
    # Label on the left
    tk.Label(input_frame, text=label_text, font=("Arial", 10),
             bg="#1e293b", fg="white", anchor="e").grid(row=i, column=0, padx=5, pady=8, sticky="e")
    # Entry on the right
    entry = tk.Entry(input_frame, bg="#334155", fg="white", font=("Arial", 10), width=15)
    entry.grid(row=i, column=1, padx=5, pady=8, sticky="w")
    entries.append(entry)

# Assign entries to variables
entry_open = entries[0]
entry_high = entries[1]
entry_low = entries[2]
entry_volume = entries[3]
entry_days = entries[4]

# Predict Button
predict_btn = tk.Button(left_panel, text="🔮 PREDICT", font=("Arial", 13, "bold"),
                        bg="#3b82f6", fg="white", command=predict_price,
                        cursor="hand2", relief="raised", bd=2)
predict_btn.pack(pady=20, fill="x")

# ===== RIGHT PANEL (Results Grid) =====
right_panel = tk.Frame(main_frame, bg="#0f172a")
right_panel.pack(side="right", fill="both", expand=True)

# Right panel title
tk.Label(right_panel, text="📊 Prediction Results", font=("Arial", 14, "bold"),
         bg="#0f172a", fg="white").pack(anchor="w", pady=(0, 10))

# Frame to hold results count
result_count_label = tk.Label(right_panel, text="Ready for predictions", 
                               font=("Arial", 10), bg="#0f172a", fg="#cbd5f5")
result_count_label.pack(anchor="w", pady=(0, 10))

# Scrollable frame for results
canvas = tk.Canvas(right_panel, bg="#0f172a", highlightthickness=0)
scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#0f172a")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Frame to hold results grid
result_frame = tk.Frame(scrollable_frame, bg="#0f172a")
result_frame.pack(fill="both", expand=True, pady=5)

# Initial instruction
tk.Label(result_frame, text="Enter values and click PREDICT\nto see results here", 
         font=("Arial", 11), bg="#0f172a", fg="#64748b", justify="center").pack(expand=True)

# Configure grid column weights for responsiveness
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)

window.mainloop()