import pandas as pd

def load_data():
    # =========================
# LOAD LIVE STOCK DATA
# =========================
 stock_symbol = "AAPL"

data = yf.download(
    stock_symbol,
    start="1980-01-01",
    end="2020-04-02"
)

 data.reset_index(inplace=True)  
 print("Data loaded successfully")
 return data