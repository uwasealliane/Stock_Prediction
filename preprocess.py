# preprocess.py
import pandas as pd
import numpy as np

def clean_data(data):
    """
    Preprocess stock market data for prediction
    
    Parameters:
    data: pandas DataFrame with stock data
    
    Returns:
    preprocessed_data: pandas DataFrame ready for training
    """
    
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # 1. Handle missing values
    df = df.dropna()
    
    # 2. If you have 'Date' column, set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # 3. Select only numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # 4. Handle any remaining NaN values
    df = df.ffill().bfill()
    
    # 5. Create additional features (optional)
    if 'Close' in df.columns:
        # Add returns
        df['Returns'] = df['Close'].pct_change()
        
        # Add moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        
        # Drop NaN values from rolling calculations
        df = df.dropna()
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    
    return df

# Alternative simple version if you just need to get it working
def simple_preprocess(data):
    """Simple preprocessing for testing"""
    return data.dropna()