import pandas as pd
import numpy as np

def compute_indicators(df):
    """Computes technical indicators using pure pandas."""
    if df is None or df.empty:
        return None
    
    # Ensure standard column names
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
        
    close = df['Close']
        
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BBL'] = sma20 - (2 * std20)
    df['BBM'] = sma20
    df['BBU'] = sma20 + (2 * std20)
        
    # Returns
    df['Log_Return'] = np.log(close).diff()
    
    df.dropna(inplace=True)
    return df
