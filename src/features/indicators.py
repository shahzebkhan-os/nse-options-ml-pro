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
    high = df['High']
    low = df['Low']
    volume = df['Volume']
        
    # --- 1. Momentum Indicators ---
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    df['Stoch_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # ROC (Rate of Change) - 10 days
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100

    # --- 2. Volatility Indicators ---

    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BBL'] = sma20 - (2 * std20)
    df['BBM'] = sma20
    df['BBU'] = sma20 + (2 * std20)
    
    # ATR (Average True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # --- 3. Trend/Volume ---
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # Log Returns (Target proxy)
    df['Log_Return'] = np.log(close).diff()
    
    # Clean up NaN values created by rolling windows
    df.dropna(inplace=True)
    return df
