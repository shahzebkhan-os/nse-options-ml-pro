import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_5y_backtest(symbol="RELIANCE"):
    logger.info(f"Starting 5-Year Backtest for {symbol}...")
    
    # 1. Fetch 5 Years of Data
    connector = DataConnector()
    df = connector.fetch_ohlcv(symbol, period="5y")
    if df is None or df.empty:
        logger.error("No data found.")
        return

    # Market Context (5y)
    nifty = connector.fetch_ohlcv("^NSEI", period="5y")
    vix = connector.fetch_ohlcv("^INDIAVIX", period="5y")

    # Align Data
    if nifty is not None:
        nifty_ret = np.log(nifty['Close']).diff()
        df = df.join(nifty_ret.rename('Nifty_Lag1').shift(1), how='left')
    else:
        df['Nifty_Lag1'] = 0

    if vix is not None:
        df = df.join(vix['Close'].rename('Vix_Level').shift(1), how='left')
    else:
        df['Vix_Level'] = 15

    # 2. Features
    df = compute_indicators(df)
    
    # Log Returns & Lags
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Lag1'] = df['Log_Return'].shift(1)
    df['Lag2'] = df['Log_Return'].shift(2)
    df['Lag3'] = df['Log_Return'].shift(3)
    
    # Volatility Target: > 1.5% Move (Absolute)
    next_return = np.log(df['Close']).diff().shift(-1)
    df['Target'] = np.where(np.abs(next_return) > 0.015, 1, 0)
    
    # Clean NaN
    feature_cols = ['RSI', 'MACD', 'ATR', 'Log_Return', 'Lag1', 'Lag2', 'Lag3', 'Nifty_Lag1', 'Vix_Level']
    df = df.dropna()
    
    # 3. Train/Test Split (Time-based)
    # Train on first 4 years (80%), Test on last 1 year (20%)
    split_idx = int(len(df) * 0.8)
    
    X = df[feature_cols]
    y = df['Target']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train Size: {len(X_train)} days | Test Size: {len(X_test)} days")
    
    # 4. Train Model
    model = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                 class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    prob_preds = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*40)
    print(f"📊 BACKTEST RESULTS: {symbol} (5 Years)")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print("-" * 20)
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=['Quiet', 'Volatile']))
    print("-" * 20)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # High Confidence Analysis
    # Check accuracy when model is >70% confident
    high_conf_indices = np.where((prob_preds > 0.7) | (prob_preds < 0.3))[0]
    if len(high_conf_indices) > 0:
        hc_y_true = y_test.iloc[high_conf_indices]
        hc_preds = preds[high_conf_indices]
        hc_acc = accuracy_score(hc_y_true, hc_preds)
        print("-" * 20)
        print(f"High Confidence Accuracy (>70% prob): {hc_acc*100:.2f}% ({len(hc_y_true)} trades)")

if __name__ == "__main__":
    run_5y_backtest("RELIANCE")
    run_5y_backtest("TCS")
