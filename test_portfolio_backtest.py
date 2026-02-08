import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.utils.logger import get_logger

logger = get_logger(__name__)

# diverse basket of 10 stocks
PORTFOLIO = {
    "Large Cap": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"],
    "Mid Cap": ["TRENT", "BEL", "TVSMOTOR"],
    "Small Cap": ["CDSL", "MCX", "IEX"]
}

def backtest_stock(symbol):
    connector = DataConnector()
    df = connector.fetch_ohlcv(symbol, period="5y")
    if df is None or len(df) < 500: return 0.0, 0
    
    # Context
    nifty = connector.fetch_ohlcv("^NSEI", period="5y")
    vix = connector.fetch_ohlcv("^INDIAVIX", period="5y")
    
    if nifty is not None:
        nifty_ret = np.log(nifty['Close']).diff()
        df = df.join(nifty_ret.rename('Nifty_Lag1').shift(1), how='left')
    else: df['Nifty_Lag1'] = 0
        
    if vix is not None:
        df = df.join(vix['Close'].rename('Vix_Level').shift(1), how='left')
    else: df['Vix_Level'] = 15
    
    # Features
    df = compute_indicators(df)
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Lag1'] = df['Log_Return'].shift(1)
    df['Lag2'] = df['Log_Return'].shift(2)
    
    # Target: > 2.0% Move for Mid/Small caps (they are more volatile naturally)
    # Adjusting threshold based on Volatility profile? 
    # Let's keep 1.5% standardized for fair comparison, or slightly higher for smallcaps.
    # We'll use 1.5% as the baseline for "Big Move".
    
    next_return = np.log(df['Close']).diff().shift(-1)
    df['Target'] = np.where(np.abs(next_return) > 0.015, 1, 0)
    
    cols = ['RSI', 'MACD', 'ATR', 'Log_Return', 'Lag1', 'Lag2', 'Nifty_Lag1', 'Vix_Level']
    df = df.dropna()
    
    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = df[cols].iloc[:split], df[cols].iloc[split:]
    y_train, y_test = df['Target'].iloc[:split], df['Target'].iloc[split:]
    
    # Train
    model = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Test
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return acc, len(y_test)

def run_portfolio_test():
    print(f"\n{'='*60}")
    print(f"{'STOCK':<15} | {'CATEGORY':<10} | {'ACCURACY':<10} | {'TEST DAYS':<10}")
    print(f"{'-'*60}")
    
    scores = []
    
    for category, stocks in PORTFOLIO.items():
        for symbol in stocks:
            try:
                acc, days = backtest_stock(symbol)
                print(f"{symbol:<15} | {category:<10} | {acc*100:.2f}%     | {days}")
                scores.append(acc)
            except Exception as e:
                print(f"{symbol:<15} | ERROR      | {e}")
                
    print(f"{'-'*60}")
    print(f"AVERAGE ACCURACY: {np.mean(scores)*100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_portfolio_test()
