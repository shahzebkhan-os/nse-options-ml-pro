import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_data(symbol):
    connector = DataConnector()
    
    # 1. Fetch Target Stock
    df = connector.fetch_ohlcv(symbol, period="5y")
    if df is None: return None, None
    
    # 2. Fetch Market Context (Nifty 50 & VIX)
    nifty = connector.fetch_ohlcv("^NSEI", period="5y")
    vix = connector.fetch_ohlcv("^INDIAVIX", period="5y")
    
    # Calculate Market Returns
    if nifty is not None:
        nifty['Nifty_Ret'] = np.log(nifty['Close']).diff()
        df = df.join(nifty[['Nifty_Ret']], how='left')
        
    if vix is not None:
        vix['Vix_Close'] = vix['Close']
        df = df.join(vix[['Vix_Close']], how='left')
    
    # 3. Features
    df = compute_indicators(df)
    
    # Lagged Features
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Lag1'] = df['Log_Return'].shift(1)
    
    # Market Context Features
    df['Nifty_Lag1'] = df['Nifty_Ret'].shift(1)
    df['Vix_Level'] = df['Vix_Close'].shift(1)
    
    df = df.dropna()
    
    # 4. Target: Volatility / Big Move Prediction
    # 1 if Absolute Return > 1.5%, else 0
    next_return = np.log(df['Close']).diff().shift(-1)
    y = np.where(np.abs(next_return) > 0.015, 1, 0)
    
    features = ['RSI', 'MACD', 'ATR', 'Log_Return', 'Lag1', 
                'Nifty_Lag1', 'Vix_Level']
                
    X = df[features].iloc[:-1]
    y = y[:-1]
    
    return X, y

def train_rf(symbol="RELIANCE"):
    logger.info(f"Training Random Forest (Big Move Prediction) on {symbol}...")
    X, y = prepare_data(symbol)
    
    if len(X) == 0:
        logger.error("No data available for training")
        return None

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class Balance: {dict(zip(unique, counts))}")
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train with class_weight='balanced' to handle rare big moves
    model = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                  class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    logger.info(f"Test Accuracy: {acc*100:.2f}%")
    logger.info(f"\n{classification_report(y_test, preds)}")
    
    return model

if __name__ == "__main__":
    train_rf("RELIANCE")
