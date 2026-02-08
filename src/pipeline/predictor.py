import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.utils.logger import get_logger
import joblib
import os

logger = get_logger(__name__)

class VolatilityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                          class_weight='balanced', random_state=42)
        self.connector = DataConnector()
        self.is_trained = False

    def prepare_data(self, symbol, period="2y"):
        df = self.connector.fetch_ohlcv(symbol, period=period)
        if df is None or df.empty: return None, None
        
        # Market Context
        nifty = self.connector.fetch_ohlcv("^NSEI", period=period)
        vix = self.connector.fetch_ohlcv("^INDIAVIX", period=period)
        
        if nifty is not None:
            nifty_ret = np.log(nifty['Close']).diff()
            df['Nifty_Lag1'] = nifty_ret.shift(1)
        else:
            df['Nifty_Lag1'] = 0
            
        if vix is not None:
            df['Vix_Level'] = vix['Close'].shift(1)
        else:
            df['Vix_Level'] = 15 # Default VIX
            
        # Features
        df = compute_indicators(df)
        df['Log_Return'] = np.log(df['Close']).diff()
        df['Lag1'] = df['Log_Return'].shift(1)
        
        # Target: Volatility > 1.5%
        next_return = np.log(df['Close']).diff().shift(-1)
        df['Target'] = np.where(np.abs(next_return) > 0.015, 1, 0)
        
        feature_cols = ['RSI', 'MACD', 'ATR', 'Log_Return', 'Lag1', 'Nifty_Lag1', 'Vix_Level']
        df = df.dropna()
        
        return df[feature_cols], df['Target']

    def train(self, symbol="RELIANCE"):
        logger.info(f"Training Volatility Model on {symbol}...")
        X, y = self.prepare_data(symbol)
        if X is None or len(X) == 0:
            logger.error("No training data")
            return
            
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Model Trained.")

    def predict(self, symbol):
        if not self.is_trained:
            self.train("RELIANCE") # Train on proxy if not trained
            
        # Fetch latest data
        X, _ = self.prepare_data(symbol, period="3mo")
        if X is None or len(X) == 0: return None
        
        latest = X.iloc[[-1]]
        prob = self.model.predict_proba(latest)[0][1] # Prob of High Vol
        
        regime = "HIGH VOLATILITY" if prob > 0.4 else "QUIET / RANGE-BOUND"
        strategy = "LONG STRADDLE/STRANGLE" if prob > 0.4 else "IRON CONDOR / CREDIT SPREAD"
        
        return {
            "Symbol": symbol,
            "Regime": regime,
            "Confidence": f"{max(prob, 1-prob)*100:.1f}%",
            "Strategy": strategy,
            "Prob_High_Vol": prob
        }
