from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.features.sentiment import SentimentEngine
from src.features.fundamentals import get_fundamentals, get_option_chain_summary
from src.utils.logger import get_logger
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = get_logger(__name__)

class VolatilityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                          class_weight='balanced', random_state=42)
        self.connector = DataConnector()
        self.sentiment_engine = SentimentEngine()
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

    def suggest_strikes(self, spot_price, strategy, options_data):
        """Generates specific Strike Prices for the suggested strategy."""
        if not options_data:
            return {}
            
        # Determine Strike Width based on Index vs Stock
        # NIFTY/BANKNIFTY usually 50/100 intervals
        is_index = spot_price > 20000 
        interval = 100 if is_index else 50
        
        atm_strike = round(spot_price / interval) * interval
        expiry = options_data.get("Expiry", "N/A")
        
        suggestions = {}
        
        if "STRADDLE" in strategy:
            # Long Straddle: Buy ATM Call & Put
            suggestions = {
                "Leg 1 (Buy CE)": f"{atm_strike} CE",
                "Leg 2 (Buy PE)": f"{atm_strike} PE",
                "Ideal Expiry": expiry,
                "Note": "Pure directional volatility play."
            }
            
        elif "IRON CONDOR" in strategy:
            # Iron Condor: Sell OTM, Buy farther OTM for protection
            # Indexes move less in % terms than stocks, so tighten the wings for indexes
            short_pct = 1.015 if is_index else 1.03
            long_pct = 1.025 if is_index else 1.05
            short_pct_down = 0.985 if is_index else 0.97
            long_pct_down = 0.975 if is_index else 0.95
            
            # Short Strikes (Inner Wings)
            short_ce = round((spot_price * short_pct) / interval) * interval
            short_pe = round((spot_price * short_pct_down) / interval) * interval
            
            # Long Strikes (Outer Protection)
            long_ce = round((spot_price * long_pct) / interval) * interval
            long_pe = round((spot_price * long_pct_down) / interval) * interval
            
            suggestions = {
                "Sell Call (Short)": f"{short_ce} CE",
                "Buy Call (Hedge)": f"{long_ce} CE",
                "Sell Put (Short)": f"{short_pe} PE",
                "Buy Put (Hedge)": f"{long_pe} PE",
                "Ideal Expiry": expiry,
                "Note": "Collect premium. Max profit if price stays between Short strikes."
            }
            
        return suggestions

    def predict(self, symbol):
        if not self.is_trained:
            self.train("RELIANCE") # Train on proxy if not trained
            
        # 1. Fetch Price Data & Run ML Prediction
        X, _ = self.prepare_data(symbol, period="3mo")
        if X is None or len(X) == 0: return None
        
        latest = X.iloc[[-1]]
        prob = self.model.predict_proba(latest)[0][1] # Prob of High Vol
        
        regime = "HIGH VOLATILITY" if prob > 0.4 else "QUIET / RANGE-BOUND"
        strategy = "LONG STRADDLE/STRANGLE" if prob > 0.4 else "IRON CONDOR / CREDIT SPREAD"
        
        # 2. Fetch Extra Features
        sent_score, sent_label, headlines = self.sentiment_engine.get_sentiment(symbol)
        fundamentals = get_fundamentals(symbol)
        options_data = get_option_chain_summary(symbol)
        live_price, change, pct_change = self.connector.get_live_price(symbol)
        
        # 3. Generate Specific Strike Suggestions
        trade_setup = self.suggest_strikes(live_price, strategy, options_data)
        
        return {
            "Symbol": symbol,
            "Regime": regime,
            "Confidence": f"{max(prob, 1-prob)*100:.1f}%",
            "Strategy": strategy,
            "Prob_High_Vol": prob,
            "Live_Price": live_price,
            "Price_Change": change,
            "Pct_Change": pct_change,
            "Sentiment": {
                "Score": sent_score,
                "Label": sent_label,
                "Headlines": headlines
            },
            "Fundamentals": fundamentals,
            "Options": options_data,
            "Trade_Setup": trade_setup
        }