from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.features.sentiment import SentimentEngine
from src.features.fundamentals import get_fundamentals, get_option_chain_summary
from src.utils.logger import get_logger
from src.options.bs_pricing import black_scholes
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

class VolatilityPredictor:
    def __init__(self):
        self.volatility_model = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                                      class_weight='balanced', random_state=42)
        self.directional_model = LogisticRegression(random_state=42, max_iter=1000)
        self.connector = DataConnector()
        self.sentiment_engine = SentimentEngine()
        self.is_trained = False
        self.sector_models = {}
        self.directional_sector_models = {}

    def prepare_directional_data(self, symbol, period="2y"):
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
        df['Lag2'] = df['Log_Return'].shift(2)
        df['Lag3'] = df['Log_Return'].shift(3)
        df['Lag5'] = df['Log_Return'].shift(5)
        
        # Multi-horizon targets
        df['Target_1d'] = np.where(np.log(df['Close']).diff().shift(-1) > 0, 1, 0)  # 1 day
        df['Target_3d'] = np.where(np.log(df['Close']).diff(periods=3).shift(-3) > 0, 1, 0)  # 3 day
        df['Target_5d'] = np.where(np.log(df['Close']).diff(periods=5).shift(-5) > 0, 1, 0)  # 5 day
        df['Target_15d'] = np.where(np.log(df['Close']).diff(periods=15).shift(-15) > 0, 1, 0)  # 15 day
        df['Target_1m'] = np.where(np.log(df['Close']).diff(periods=22).shift(-22) > 0, 1, 0)  # 1 month (~22 trading days)
        df['Target_3m'] = np.where(np.log(df['Close']).diff(periods=66).shift(-66) > 0, 1, 0)  # 3 months
        df['Target_6m'] = np.where(np.log(df['Close']).diff(periods=132).shift(-132) > 0, 1, 0)  # 6 months
        df['Target_1y'] = np.where(np.log(df['Close']).diff(periods=264).shift(-264) > 0, 1, 0)  # 1 year
        
        feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'ATR_Percent', 
                       'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D', 'Log_Return', 
                       'Lag1', 'Lag2', 'Lag3', 'Lag5', 'Nifty_Lag1', 'Vix_Level',
                       'Volume_MA_Ratio', 'Price_Volume_Trend', 'OBV']
        
        df = df.dropna()
        
        return df[feature_cols], {
            'Target_1d': df['Target_1d'],
            'Target_3d': df['Target_3d'],
            'Target_5d': df['Target_5d'],
            'Target_15d': df['Target_15d'],
            'Target_1m': df['Target_1m'],
            'Target_3m': df['Target_3m'],
            'Target_6m': df['Target_6m'],
            'Target_1y': df['Target_1y']
        }

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

    def train_directional_sector_models(self):
        """Trains directional prediction models for different sectors."""
        sectors = {
            "BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "^NSEBANK"],
            "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
            "ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL"],
            "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
            "GENERIC": ["^NSEI", "^BSESN", "ITC", "LT", "TITAN"]
        }
        
        self.directional_sector_models = {}
        
        for sector, stocks in sectors.items():
            logger.info(f"Training Directional {sector} Model...")
            # Aggregate data from multiple stocks in the sector to build a robust model
            X_all, y_all = [], []
            
            for symbol in stocks:
                try:
                    X, y_dict = self.prepare_directional_data(symbol)
                    if X is not None and not X.empty:
                        # Combine all targets for training
                        y_combined = pd.DataFrame(y_dict)
                        X_repeated = pd.concat([X] * len(y_combined.columns), ignore_index=True)
                        y_combined_flat = pd.concat([y_combined[col] for col in y_combined.columns], ignore_index=True)
                        
                        # Only use non-NaN values
                        mask = ~y_combined_flat.isna()
                        if mask.any():
                            X_all.append(X_repeated[mask])
                            y_all.append(y_combined_flat[mask])
                except Exception as e:
                    logger.warning(f"Skipping {symbol} for directional training: {e}")
            
            if X_all:
                X_train = pd.concat(X_all)
                y_train = pd.concat(y_all)
                
                model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                model.fit(X_train, y_train.astype(int))
                self.directional_sector_models[sector] = model
                logger.info(f"✅ Directional {sector} Model Trained.")
            else:
                logger.warning(f"❌ Could not train Directional {sector} model.")

    def train_sector_models(self):
        """Trains specific models for different sectors."""
        sectors = {
            "BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "^NSEBANK"],
            "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
            "ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL"],
            "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
            "GENERIC": ["^NSEI", "^BSESN", "ITC", "LT", "TITAN"]
        }
        
        self.sector_models = {}
        
        for sector, stocks in sectors.items():
            logger.info(f"Training {sector} Model...")
            # Aggregate data from multiple stocks in the sector to build a robust model
            X_all, y_all = [], []
            
            for symbol in stocks:
                try:
                    X, y = self.prepare_data(symbol)
                    if X is not None and not X.empty:
                        X_all.append(X)
                        y_all.append(y)
                except Exception as e:
                    logger.warning(f"Skipping {symbol} for training: {e}")
            
            if X_all:
                X_train = pd.concat(X_all)
                y_train = np.concatenate(y_all)
                
                model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                             class_weight='balanced', random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                self.sector_models[sector] = model
                logger.info(f"✅ {sector} Model Trained.")
            else:
                logger.warning(f"❌ Could not train {sector} model.")

    def get_sector(self, symbol):
        # Simple mapping
        if symbol in ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "^NSEBANK", "INDUSINDBK", "BANKBARODA"]:
            return "BANK"
        if symbol in ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"]:
            return "IT"
        if symbol in ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "COALINDIA"]:
            return "ENERGY"
        if symbol in ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"]:
            return "AUTO"
        return "GENERIC"

    def predict_direction(self, symbol, horizon="1d"):
        """Predicts direction (up/down) for a given stock and timeframe."""
        if not self.is_trained:
            # Models should be trained by now, but just in case
            self.train_sector_models()
            self.train_directional_sector_models()
            
        # Select appropriate model
        sector = self.get_sector(symbol)
        model = self.directional_sector_models.get(sector, self.directional_sector_models.get("GENERIC"))
        
        if not model:
            return None
            
        # Fetch data for prediction
        X, _ = self.prepare_directional_data(symbol, period="3mo")
        if X is None or len(X) == 0: return None
        
        latest = X.iloc[[-1]]
        
        # Get prediction probability
        try:
            prob = model.predict_proba(latest)[0][1]  # Probability of going UP
            prediction = "UP" if prob > 0.5 else "DOWN"
            confidence = max(prob, 1-prob) * 100
            
            return {
                "direction": prediction,
                "confidence": f"{confidence:.1f}%",
                "probability_up": prob,
                "horizon": horizon
            }
        except Exception as e:
            logger.error(f"Error in directional prediction for {symbol}: {e}")
            return None

    def suggest_strikes(self, spot_price, strategy, options_data=None, vix=15.0):
        """Generates specific Strike Prices for the suggested strategy with price lookups."""
        # Fallback for Index/Interval logic if no options data
        # Even without expiry date, we can suggest strikes based on spot price.
        
        if spot_price <= 0:
            return {}
            
        # Determine Strike Width based on Index vs Stock
        is_index = spot_price > 20000 
        interval = 100 if is_index else 50
        
        atm_strike = round(spot_price / interval) * interval
        
        # Get expiry if available, else generic
        expiry = options_data.get("Expiry", "Nearest Weekly") if options_data else "Nearest Weekly"
        price_map = options_data.get("Prices", {}) if options_data else {}
        
        def get_price_str(strike, type_):
            # Try real market data first
            if (strike, type_) in price_map:
                return f"{strike} {type_} (@ ₹{price_map[(strike, type_)]:.1f})"
            
            # Fallback to Black-Scholes
            # Estimating time to expiry (approx 4 days for weekly or just generic 5 days)
            t = 5/252 
            est_price = black_scholes(spot_price, strike, t, 0.07, vix/100, "call" if type_ == "CE" else "put")
            return f"{strike} {type_} (Est ₹{est_price:.1f})"
        
        suggestions = {}
        
        if "STRADDLE" in strategy:
            # Long Straddle: Buy ATM Call & Put
            suggestions = {
                "Leg 1 (Buy CE)": get_price_str(atm_strike, "CE"),
                "Leg 2 (Buy PE)": get_price_str(atm_strike, "PE"),
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
                "Sell Call (Short)": get_price_str(short_ce, "CE"),
                "Buy Call (Hedge)": get_price_str(long_ce, "CE"),
                "Sell Put (Short)": get_price_str(short_pe, "PE"),
                "Buy Put (Hedge)": get_price_str(long_pe, "PE"),
                "Ideal Expiry": expiry,
                "Note": "Collect premium. Max profit if price stays between Short strikes."
            }
            
        return suggestions

    def predict(self, symbol):
        if not self.is_trained:
            # First time load: Train all sector models
            self.train_sector_models() 
            self.train_directional_sector_models()
            self.is_trained = True
            
        # Select appropriate model
        sector = self.get_sector(symbol)
        model = self.sector_models.get(sector, self.sector_models.get("GENERIC"))
        
        if not model:
            return None 
            
        # 1. Fetch Price Data & Run ML Prediction
        X, _ = self.prepare_data(symbol, period="3mo")
        if X is None or len(X) == 0: return None
        
        latest = X.iloc[[-1]]
        prob = model.predict_proba(latest)[0][1] # Prob of High Vol
        
        # Get VIX from latest row for pricing
        current_vix = latest['Vix_Level'].values[0] if 'Vix_Level' in latest else 15.0
        
        regime = "HIGH VOLATILITY" if prob > 0.4 else "QUIET / RANGE-BOUND"
        strategy = "LONG STRADDLE/STRANGLE" if prob > 0.4 else "IRON CONDOR / CREDIT SPREAD"
        
        # 2. Fetch Extra Features
        sent_score, sent_label, headlines = self.sentiment_engine.get_sentiment(symbol)
        fundamentals = get_fundamentals(symbol)
        options_data = get_option_chain_summary(symbol)
        live_price, change, pct_change = self.connector.get_live_price(symbol)
        
        # 3. Generate Specific Strike Suggestions
        trade_setup = self.suggest_strikes(live_price, strategy, options_data, vix=current_vix)
        
        # 4. Get Directional Predictions for various horizons
        directional_predictions = {}
        horizons = ["1d", "3d", "5d", "15d", "1m", "3m", "6m", "1y"]
        
        for horizon in horizons:
            try:
                dir_pred = self.predict_direction(symbol, horizon)
                if dir_pred:
                    directional_predictions[horizon] = dir_pred
            except Exception as e:
                logger.warning(f"Could not get directional prediction for {symbol} {horizon}: {e}")
                directional_predictions[horizon] = None
        
        return {
            "Symbol": symbol,
            "Regime": regime,
            "Confidence": f"{max(prob, 1-prob)*100:.1f}%",
            "Strategy": strategy,
            "Prob_High_Vol": prob,
            "Live_Price": live_price,
            "Price_Change": change,
            "Pct_Change": pct_change,
            "Sector_Model": sector, # Info for UI
            "Sentiment": {
                "Score": sent_score,
                "Label": sent_label,
                "Headlines": headlines
            },
            "Fundamentals": fundamentals,
            "Options": options_data,
            "Trade_Setup": trade_setup,
            "Directional_Predictions": directional_predictions
        }