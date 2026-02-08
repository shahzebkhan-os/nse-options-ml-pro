import yfinance as yf
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataConnector:
    def __init__(self):
        pass

    def fetch_ohlcv(self, symbol, period="1y", interval="1d"):
        """Fetches historical data from yfinance."""
        if symbol.startswith("^"):
            ticker = symbol
        else:
            ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
            
        logger.info(f"Fetching data for {ticker}...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return None
            
            # Flatten MultiIndex columns if present (common in new yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                # If level 1 is the ticker, drop it
                if len(df.columns.levels) > 1:
                    df.columns = df.columns.get_level_values(0)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def get_live_price(self, symbol):
        """Fetches the latest real-time price using yfinance fast_info."""
        ticker_name = f"{symbol}.NS" if not symbol.endswith(".NS") and not symbol.startswith("^") else symbol
        try:
            ticker = yf.Ticker(ticker_name)
            # fast_info is faster and often more up-to-date than .info
            price = ticker.fast_info.get('last_price')
            prev_close = ticker.fast_info.get('previous_close')
            
            if price and prev_close:
                change = price - prev_close
                pct_change = (change / prev_close) * 100
                return price, change, pct_change
            return 0.0, 0.0, 0.0
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
            return 0.0, 0.0, 0.0

    def mock_option_chain(self, symbol, spot_price):
        """Generates a mock option chain for demo purposes."""
        logger.info(f"Generating mock option chain for {symbol} at spot {spot_price}")
        strikes = np.arange(int(spot_price * 0.9), int(spot_price * 1.1), 20)
        chain = []
        for k in strikes:
            # Simple mock logic: calls cheaper as strike goes up
            call_price = max(0, spot_price - k) + np.random.uniform(5, 20)
            put_price = max(0, k - spot_price) + np.random.uniform(5, 20)
            chain.append({
                "strike": k,
                "ce_price": call_price,
                "pe_price": put_price,
                "ce_oi": np.random.randint(1000, 50000),
                "pe_oi": np.random.randint(1000, 50000)
            })
        return pd.DataFrame(chain)
