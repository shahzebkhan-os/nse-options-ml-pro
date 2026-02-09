import pandas as pd
import numpy as np
from src.options.bs_pricing import black_scholes
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StrategyBacktester:
    def __init__(self):
        self.connector = DataConnector()
        self.risk_free_rate = 0.07 # 7% India risk free

    def simulate_strategy(self, symbol, period="1y", initial_capital=100000):
        """
        Simulates P&L for AI-recommended strategies over history.
        Uses Black-Scholes to estimate historical option prices.
        """
        logger.info(f"Backtesting Strategy on {symbol}...")
        
        # 1. Fetch History
        df = self.connector.fetch_ohlcv(symbol, period=period)
        if df is None or df.empty: return None

        # Fetch VIX for pricing
        vix = self.connector.fetch_ohlcv("^INDIAVIX", period=period)
        if vix is not None:
            df['VIX'] = vix['Close']
        else:
            df['VIX'] = 15.0 # Default

        # 2. Compute Features & Signals (Re-create AI Logic simple version)
        df = compute_indicators(df)
        df['Log_Ret'] = np.log(df['Close']).diff()
        
        # Simplified Regime Rule for Backtest (Proxy for ML Model)
        # In real usage, we would run the full RF model day-by-day, 
        # but for speed we use a rolling volatility proxy which matches the ML target.
        df['Vol_Proxy'] = df['Log_Ret'].rolling(5).std() * np.sqrt(252)
        
        # Signal: High Vol if recent vol > 75th percentile, else Quiet
        high_vol_thresh = df['Vol_Proxy'].quantile(0.75)
        
        trades = []
        equity = initial_capital
        equity_curve = [initial_capital]
        
        # Weekly Expiry Simulation (approx 5 trading days)
        holding_period = 5 
        
        for i in range(20, len(df) - holding_period, holding_period):
            date = df.index[i]
            entry_price = df['Close'].iloc[i]
            exit_price = df['Close'].iloc[i+holding_period]
            curr_vix = df['VIX'].iloc[i] / 100
            
            # Determine Regime
            is_high_vol = df['Vol_Proxy'].iloc[i] > high_vol_thresh
            
            pnl = 0
            strategy_name = ""
            
            if is_high_vol:
                # === LONG STRADDLE ===
                # Buy ATM Call + Buy ATM Put
                strategy_name = "Long Straddle"
                strike = entry_price
                t = 5/252 # 5 days to expiry
                
                # Estimate Cost (Debit)
                call_price = black_scholes(entry_price, strike, t, self.risk_free_rate, curr_vix, "call")
                put_price = black_scholes(entry_price, strike, t, self.risk_free_rate, curr_vix, "put")
                debit = call_price + put_price
                
                # Estimate Exit Value (Intrinsic Value at Expiry)
                call_value = max(0, exit_price - strike)
                put_value = max(0, strike - exit_price)
                exit_value = call_value + put_value
                
                pnl = (exit_value - debit) * 100 # Assume 1 lot = 100 qty (simplified)
                
            else:
                # === IRON CONDOR (Short Vol) ===
                # Sell OTM Call/Put, Buy Far OTM Call/Put
                strategy_name = "Iron Condor"
                t = 5/252
                
                # Wings
                short_ce = entry_price * 1.03
                long_ce = entry_price * 1.05
                short_pe = entry_price * 0.97
                long_pe = entry_price * 0.95
                
                # 1. Short Strangle Premium (Credit)
                sc_price = black_scholes(entry_price, short_ce, t, self.risk_free_rate, curr_vix, "call")
                sp_price = black_scholes(entry_price, short_pe, t, self.risk_free_rate, curr_vix, "put")
                
                # 2. Long Strangle Hedge (Debit)
                lc_price = black_scholes(entry_price, long_ce, t, self.risk_free_rate, curr_vix, "call")
                lp_price = black_scholes(entry_price, long_pe, t, self.risk_free_rate, curr_vix, "put")
                
                net_credit = (sc_price + sp_price) - (lc_price + lp_price)
                
                # Exit Liability
                sc_val = max(0, exit_price - short_ce)
                sp_val = max(0, short_pe - exit_price)
                lc_val = max(0, exit_price - long_ce)
                lp_val = max(0, long_pe - exit_price)
                
                net_debit_exit = (sc_val + sp_val) - (lc_val + lp_val)
                
                pnl = (net_credit - net_debit_exit) * 100
                
            equity += pnl
            equity_curve.extend([equity] * holding_period) # Fill days
            
            trades.append({
                "Date": date,
                "Strategy": strategy_name,
                "Entry": round(entry_price, 2),
                "Exit": round(exit_price, 2),
                "PnL": round(pnl, 2),
                "Equity": round(equity, 2)
            })
            
        return pd.DataFrame(trades), equity_curve[:len(df)]
