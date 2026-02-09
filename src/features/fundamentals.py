import yfinance as yf
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_fundamentals(symbol):
    ticker_name = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
    try:
        t = yf.Ticker(ticker_name)
        info = t.info
        
        data = {
            "PE_Ratio": info.get("trailingPE", "N/A"),
            "Forward_PE": info.get("forwardPE", "N/A"),
            "Market_Cap_Cr": round(info.get("marketCap", 0) / 10000000, 2) if info.get("marketCap") else "N/A",
            "Dividend_Yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
            "Sector": info.get("sector", "Unknown"),
            "52W_High": info.get("fiftyTwoWeekHigh", 0),
            "52W_Low": info.get("fiftyTwoWeekLow", 0),
            "Current_Price": info.get("currentPrice", 0)
        }
        return data
    except Exception as e:
        logger.error(f"Fundamentals error for {symbol}: {e}")
        return {}

def get_option_chain_summary(symbol):
    # Handle Indices vs Stocks naming for Yahoo
    if symbol.startswith("^"):
        ticker_name = symbol
    else:
        ticker_name = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        
    try:
        t = yf.Ticker(ticker_name)
        
        # Get nearest expiry
        expirations = t.options
        if not expirations:
            # Fallback: Try removing/adding .NS if failed
            alt_name = symbol if ticker_name.endswith(".NS") else f"{symbol}.NS"
            t = yf.Ticker(alt_name)
            expirations = t.options
            
            if not expirations:
                return None
            
        expiry = expirations[0]
        chain = t.option_chain(expiry)
        
        calls = chain.calls
        puts = chain.puts
        
        # Calculate PCR (Open Interest)
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0
        
        # Calculate Max Pain
        strikes = set(calls['strike']).union(set(puts['strike']))
        if not strikes:
            return None
            
        # Optimization: Filter strikes near spot price to speed up Max Pain calc
        # Get approx price from options mean if live price unavailable
        avg_strike = np.mean(list(strikes))
        relevant_strikes = [k for k in strikes if 0.8 * avg_strike < k < 1.2 * avg_strike]
        
        min_loss = float('inf')
        max_pain_strike = 0
        
        for k in relevant_strikes:
            call_loss = calls.apply(lambda x: max(0, k - x['strike']) * x['openInterest'], axis=1).sum()
            put_loss = puts.apply(lambda x: max(0, x['strike'] - k) * x['openInterest'], axis=1).sum()
            
            total_loss = call_loss + put_loss
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = k
                
        return {
            "PCR_OI": pcr,
            "Max_Pain": max_pain_strike,
            "Expiry": expiry,
            "Call_OI_Change": calls.get('change', pd.Series()).sum(),
            "Put_OI_Change": puts.get('change', pd.Series()).sum()
        }
        
    except Exception as e:
        logger.error(f"Option Chain error for {symbol}: {e}")
        return None
