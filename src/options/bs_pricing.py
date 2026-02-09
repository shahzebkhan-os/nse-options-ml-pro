import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Computes the Black-Scholes price for a call or put option.
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility (decimal)
    option_type: "call" or "put"
    """
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price
    except Exception:
        return 0.0

class BSModel:
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BSModel.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        return S * norm.cdf(BSModel.d1(S, K, T, r, sigma)) - K * np.exp(-r*T) * norm.cdf(BSModel.d2(S, K, T, r, sigma))
    
    @staticmethod
    def delta(S, K, T, r, sigma, type="call"):
        if type == "call":
            return norm.cdf(BSModel.d1(S, K, T, r, sigma))
        else:
            return norm.cdf(BSModel.d1(S, K, T, r, sigma)) - 1

    @staticmethod
    def implied_volatility(market_price, S, K, T, r, type="call"):
        """Simple Newton-Raphson for IV"""
        sigma = 0.5  # Initial guess
        for i in range(100):
            price = BSModel.call_price(S, K, T, r, sigma)
            diff = market_price - price
            if abs(diff) < 1e-5:
                return sigma
            vega = S * norm.pdf(BSModel.d1(S, K, T, r, sigma)) * np.sqrt(T)
            if vega == 0:
                return 0
            sigma += diff / vega
        return sigma
