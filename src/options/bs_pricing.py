import numpy as np
from scipy.stats import norm

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
