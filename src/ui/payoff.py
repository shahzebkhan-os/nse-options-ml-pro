import plotly.graph_objects as go
import numpy as np

def plot_payoff_diagram(strategy, spot_price, volatility="LOW"):
    """
    Plots profit/loss diagrams for recommended strategies.
    strategy: "IRON CONDOR..." or "LONG STRADDLE..."
    spot_price: Current stock price
    """
    
    # Define range of prices for simulation (+/- 10%)
    prices = np.linspace(spot_price * 0.9, spot_price * 1.1, 100)
    profit = np.zeros_like(prices)
    
    # Strategy Logic
    strategy_name = ""
    
    if "STRADDLE" in strategy or "STRANGLE" in strategy:
        # Long Straddle: Buy ATM Call + Buy ATM Put
        # Payoff: V-shape. Profit if price moves big in ANY direction.
        strike = spot_price
        premium = spot_price * 0.02 # Approx premium (2%)
        
        # P&L Calculation
        call_payoff = np.maximum(prices - strike, 0) - premium
        put_payoff = np.maximum(strike - prices, 0) - premium
        profit = call_payoff + put_payoff
        
        strategy_name = "Long Straddle (High Volatility Setup)"
        desc = "Profits from BIG moves in either direction. Max Loss = Premium Paid."

    else:
        # Iron Condor / Credit Spread: Sell OTM Call + Sell OTM Put (with hedges)
        # Payoff: Table-shape. Profit if price stays in middle.
        
        # Strikes
        short_call = spot_price * 1.03
        long_call = spot_price * 1.05
        short_put = spot_price * 0.97
        long_put = spot_price * 0.95
        
        net_credit = (spot_price * 0.015) # Approx credit received
        
        # P&L Calculation (Short Volatility)
        # 1. Short Call Spread
        sc_pnl = -np.maximum(prices - short_call, 0) + np.maximum(prices - long_call, 0)
        # 2. Short Put Spread
        sp_pnl = -np.maximum(short_put - prices, 0) + np.maximum(long_put - prices, 0)
        
        profit = sc_pnl + sp_pnl + net_credit
        
        strategy_name = "Iron Condor (Range-Bound Setup)"
        desc = "Profits if price stays between wings. Max Risk is defined/limited."

    # Plot
    fig = go.Figure()
    
    # Payoff Line
    fig.add_trace(go.Scatter(
        x=prices, y=profit,
        mode='lines',
        name='P&L at Expiry',
        line=dict(color='cyan', width=3),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 255, 0.1)' # Slight fill
    ))
    
    # Zero Line (Break-even)
    fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Break Even")
    
    # Current Price Marker
    fig.add_vline(x=spot_price, line_dash="dot", line_color="yellow", annotation_text="Current Price")

    fig.update_layout(
        title=f"Strategy Payoff: {strategy_name}",
        xaxis_title="Stock Price at Expiry",
        yaxis_title="Profit / Loss",
        template="plotly_dark",
        height=400
    )
    
    return fig, desc
