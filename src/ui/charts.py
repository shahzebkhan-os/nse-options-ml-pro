import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_interactive_chart(df, symbol):
    """
    Creates an interactive Plotly chart with Candlesticks, Bollinger Bands, and Volume.
    """
    if df is None or df.empty:
        return None

    # Create subplots: Row 1 for Price, Row 2 for Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Bollinger Bands
    # Upper
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBU'],
        line=dict(color='gray', width=1, dash='dot'),
        name='Upper BB', showlegend=False
    ), row=1, col=1)
    
    # Lower
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBL'],
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        name='Lower BB', showlegend=False
    ), row=1, col=1)

    # 3. Moving Averages
    if 'MACD' in df.columns: # Just using EMA logic if available or calculating fresh
        ema_20 = df['Close'].ewm(span=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ema_20,
            line=dict(color='orange', width=1.5),
            name='EMA 20'
        ), row=1, col=1)

    # 4. Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=colors,
        name='Volume'
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{symbol} Price Action & Volatility Bands",
        yaxis_title="Price (INR)",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_dark"
    )

    return fig
