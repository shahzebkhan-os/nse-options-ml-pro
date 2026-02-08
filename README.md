# NIFTY50 Options ML Pro

> **DISCLAIMER:** This software is for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**. It is **NOT** financial advice. Trading options involves high risk and you can lose more than your initial investment. The authors and contributors assume no responsibility for any financial losses.

## Overview

A production-grade pipeline to analyze NIFTY50 stocks, predict movements using Neural Networks (LSTM/Transformer), and suggest option strategies based on Implied Volatility surfaces and Greeks.

## Features

- **Data Ingestion:** Connectors for `yfinance` and mock Kite Connect.
- **Advanced Features:** Technical indicators (RSI, MACD, Bollinger) + Option Greeks (Delta, Gamma, Vega).
- **Deep Learning:** PyTorch implementations of LSTM and Transformer encoders for time-series forecasting.
- **Option Engine:** Black-Scholes solver, Newton-Raphson IV calculation, and IV surface fitting.
- **Real-time Pipeline:** Worker queue with ETA calculation (Exponential Moving Average).
- **UI:** Interactive Streamlit dashboard.

## Quickstart (Local)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Demo:**
    ```bash
    chmod +x sample_run.sh
    ./sample_run.sh
    ```
    This will:
    - Train a model on a sample stock (RELIANCE).
    - Run the ETA simulation.
    - Launch the UI.

## Docker

```bash
docker-compose up --build
```

## Architecture

1.  **Ingest:** Fetches OHLCV.
2.  **Process:** Computes indicators.
3.  **Model:** Predicting next-day return direction.
4.  **Options:** Scans for optimal Risk/Reward strikes.
5.  **UI:** Displays results.
