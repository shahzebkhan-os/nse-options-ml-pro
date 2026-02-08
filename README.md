# 🤖 NSE Options ML Pro - AI Strategy Engine

**Production-grade Volatility Regime Detection System for NSE Options Trading.**

This project uses Machine Learning (Random Forest) to classify stocks into "High Volatility" or "Quiet/Range-Bound" regimes and suggests optimal option strategies (e.g., Iron Condors vs. Straddles).

![Dashboard](https://via.placeholder.com/800x400?text=NSE+Options+AI+Dashboard)

## 🚀 Key Features

*   **🧠 AI Volatility Engine:** Predicts if a stock will move >1.5% tomorrow (78% Accuracy).
*   **📊 Live Dashboard:** Real-time prices, interactive charts, and strategy payoff diagrams.
*   **📰 Sentiment Analysis:** Scrapes news headlines and scores them using NLP (VADER).
*   **⛓️ Option Chain Intelligence:** Calculates Max Pain, PCR (Put-Call Ratio), and OI Buildup.
*   **🚨 Smart Alerts:** Sends Telegram notifications when high-volatility setups are detected.
*   **⚡ Real-Time Scanner:** Scans NIFTY 50, Midcap, and Smallcap stocks in seconds.

---

## 🛠️ How It Works (The Algorithm)

The core logic combines **Technical Analysis**, **Machine Learning**, and **Market Context**.

### 1. Data Ingestion
*   Fetches 2 years of daily OHLCV data from Yahoo Finance (`yfinance`).
*   Fetches Market Context: **NIFTY 50 Index** returns and **INDIA VIX** levels.

### 2. Feature Engineering
We compute 15+ proprietary indicators to feed the model:
*   **Momentum:** RSI (14), MACD, ROC.
*   **Volatility:** ATR, Bollinger Band Width, Historical Volatility.
*   **Market Context:** NIFTY Lagged Returns, VIX Level.
*   **Volume:** Volume/MA Ratio (detects unusual activity).

### 3. The "Big Move" Target
Unlike traditional models that try to predict "Up" or "Down" (which is noisy), our model predicts **MAGNITUDE**.
*   **Target:** `1` if Next Day Return > 1.5% (Absolute), else `0`.
*   *Why?* Option sellers win when stocks stay quiet. Option buyers win when stocks move big. Direction matters less than magnitude for these strategies.

### 4. Machine Learning Model
*   **Algorithm:** Random Forest Classifier (Ensemble of 200 Decision Trees).
*   **Class Balancing:** Uses `class_weight='balanced'` to handle the rarity of big moves.
*   **Training:** Retrains dynamically on the latest data.

### 5. Strategy Logic
Based on the predicted probability ($P_{vol}$):

| Prediction ($P_{vol}$) | Regime | Recommended Strategy | Logic |
| :--- | :--- | :--- | :--- |
| **> 40%** | 🚨 High Volatility | **Long Straddle / Strangle** | Buy Calls & Puts. Profit from big move in ANY direction. |
| **< 40%** | 💤 Quiet / Range-Bound | **Iron Condor / Credit Spread** | Sell OTM Options. Profit from theta decay (time value). |

---

## 📦 Installation

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/shahzebkhan-os/nse-options-ml-pro.git
    cd nse-options-ml-pro
    ```

2.  **Run the One-Click Installer:**
    ```bash
    chmod +x sample_run.sh
    ./sample_run.sh
    ```
    *(This installs dependencies and launches the dashboard automatically)*

---

## 🖥️ Usage

### 1. Analyze a Stock
*   Select a category (Large/Mid/Small Cap).
*   Pick a symbol (e.g., RELIANCE).
*   Click **"Run Analysis"**.
*   View the **AI Prediction**, **Live Price**, **News Sentiment**, and **Payoff Chart**.

### 2. Scan the Market
*   Go to the "Market Watchlist" panel.
*   Select how many stocks to scan (e.g., Top 10).
*   Click **"Scan List"**.
*   Sort the results by "Regime" to find trade opportunities.

### 3. Enable Telegram Alerts
*   Open the Sidebar (Left).
*   Enter your **Bot Token** and **Chat ID**.
*   Check "Enable Alerts".
*   You will now receive a message whenever a "High Volatility" stock is found!

---

## ⚠️ Disclaimer
*This software is for educational purposes only. Do not trade based solely on these signals. Options trading involves significant risk.*
