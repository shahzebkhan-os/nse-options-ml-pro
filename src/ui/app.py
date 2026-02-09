import streamlit as st
import pandas as pd
from src.pipeline.predictor import VolatilityPredictor
from src.utils.stock_lists import ALL_STOCKS
from src.utils.alerts import send_telegram_alert
from src.ui.charts import plot_interactive_chart
from src.ui.payoff import plot_payoff_diagram
from src.features.indicators import compute_indicators
from src.backtest.strategy_simulator import StrategyBacktester
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="NIFTY Options AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.subheader("Market Scope")
    # Merge Indices + Large Cap for the dropdown
    combined_list = ALL_STOCKS["Indices (NIFTY/BANKNIFTY/SENSEX)"] + ALL_STOCKS["Large Cap (Nifty 50 Stocks)"]
    st.info(f"Active Universe: {len(combined_list)} Assets\n(Indices + NIFTY 50)")

    st.markdown("---")
    st.subheader("📡 Watchlist Scanner")
    scan_limit = st.slider("Scan Depth", 5, 50, 10)
    run_scan = st.button("🚀 Scan Market Now")
    
    st.markdown("---")
    st.subheader("🔔 Alerts")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat_id = st.text_input("Chat ID")
    send_alerts = st.checkbox("Enable Telegram Alerts")

# --- Initialize ---
@st.cache_resource
def get_predictor():
    pred = VolatilityPredictor()
    pred.train("RELIANCE") 
    return pred

@st.cache_resource
def get_backtester():
    return StrategyBacktester()

predictor = get_predictor()
backtester = get_backtester()

# --- Main Page ---
st.title("🤖 NIFTY50 Options AI Engine")
st.markdown(f"**AI-Powered Volatility Regime Detection & Strategy Generator** | *Model Accuracy: ~80%*")

# --- Stock Selector Row ---
col_sel1, col_sel2 = st.columns([1, 4])
with col_sel1:
    symbol = st.selectbox("Select Asset", combined_list, index=0)
with col_sel2:
    st.write("") # Spacer
    st.write("") 
    run_analysis = st.button("🔍 Analyze Asset", type="primary")

# --- Main Logic ---

# 1. SCANNER LOGIC (Sidebar Trigger)
if run_scan:
    st.divider()
    st.subheader(f"🔍 Market Scan Results (Top {scan_limit})")
    
    results = []
    progress = st.progress(0)
    status_text = st.empty()
    
    watchlist = combined_list[:scan_limit]
    
    for i, stock in enumerate(watchlist):
        status_text.caption(f"Scanning {stock} ({i+1}/{len(watchlist)})...")
        res = predictor.predict(stock)
        if res:
            row = {
                "Symbol": res["Symbol"],
                "Price": f"₹{res['Live_Price']:.2f}",
                "Change": f"{res['Pct_Change']:.2f}%",
                "Regime": res["Regime"],
                "Conf": res["Confidence"],
                "PCR": res["Options"]["PCR_OI"] if res["Options"] else 0,
            }
            results.append(row)
        progress.progress((i + 1) / len(watchlist))
        
    status_text.empty()
    if results:
        df = pd.DataFrame(results)
        
        # Color styling
        def highlight_regime(val):
            color = 'red' if 'HIGH' in val else 'green'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df.style.applymap(highlight_regime, subset=['Regime']),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("No data found.")

# 2. ANALYSIS LOGIC (Main Button)
if run_analysis:
    with st.spinner(f"Running AI Models on {symbol}..."):
        result = predictor.predict(symbol)
        
    if result:
        # --- Signal Card ---
        with st.container():
            # Color code based on regime
            is_high_vol = "HIGH" in result["Regime"]
            color_emoji = "🚨" if is_high_vol else "💤"
            
            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Price", f"₹{result['Live_Price']:.2f}", f"{result['Price_Change']:.2f} ({result['Pct_Change']:.2f}%)")
            m2.metric("AI Confidence", result["Confidence"])
            m3.metric("Big Move Prob", f"{result['Prob_High_Vol']:.2f}")
            m4.metric("PCR (OI)", result["Options"]["PCR_OI"] if result["Options"] else "N/A")

            # The Verdict
            st.divider()
            v_col1, v_col2 = st.columns([2, 1])
            
            with v_col1:
                st.subheader(f"{color_emoji} {result['Regime']}")
                st.caption("AI Detected Market Condition")
                
                if is_high_vol:
                    st.error(f"Strategy: **{result['Strategy']}**")
                    st.markdown("*Expect significant price movement. Buy Volatility.*")
                else:
                    st.success(f"Strategy: **{result['Strategy']}**")
                    st.markdown("*Expect range-bound action. Sell Volatility / Eat Theta.*")
                
                # --- Trade Setup Card ---
                setup = result.get("Trade_Setup", {})
                if setup:
                    st.markdown("#### 🎯 Execution Plan")
                    # Create a nice grid for the strikes
                    cols = st.columns(len(setup)-1) # -1 to exclude 'Note'
                    for i, (key, value) in enumerate(setup.items()):
                        if key != "Note":
                            cols[i % len(cols)].metric(key, value)
                    st.caption(f"📝 *Note: {setup.get('Note', '')}*")
            
            with v_col2:
                # Mini Sentiment
                sent = result["Sentiment"]
                s_color = "green" if sent["Label"] == "BULLISH" else "red" if sent["Label"] == "BEARISH" else "gray"
                st.markdown(f"**News Sentiment:** :{s_color}[{sent['Label']}]")
                st.progress((sent['Score'] + 1) / 2) # Normalize -1..1 to 0..1

        # --- Visualizations ---
        st.divider()
        tab_chart, tab_payoff, tab_backtest, tab_fund = st.tabs(["📈 Technical Chart", "💰 Payoff Diagram", "🧪 Strategy Backtest", "📊 Deep Dive"])
        
        with tab_chart:
            df_chart = predictor.connector.fetch_ohlcv(symbol, period="1y")
            df_chart = compute_indicators(df_chart)
            fig_chart = plot_interactive_chart(df_chart, symbol)
            st.plotly_chart(fig_chart, use_container_width=True)
            
        with tab_payoff:
            fig_payoff, desc = plot_payoff_diagram(result['Strategy'], result['Live_Price'])
            st.plotly_chart(fig_payoff, use_container_width=True)
            st.info(desc)
            
        with tab_backtest:
            st.markdown(f"### 🧪 Historical Performance: {symbol}")
            st.caption("Simulating AI Strategy (Iron Condor vs Straddle) over the last 1 year.")
            
            if st.button("Run Backtest Simulation"):
                with st.spinner("Simulating trades..."):
                    trades_df, equity_curve = backtester.simulate_strategy(symbol, period="1y")
                    
                    if trades_df is not None and not trades_df.empty:
                        # KPI Cards
                        total_pnl = trades_df['PnL'].sum()
                        win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
                        num_trades = len(trades_df)
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{(total_pnl/100000)*100:.1f}% Return")
                        k2.metric("Win Rate", f"{win_rate:.1f}%")
                        k3.metric("Trades Executed", num_trades)
                        
                        # Growth Chart
                        st.subheader("Equity Curve (Start: ₹1L)")
                        st.line_chart(trades_df.set_index("Date")["Equity"])
                        
                        # Trade Log
                        with st.expander("View Trade Log"):
                            st.dataframe(trades_df)
                    else:
                        st.warning("Not enough data to backtest.")
            else:
                st.info("Click the button to simulate strategy performance.")
            
        with tab_fund:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Fundamentals")
                fun = result["Fundamentals"]
                if fun:
                    st.json(fun)
                else:
                    st.write("No data.")
            with c2:
                st.markdown("### Option Chain Stats")
                opt = result["Options"]
                if opt:
                    st.write(f"**Max Pain:** {opt['Max_Pain']}")
                    st.write(f"**Call OI Chg:** {opt['Call_OI_Change']}")
                    st.write(f"**Put OI Chg:** {opt['Put_OI_Change']}")
                else:
                    st.write("No Option Chain data.")

        # --- Alerts Trigger ---
        if send_alerts and is_high_vol:
            msg = f"🚨 *High Volatility Detected!* \nSymbol: {symbol}\nPrice: {result['Live_Price']}\nStrategy: {result['Strategy']}"
            send_telegram_alert(tg_token, tg_chat_id, msg)

    else:
        st.error("Could not fetch data for this symbol.")
