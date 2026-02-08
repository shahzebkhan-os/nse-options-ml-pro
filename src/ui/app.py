import streamlit as st
import pandas as pd
from src.pipeline.predictor import VolatilityPredictor
from src.utils.stock_lists import ALL_STOCKS
from src.utils.alerts import send_telegram_alert
from src.ui.charts import plot_interactive_chart
from src.ui.payoff import plot_payoff_diagram
from src.features.indicators import compute_indicators

st.set_page_config(page_title="NIFTY Options AI", layout="wide")

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### Telegram Alerts")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")
send_alerts = st.sidebar.checkbox("Enable Alerts")

st.title("🤖 NIFTY50 Options AI Strategy Engine")
st.markdown("Use Machine Learning to detect **Volatility Regimes** and suggest Option Strategies.")
st.markdown("---")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    pred = VolatilityPredictor()
    # Pre-train on a representative stock to initialize model weights
    pred.train("RELIANCE") 
    return pred

predictor = get_predictor()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Analyze Stock")
    
    # Category Selection
    category = st.selectbox("Market Cap Category", list(ALL_STOCKS.keys()))
    stock_list = ALL_STOCKS[category]
    
    symbol = st.selectbox("Select Symbol", stock_list)
    
    if st.button("Run Analysis"):
        with st.spinner(f"Analyzing {symbol}..."):
            result = predictor.predict(symbol)
            
            if result:
                st.success("Analysis Complete")
                
                # --- Live Price Header ---
                p_col1, p_col2 = st.columns([1, 3])
                with p_col1:
                    st.metric("Live Price", 
                             f"₹{result['Live_Price']:.2f}", 
                             f"{result['Price_Change']:.2f} ({result['Pct_Change']:.2f}%)")
                
                # --- Main ML Prediction ---
                st.subheader("🤖 AI Prediction")
                m1, m2, m3 = st.columns(3)
                m1.metric("Regime", result["Regime"])
                m2.metric("Confidence", result["Confidence"])
                m3.metric("Prob > 1.5% Move", f"{result['Prob_High_Vol']:.2f}")
                
                st.info(f"**Recommended Strategy:** {result['Strategy']}")
                
                # --- VISUALIZATION TAB ---
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["📈 Technical Chart", "💰 Strategy Payoff", "📰 Sentiment"])
                
                with tab1:
                    # Fetch data again for plotting (efficient caching handles this)
                    df_chart = predictor.connector.fetch_ohlcv(symbol, period="1y")
                    df_chart = compute_indicators(df_chart) # Add bands
                    fig_chart = plot_interactive_chart(df_chart, symbol)
                    st.plotly_chart(fig_chart, use_container_width=True)
                    
                with tab2:
                    fig_payoff, desc = plot_payoff_diagram(result['Strategy'], result['Live_Price'])
                    st.plotly_chart(fig_payoff, use_container_width=True)
                    st.caption(f"ℹ️ **Strategy Logic:** {desc}")
                    
                with tab3:
                    sent = result["Sentiment"]
                    s1, s2 = st.columns([1, 3])
                    s1.metric("Sentiment Score", f"{sent['Score']:.2f}", sent["Label"])
                    with s2:
                        if sent["Headlines"]:
                            for h in sent["Headlines"]:
                                st.write(f"- {h}")
                        else:
                            st.caption("No recent news found.")

                # --- Option Chain ---
                st.markdown("---")
                st.subheader("⛓️ Option Chain Data")
                opt = result["Options"]
                if opt:
                    o1, o2, o3 = st.columns(3)
                    o1.metric("PCR (OI)", opt["PCR_OI"])
                    o2.metric("Max Pain", opt["Max_Pain"])
                    o3.metric("Expiry", str(opt["Expiry"]))
                else:
                    st.warning("Option Chain data unavailable (Market Closed/No Data).")
                    
                # --- Fundamentals ---
                st.markdown("---")
                st.subheader("📊 Fundamentals")
                fun = result["Fundamentals"]
                if fun:
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("P/E Ratio", fun.get("PE_Ratio", "N/A"))
                    f2.metric("Mkt Cap (Cr)", fun.get("Market_Cap_Cr", "N/A"))
                    f3.metric("Div Yield", f"{fun.get('Dividend_Yield', 0)}%")
                    f4.metric("Sector", fun.get("Sector", "N/A"))
                
                # --- Alerts ---
                if send_alerts and "HIGH" in result["Regime"]:
                    msg = f"🚨 *High Volatility Detected!* \nSymbol: {symbol}\nPrice: {result['Live_Price']}\nStrategy: {result['Strategy']}"
                    if send_telegram_alert(tg_token, tg_chat_id, msg):
                        st.toast("Alert Sent to Telegram!", icon="✅")
            else:
                st.error("Failed to fetch data.")

with col2:
    st.header(f"Market Watchlist ({category})")
    
    # Allow user to limit scan size
    scan_limit = st.slider("Stocks to Scan", min_value=5, max_value=len(stock_list), value=10)
    
    if st.button("Scan List"):
        results = []
        # Take the first N stocks from the selected category
        watchlist = stock_list[:scan_limit]
        
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, stock in enumerate(watchlist):
            status_text.text(f"Scanning {stock}...")
            res = predictor.predict(stock)
            if res:
                # Flatten for table
                row = {
                    "Symbol": res["Symbol"],
                    "Price": f"₹{res['Live_Price']:.2f}",
                    "Change": f"{res['Pct_Change']:.2f}%",
                    "Regime": res["Regime"],
                    "Conf": res["Confidence"],
                    "Sentiment": res["Sentiment"]["Label"],
                    "PCR": res["Options"]["PCR_OI"] if res["Options"] else 0,
                    "PE": res["Fundamentals"].get("PE_Ratio", 0) if res["Fundamentals"] else 0
                }
                results.append(row)
            progress.progress((i + 1) / len(watchlist))
            
        status_text.text("Scan Complete!")
        
        if results:
            df = pd.DataFrame(results)
            # Styling
            st.dataframe(df.style.applymap(lambda x: 'color: red' if 'HIGH' in str(x) else 'color: green', subset=['Regime']))
        else:
            st.warning("No results found.")
