import streamlit as st
import pandas as pd
from src.pipeline.predictor import VolatilityPredictor

st.set_page_config(page_title="NIFTY Options AI", layout="wide")

st.title("🤖 NIFTY50 Options AI Strategy Engine")
st.markdown("Use Machine Learning to detect **Volatility Regimes** and suggest Option Strategies.")
st.markdown("---")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    pred = VolatilityPredictor()
    pred.train("RELIANCE") # Pre-train on a major stock
    return pred

predictor = get_predictor()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Analyze Stock")
    symbol = st.selectbox("Select Symbol", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"])
    
    if st.button("Run Analysis"):
        with st.spinner(f"Analyzing {symbol}..."):
            result = predictor.predict(symbol)
            
            if result:
                st.success("Analysis Complete")
                
                # Display Card
                st.metric("Detected Regime", result["Regime"], delta=result["Confidence"])
                
                st.subheader("Recommended Strategy")
                st.info(f"**{result['Strategy']}**")
                
                st.write(f"Probability of Big Move (>1.5%): **{result['Prob_High_Vol']:.2f}**")
            else:
                st.error("Failed to fetch data.")

with col2:
    st.header("Market Watchlist (Live)")
    if st.button("Scan All"):
        results = []
        watchlist = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "LT", "ITC"]
        progress = st.progress(0)
        
        for i, stock in enumerate(watchlist):
            res = predictor.predict(stock)
            if res:
                results.append(res)
            progress.progress((i + 1) / len(watchlist))
            
        df = pd.DataFrame(results)
        st.dataframe(df.style.applymap(lambda x: 'color: red' if 'HIGH' in str(x) else 'color: green', subset=['Regime']))
