import streamlit as st
import pandas as pd
from src.pipeline.predictor import VolatilityPredictor
from src.utils.stock_lists import ALL_STOCKS

st.set_page_config(page_title="NIFTY Options AI", layout="wide")

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

col1, col2 = st.columns([1, 2])

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
                
                # Display Card
                st.metric("Detected Regime", result["Regime"], delta=result["Confidence"])
                
                st.subheader("Recommended Strategy")
                st.info(f"**{result['Strategy']}**")
                
                st.write(f"Probability of Big Move (>1.5%): **{result['Prob_High_Vol']:.2f}**")
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
                results.append(res)
            progress.progress((i + 1) / len(watchlist))
            
        status_text.text("Scan Complete!")
        
        if results:
            df = pd.DataFrame(results)
            # Styling
            st.dataframe(df.style.applymap(lambda x: 'color: red' if 'HIGH' in str(x) else 'color: green', subset=['Regime']))
        else:
            st.warning("No results found.")
