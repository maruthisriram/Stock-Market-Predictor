import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import fetch_historical_data, fetch_news_sentiment, get_market_movers
from predictor import predict_movement
import yfinance as yf
import time

# Page Config
st.set_page_config(page_title="Stock Sentiment Predictor Pro", layout="wide")

# Safe CSS (avoiding blocking styles)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4461;
        margin-bottom: 10px;
    }
    .prediction-card {
        padding: 2.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        color: white;
    }
    .up-card {
        background: linear-gradient(135deg, #1d4d2f 0%, #2e7d32 100%);
        border: 2px solid #4caf50;
    }
    .down-card {
        background: linear-gradient(135deg, #4d1d1d 0%, #c62828 100%);
        border: 2px solid #f44336;
    }
    .neutral-card {
        background: linear-gradient(135deg, #222 0%, #444 100%);
        border: 2px solid #666;
    }
    .news-item {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #1e2130;
        margin-bottom: 0.5rem;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📈 Stock Intelligence Dashboard")

# Caching
@st.cache_data(ttl=600)
def get_cached_movers():
    try:
        return get_market_movers()
    except:
        return [], []

@st.cache_data(ttl=300)
def get_cached_ticker_data(symbol, period):
    df = fetch_historical_data(symbol, period=period)
    avg_sent, news = fetch_news_sentiment(symbol)
    return df, avg_sent, news

# Sidebar
st.sidebar.header("Analysis Parameters")
symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y"])

# Layout: Two columns (Main Analysis vs Market Overview)
col_main, col_market = st.columns([0.7, 0.3])

with col_main:
    if symbol:
        try:
            with st.status(f"Fetching intelligence for {symbol}...", expanded=True) as status:
                st.write("Retreiving technical data...")
                df, avg_sentiment, news_articles = get_cached_ticker_data(symbol, period)
                
                if df is not None and not df.empty:
                    st.write("Calculating market sentiment...")
                    prediction, reason = predict_movement(df, avg_sentiment)
                    status.update(label=f"Analysis Complete for {symbol}", state="complete", expanded=False)
                    
                    # Metrics Grid
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    m2.metric("News Sentiment", f"{avg_sentiment:.2f}")
                    if len(df) > 1:
                        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                        m3.metric("Daily Shift", f"${change:.2f}", delta=f"{change:.2f}")

                    # Prediction Visual
                    card_class = "up-card" if prediction == "UP" else ("down-card" if prediction == "DOWN" else "neutral-card")
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h1 style="margin:0; font-size: 3rem;">{prediction}</h1>
                        <p style="font-size: 1.2rem; opacity: 0.9;">{reason}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Content Tabs
                    tab_c, tab_n = st.tabs(["📊 Technical Chart", "📰 Logic Source News"])
                    
                    with tab_c:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='#00d1ff', width=3)))
                        fig.update_layout(template="plotly_dark", height=400, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)

                    with tab_n:
                        if news_articles:
                            for art in news_articles[:10]:
                                color = "#4caf50" if art['sentiment'] > 0 else ("#f44336" if art['sentiment'] < 0 else "#aaa")
                                st.markdown(f"""
                                <div class="news-item">
                                    <h4 style="margin:0;"><a href="{art['link']}" target="_blank" style="color:#00d1ff; text-decoration:none;">{art['title']}</a></h4>
                                    <div style="margin-top:5px; color:#888;">
                                        <span>{art['publisher']}</span> | 
                                        <span>Sentiment Impact: <b style="color:{color};">{art['sentiment']:.2f}</b></span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No recent news headers found. The prediction is based on technical trends.")
                else:
                    st.error(f"Could not load data for '{symbol}'. Please verify the ticker symbol.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    else:
        st.info("Enter a ticker in the sidebar to begin.")

with col_market:
    st.subheader("🔥 Top Movers")
    with st.spinner("Updating market..."):
        gainers, losers = get_cached_movers()
        
        if gainers:
            st.write("🚀 **Top Gainers**")
            for g in gainers[:5]:
                st.write(f"{g['symbol']} | ${g['price']:.2f} | :green[+{g['change']:.1f}%]")
        
        st.write("---")
        
        if losers:
            st.write("📉 **Top Losers**")
            for l in losers[:5]:
                st.write(f"{l['symbol']} | ${l['price']:.2f} | :red[{l['change']:.1f}%]")

# Sidebar footer
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Diagnostics"):
    st.write(f"V: {yf.__version__}")
    if symbol:
        st.write(f"Target: {symbol}")
        try:
            t = yf.Ticker(symbol)
            raw = getattr(t, 'news', [])
            st.write(f"Raw Items: {len(raw)}")
            if raw:
                st.write("Sample Keys:", list(raw[0].keys()))
                if 'content' in raw[0]:
                    st.write("Content Keys:", list(raw[0]['content'].keys()))
        except Exception as ex:
            st.write(f"Error: {ex}")
