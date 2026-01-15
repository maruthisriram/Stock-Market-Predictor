import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import fetch_historical_data, fetch_news_sentiment, get_market_movers, search_stocks, get_sector_performance
from predictor import predict_movement
import yfinance as yf
import time

# Page Config
st.set_page_config(page_title="Stock Sentiment Predictor Pro", layout="wide")

# Initialize session state for symbol if not present
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = "AAPL"

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
    .sector-card {
        padding: 10px;
        border-radius: 5px;
        background: #1e2130;
        margin-bottom: 5px;
        border-left: 5px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📈 Stock Intelligence Dashboard")

# Caching
@st.cache_resource
def load_models():
    """Cache the AI models across sessions and reruns."""
    from data_fetcher import get_finbert_pipeline, get_summarizer_pipeline
    get_finbert_pipeline()
    get_summarizer_pipeline()
    return True

@st.cache_data(ttl=600)
def get_cached_movers():
    try:
        return get_market_movers()
    except:
        return [], []

@st.cache_data(ttl=600)
def get_cached_sectors():
    try:
        return get_sector_performance()
    except:
        return []

@st.cache_data(ttl=300)
def get_cached_ticker_data(symbol, period):
    # Pre-warm models cache
    load_models()
    df = fetch_historical_data(symbol, period=period)
    avg_sent, news = fetch_news_sentiment(symbol)
    return df, avg_sent, news

# Sidebar
st.sidebar.header("Analysis Parameters")

# Sidebar Helper to change symbol and clear search
def set_symbol(new_sym):
    st.session_state.current_symbol = new_sym
    # Reset search inputs if keys are used
    if 'search_input' in st.session_state:
        st.session_state.search_input = ""
    st.rerun()

# Search with Suggestions
search_query = st.sidebar.text_input("Search Company or Ticker", placeholder="e.g. Reliance, Apple...", key="search_input")
if search_query:
    suggestions = search_stocks(search_query)
    if suggestions:
        options = [f"{s['name']} ({s['symbol']})" for s in suggestions]
        selected_option = st.sidebar.selectbox("Select a company", options, index=None, placeholder="Choose a result...", key="search_select")
        if selected_option:
            # Extract symbol from "Name (SYMBOL)"
            new_symbol = selected_option.split('(')[-1].strip(')')
            if new_symbol != st.session_state.current_symbol:
                set_symbol(new_symbol)
    
    if st.sidebar.button("Clear Search"):
        st.session_state.search_input = ""
        st.rerun()

symbol_input = st.sidebar.text_input("Active Ticker", value=st.session_state.current_symbol).upper().strip()
if symbol_input != st.session_state.current_symbol:
    st.session_state.current_symbol = symbol_input

period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y"])

# Layout: Two columns (Main Analysis vs Market Overview)
col_main, col_market = st.columns([0.7, 0.3])

with col_main:
    if st.session_state.current_symbol:
        try:
            with st.status(f"Fetching intelligence for {st.session_state.current_symbol}...", expanded=True) as status:
                st.write("Retreiving technical data...")
                df, avg_sentiment, news_articles = get_cached_ticker_data(st.session_state.current_symbol, period)
                
                if df is not None and not df.empty:
                    st.write("Calculating market sentiment...")
                    prediction, reason = predict_movement(df, avg_sentiment)
                    status.update(label=f"Analysis Complete for {st.session_state.current_symbol}", state="complete", expanded=False)
                    
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
                    tab_c, tab_n, tab_s = st.tabs(["📊 Technical Chart", "🌐 Meaning-aware Logic", "📈 Sector Analysis"])
                    
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
                                    <p style="font-size: 0.9rem; color: #ccc; margin: 10px 0;"><b>Summary:</b> {art['summary']}</p>
                                    <div style="margin-top:5px; color:#888;">
                                        <span>{art['publisher']}</span> | 
                                        <span>Confidence: <b style="color:{color};">{abs(art['sentiment']):.2f}</b></span>
                                    </div>
                                    <div style="margin-top:10px; padding: 5px 10px; background: rgba(0,209,255,0.1); border-radius: 5px; border-left: 3px solid #00d1ff;">
                                        <span style="font-size: 0.85rem; color: #00d1ff;">🎯 <b>Market Outlook:</b> {art['reasoning']}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No recent news headers found. The prediction is based on technical trends.")

                    with tab_s:
                        st.subheader("Sector Performance (US Benchmarks)")
                        sectors = get_cached_sectors()
                        if sectors:
                            cols = st.columns(2)
                            for i, sec in enumerate(sectors):
                                with cols[i % 2]:
                                    color = "#4caf50" if sec['change'] > 0 else "#f44336"
                                    st.markdown(f"""
                                    <div class="sector-card" style="border-left-color: {color};">
                                        <div style="display: flex; justify-content: space-between;">
                                            <span>{sec['sector']}</span>
                                            <span style="color: {color}; font-weight: bold;">{sec['change']:.2f}%</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Could not load data for '{st.session_state.current_symbol}'. Please verify the ticker symbol.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    else:
        st.info("Search or enter a ticker in the sidebar to begin.")

with col_market:
    st.subheader("🔥 Top Movers")
    with st.spinner("Updating market..."):
        gainers, losers = get_cached_movers()
        
        if gainers:
            st.write("🚀 **Top Gainers**")
            for g in gainers[:5]:
                if st.button(f"{g['symbol']} (+{g['change']:.1f}%)", key=f"btn_gain_{g['symbol']}", use_container_width=True):
                    set_symbol(g['symbol'])
        
        st.write("---")
        
        if losers:
            st.write("📉 **Top Losers**")
            for l in losers[:5]:
                if st.button(f"{l['symbol']} ({l['change']:.1f}%)", key=f"btn_lose_{l['symbol']}", use_container_width=True):
                    set_symbol(l['symbol'])

# Sidebar footer
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Diagnostics"):
    st.write(f"V: {yf.__version__}")
    if st.session_state.current_symbol:
        st.write(f"Target: {st.session_state.current_symbol}")
        try:
            t = yf.Ticker(st.session_state.current_symbol)
            raw = getattr(t, 'news', [])
            st.write(f"Raw Items: {len(raw)}")
        except Exception as ex:
            st.write(f"Error: {ex}")
