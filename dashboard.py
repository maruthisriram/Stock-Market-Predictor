import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_fetcher import StockFetcher, MutualFundFetcher
from predictor import StockPredictor
import yfinance as yf
from datetime import datetime

import requests_cache
from requests import Session
from requests_cache import CachedSession
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Page Config
st.set_page_config(page_title="FinSight Pro: Wealth Intelligence", layout="wide", page_icon="📈")

@st.cache_resource
def get_yfinance_session():
    """Create a smart session with caching and rate-limiting to avoid YFRateLimitError"""
    try:
        session = CachedSession(
            'yfinance_cache',
            use_cache_dir=True,
            cache_control=True,
            expire_after=datetime.timedelta(hours=1),
            backend='sqlite'
        )
    except:
        session = Session() # Fallback if caching fails (e.g. filesystem permissions)

    # Add headers to mimic a browser
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    # Retry logic for rate limits
    retry = Retry(connect=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Persistent State Initialization
if 'current_symbol' not in st.session_state: st.session_state.current_symbol = "AAPL"
if 'current_mf_code' not in st.session_state: st.session_state.current_mf_code = "120716"
if 'stock_search_val' not in st.session_state: st.session_state.stock_search_val = ""
if 'mf_search_val' not in st.session_state: st.session_state.mf_search_val = ""

def format_number(val):
    """Formats large numbers into human-readable strings (e.g., 1.2B, 1.2M, 45K)"""
    try:
        if val is None or pd.isna(val) or val == 0: return "N/A"
        val = float(val)
        if val >= 1_000_000_000: return f"{val / 1_000_000_000:.2f}B"
        elif val >= 1_000_000: return f"{val / 1_000_000:.2f}M"
        elif val >= 1_000: return f"{val / 1_000:.1f}K"
        return f"{val:,.0f}"
    except: return "N/A"

class DashboardApp:
    def __init__(self):
        self.stock_fetcher = StockFetcher()
        self.mf_fetcher = MutualFundFetcher()
        self.predictor = StockPredictor()

    def apply_styles(self):
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
            html, body, [class*="st-"] { font-family: 'Outfit', sans-serif; }
            .stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); }
            .main-header { font-size: 3rem; font-weight: 600; background: linear-gradient(90deg, #00d1ff, #00ff88); 
                          -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
            .metric-card { background: rgba(30,33,48,0.4); backdrop-filter: blur(15px); padding: 20px; 
                          border-radius: 15px; border: 1px solid rgba(255,255,255,0.05); transition: 0.3s; }
            .metric-card:hover { border-color: #00d1ff; transform: translateY(-2px); }
            .prediction-banner { padding: 1.5rem; border-radius: 1.2rem; text-align: center; margin: 1rem 0; }
            .up-gradient { background: linear-gradient(135deg, rgba(29,77,47,0.7), rgba(46,125,50,0.7)); border: 1px solid #4caf50; }
            .down-gradient { background: linear-gradient(135deg, rgba(77,29,29,0.7), rgba(198,40,40,0.7)); border: 1px solid #f44336; }
            .neutral-gradient { background: rgba(30,33,48,0.8); border: 1px solid #555; }
            .news-card { padding: 1.2rem; border-radius: 1rem; background: rgba(255,255,255,0.02); 
                        margin-bottom: 0.8rem; border: 1px solid rgba(0,209,255,0.1); }
        </style>
        """, unsafe_allow_html=True)

    def render_metrics(self, df, label="Price", unit="$", extra_metric=None, extra_label="Market Volume"):
        latest = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2] if len(df) > 1 else latest
        change = latest - prev
        pct = (change / prev) * 100 if prev != 0 else 0
        extra_val = extra_metric if extra_metric else format_number(df['Volume'].iloc[-1] if 'Volume' in df.columns else 0)
        
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f'<div class="metric-card"><p style="color:#888;margin:0;">Current {label}</p><h2 style="margin:0;">{unit}{latest:,.2f}</h2></div>', unsafe_allow_html=True)
        with m2: 
            color = "#4caf50" if change >= 0 else "#f44336"
            st.markdown(f'<div class="metric-card"><p style="color:#888;margin:0;">Day Change</p><h2 style="margin:0;color:{color};">{unit}{change:+,.2f} ({pct:+,.2f}%)</h2></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><p style="color:#888;margin:0;">{extra_label}</p><h2 style="margin:0;">{extra_val}</h2></div>', unsafe_allow_html=True)

    def render_diversification(self, code, fund_name):
        """Renders diversification pie chart for US or category fallback for India"""
        st.markdown("### 🧩 Portfolio Insights")
        weights = None
        
        # Scenario 1: US Fund (yfinance)
        # Scenario 1: US Fund (yfinance)
        if len(code) == 5 and code.endswith('X'):
            try:
                # Use our cached session to prevent rate limits
                session = get_yfinance_session()
                ticker = yf.Ticker(code, session=session)
                
                # yfinance propertry access triggering network request
                if hasattr(ticker, 'funds_data'):
                    # This call can trigger RateLimitError in cloud
                    fd = ticker.funds_data
                    if fd:
                        weights = fd.sector_weightings
            except Exception as e:
                # Log error but don't crash. Fallback to category weights below.
                print(f"Diversification fetch failed for {code}: {e}")
                weights = None
        
        # Scenario 2: Indian Fund (Category Fallback)
        if not weights:
            # Benchmark weights for major categories
            if "Index" in fund_name or "Blue" in fund_name:
                weights = {'Financials': 0.35, 'IT': 0.15, 'Energy': 0.12, 'Consumer': 0.10, 'Health': 0.08, 'Others': 0.20}
            elif "Small" in fund_name or "Mid" in fund_name:
                weights = {'Industrials': 0.25, 'Cons. Cyclical': 0.20, 'Health': 0.15, 'Tech': 0.10, 'Services': 0.10, 'Others': 0.20}
            elif "Debt" in fund_name:
                weights = {'Govt Bonds': 0.45, 'Corp Debt': 0.35, 'Cash': 0.15, 'TBills': 0.05}
            else:
                weights = {'Equity': 0.65, 'Debt': 0.25, 'Cash': 0.10}

        if weights:
            labels = [k.replace('_', ' ').capitalize() for k in weights.keys()]
            values = list(weights.values())
            fig = px.pie(names=labels, values=values, hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig.update_layout(template="plotly_dark", height=320, margin=dict(t=0, b=0, l=0, r=0),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Diversification data unavailable for this specific scheme.")

    def render_prediction(self, pred, reason):
        css = "up-gradient" if pred == "UP" else ("down-gradient" if pred == "DOWN" else "neutral-gradient")
        icon = "🚀" if pred == "UP" else ("📉" if pred == "DOWN" else "⚖️")
        st.markdown(f'<div class="prediction-banner {css}"><h2 style="margin:0;">{icon} {pred} FORECAST</h2><p style="margin:5px 0;opacity:0.9;">{reason}</p></div>', unsafe_allow_html=True)

    def render_news_feed(self, news):
        if not news:
            st.info("No active news headers for this asset.")
            return
        for art in news:
            color = "#4caf50" if art['sentiment'] > 0 else ("#f44336" if art['sentiment'] < 0 else "#888")
            sign = "+" if art['sentiment'] > 0 else ""
            st.markdown(f"""
            <div class="news-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <h4 style="margin:0; color:#00d1ff; flex:1; padding-right:10px;">{art['title']}</h4>
                    <div style="text-align:right; min-width:80px;">
                        <span style="color:{color}; font-weight:700; font-size:1.1rem;">{sign}{art['sentiment']:.2f}</span><br>
                        <small style="color:{color}; font-size:0.7rem; text-transform:uppercase;">{art['reasoning']}</small>
                    </div>
                </div>
                <p style="font-size:0.9rem; color:#bbb; margin:12px 0; line-height:1.4;">{art['summary']}</p>
                <div style="font-size:0.75rem; color:#666; display:flex; justify-content:space-between;">
                    <span>Source: {art['publisher']}</span>
                    <a href="{art['link']}" target="_blank" style="color:#00d1ff; text-decoration:none;">Full Story ↗</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_stock_module(self, period):
        # Google-style Autocomplete for Stocks
        st.markdown("### 🔍 Stock Discovery")
        
        # Pre-populate with popular stocks to make it feel like Google suggestions
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", 
                          "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"]
        
        if st.session_state.current_symbol not in popular_stocks:
            popular_stocks.insert(0, st.session_state.current_symbol)
            
        selected_stock = st.selectbox("Search Ticker (Autocomplete)", 
                                     popular_stocks, 
                                     index=popular_stocks.index(st.session_state.current_symbol) if st.session_state.current_symbol in popular_stocks else 0,
                                     placeholder="Type to search e.g. Reliance, Apple...",
                                     key="stock_search_main")
        
        if selected_stock and selected_stock != st.session_state.current_symbol:
            st.session_state.current_symbol = selected_stock
            st.rerun()
        
        symbol = st.session_state.current_symbol
        with st.spinner(f"Analysing {symbol}..."):
            df = self.stock_fetcher.fetch_historical_data(symbol, period)
            sentiment, news = self.stock_fetcher.fetch_news_sentiment(symbol)
        
        if df is not None:
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                pred, reason = self.predictor.predict(df, sentiment)
                self.render_metrics(df)
                self.render_prediction(pred, reason)
                
                t_tech, t_news = st.tabs(["📊 Technical Analysis", "📰 Intelligence Feed"])
                with t_tech:
                    fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#00d1ff', width=2.5)))
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=10, b=0, l=0, r=0), 
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, width='stretch')
                with t_news:
                    self.render_news_feed(news)
            with col2:
                st.markdown("### 🔥 Trending")
                
                @st.cache_data(ttl=3600)
                def get_cached_stocks():
                    movers, _ = self.stock_fetcher.get_market_movers()
                    return movers
                
                movers = get_cached_stocks()
                for m in movers:
                    if st.button(f"{m['symbol']} {m['change']:+}%", key=f"trend_{m['symbol']}", width="stretch"):
                        # Priority fix: Delete the search widget key to force it to reset on rerun
                        if "stock_search_main" in st.session_state:
                            del st.session_state["stock_search_main"]
                        st.session_state.current_symbol = m['symbol']
                        st.rerun()
                
                # Tiny holdings preview for stocks (top sectors)
                st.markdown("---")
                try:
                    info = yf.Ticker(symbol).info
                    sector = info.get('sector', 'Unknown')
                    industry = info.get('industry', 'Unknown')
                    st.markdown(f"**Sector**: {sector}")
                    st.markdown(f"**Industry**: {industry}")
                except: pass

        else: st.error(f"Data offline for {symbol}.")

    def render_mf_module(self, period):
        # Improved Search UI - Full schemes list for instant autocomplete
        st.markdown("### 🏦 Mutual Fund Explorer")
        
        # Load all codes for autocomplete
        if 'all_mf_options' not in st.session_state:
            with st.spinner("Loading Fund Database..."):
                all_schemes = self.mf_fetcher.mf.get_scheme_codes() if self.mf_fetcher.mf else {}
                st.session_state.all_mf_options = [f"{name} ({code})" for code, name in all_schemes.items()]
        
        # Add popular US funds if not present
        if not st.session_state.all_mf_options:
            st.session_state.all_mf_options = ["Vanguard 500 Index (VFIAX)", "Fidelity 500 (FXAIX)"]

        selected_fund = st.selectbox("Search Scheme (Autocomplete)", 
                                    st.session_state.all_mf_options, 
                                    index=None, 
                                    placeholder="Type to search e.g. HDFC, Index, SBI...",
                                    key="mf_autocomplete")
        
        if selected_fund:
            new_code = selected_fund.split('(')[-1].strip(')')
            if new_code != st.session_state.current_mf_code:
                st.session_state.current_mf_code = new_code
                st.rerun()

        code = st.session_state.current_mf_code
        with st.spinner(f"Analysing Fund {code}..."):
            df = self.mf_fetcher.fetch_historical_data(code, period)
            fund_name = code
            if self.mf_fetcher.mf and code.isdigit():
                try: 
                    # Handle potential 'restricted' API error gracefully
                    details = self.mf_fetcher.mf.get_scheme_details(code)
                    if details and isinstance(details, dict):
                        fund_name = details.get('scheme_name', code)
                except: pass
            sentiment, news = self.mf_fetcher.fetch_news_sentiment(code, company_name=fund_name)
        
        if df is not None:
            col1, col2 = st.columns([0.65, 0.35])
            with col1:
                is_us = len(code) == 5 and code.endswith('X')
                pred, reason = self.predictor.predict(df, sentiment)
                
                asset_label = "Equity" if "Equity" in fund_name else ("Debt" if "Debt" in fund_name else "Hybrid")
                self.render_metrics(df, label="NAV", unit="$" if is_us else "₹", extra_metric=asset_label, extra_label="Fund Category")
                self.render_prediction(pred, reason)
                
                t_nav, t_news = st.tabs(["📈 Price Performance", "📰 Fund Intel"])
                with t_nav:
                    fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#00ff88', width=2.5)))
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=10, b=0, l=0, r=0), 
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, width='stretch')
                with t_news:
                    self.render_news_feed(news)
            with col2:
                self.render_diversification(code, fund_name)
                
                st.markdown("---")
                st.markdown("### 🔥 Trending Funds")
                
                # Cache the real-time calculations to keep UI snappy
                @st.cache_data(ttl=3600)
                def get_cached_mfs():
                    return self.mf_fetcher.get_trending_mfs()
                
                trending_mfs = get_cached_mfs()
                for f in trending_mfs:
                    if st.button(f"{f['name']} ({f['change']:+2}%)", key=f"trend_mf_{f['symbol']}", width="stretch"):
                        # Priority fix: Delete the search widget key to force it to reset on rerun
                        if "mf_autocomplete" in st.session_state:
                            del st.session_state["mf_autocomplete"]
                        st.session_state.current_mf_code = f['symbol']
                        st.rerun()
        else: st.error(f"Fund data {code} is temporarily restricted.")

    def run(self):
        self.apply_styles()
        st.markdown("<h1 class='main-header'>FinSight Pro</h1>", unsafe_allow_html=True)
        
        # Tabs for Asset Selection
        t_stocks, t_mfs = st.tabs(["📈 Stock Market", "🏦 Mutual Funds"])
        
        with st.sidebar:
            st.markdown("### 📊 Horizon")
            period = st.selectbox("Analysis Window", ["1mo", "3mo", "6mo", "1y", "2y"], index=0)
            st.markdown("---")
            st.caption("AI-Powered Sentiment & Technical Forecasting Engine.")
        
        with t_stocks:
            self.render_stock_module(period)
            
        with t_mfs:
            self.render_mf_module(period)

@st.cache_resource
def get_app(): return DashboardApp()

if __name__ == "__main__":
    get_app().run()
