# Development Log - Stock Predictor App

## Architecture Evolution
The app started as a simplified script but evolved into a robust dashboard with multi-layered data fetching.

### 1. Data Ingestion Challenges (yfinance)
- **Problem**: News headlines were missing even though they appeared in raw API responses.
- **Discovery**: `yfinance` changed its response schema. Metadata is now nested under a `content` key.
- **Solution**: Implemented a "Deeply Robust" parser in `data_fetcher.py` that probes multiple dictionary levels and handles dictionaries inside URLs (e.g., `clickThroughUrl`).

### 2. UI/UX Issues (Streamlit)
- **Problem**: Blank/White Screen of Death.
- **Causes**: 
    1. Blocking processes: Market-wide data fetching (movers) was blocking the main thread.
    2. CSS Conflicts: Overriding `.main` background-color sometimes prevents Streamlit from rendering the content DOM.
- **Fixes**: 
    - Moved Movers to a side column and added `st.cache_data`.
    - Used `st.status` for feedback.
    - Simplified CSS to use `[data-testid]` and `.stApp`.

### 3. Logic & State Errors
- **NameError**: Attempted to use `symbol` in a diagnostics expander before the `st.sidebar.text_input` call.
- **Fix**: Reordered code to ensure all UI variables are initialized before use.

### 4. Search & Relevance
- **Problem**: News fetched via `yfinance` for specific tickers often include general market news or news about other "hot" companies (e.g., AAPL news containing NVDA or general S&P 500 updates).
- **Discovery**: News endpoints are often keyword-broad. Without strict filtering, sentiment analysis becomes noisy and inaccurate.
- **Solution**: 
    1. Fetch full company name (e.g., "Apple Inc.") via `ticker.info`.
    2. Implement a strict relevance filter in `data_fetcher.py` that checks for both ticker and company name matches in the title/summary.
    3. Significantly reduces "noise" (e.g., filtering 10 articles down to 2 highly relevant ones for AAPL).

## Technical Specs
- **Backend**: Python, yfinance, vaderSentiment.
- **Frontend**: Streamlit 1.53.0+, Plotly.
- **Optimization**: Multi-TTL caching (600s for market data, 300s for ticker analysis).
