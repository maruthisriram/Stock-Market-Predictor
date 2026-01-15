from data_fetcher import StockFetcher
import pandas as pd

try:
    print("Initializing StockFetcher...")
    fetcher = StockFetcher()
    
    print("Testing _fetch_nifty50_tickers internally...")
    tickers = fetcher._fetch_nifty50_tickers()
    
    if tickers:
        print(f"Success! Fetched {len(tickers)} tickers.")
        print(f"Sample: {tickers[:5]}")
    else:
        print("Failed to fetch tickers.")
        
    print("\nTesting get_market_movers()...")
    gainers, losers = fetcher.get_market_movers()
    
    print(f"Top 5 Gainers:")
    for g in gainers:
        print(f"  {g['symbol']}: {g['change']}%")
        
    print(f"Top 5 Losers:")
    for l in losers:
        print(f"  {l['symbol']}: {l['change']}%")
        
except Exception as e:
    print(f"Verification Error: {e}")
