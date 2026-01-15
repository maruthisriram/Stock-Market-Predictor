from data_fetcher import MutualFundFetcher
import pandas as pd

try:
    print("Initializing MutualFundFetcher (New API Mode)...")
    fetcher = MutualFundFetcher()
    
    # 1. Test Master List
    print("\nFetching All Schemes...")
    schemes = fetcher.get_all_schemes()
    print(f"Total Schemes Found: {len(schemes)}")
    if schemes:
        print(f"Sample: {list(schemes.items())[:3]}")
    
    # 2. Test Historical Data (HDFC Index Fund)
    code = "120716"
    print(f"\nFetching Historical Data for {code}...")
    df = fetcher.fetch_historical_data(code, period="1mo")
    if df is not None:
        print("Success!")
        print(df.head())
        print(df.tail())
    else:
        print("Failed to fetch historical data")

    # 3. Test Trending Funds
    print("\nFetching Trending Funds...")
    trending = fetcher.get_trending_mfs()
    print("Trending Results:")
    for t in trending:
        print(f"  {t['name']} ({t['symbol']}): {t['change']}%")

except Exception as e:
    print(f"Verification Error: {e}")
