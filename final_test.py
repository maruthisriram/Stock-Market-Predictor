from data_fetcher import StockFetcher, MutualFundFetcher
import pandas as pd

def test_stock():
    print("Testing StockFetcher...")
    sf = StockFetcher()
    df = sf.fetch_historical_data("AAPL")
    if df is not None and not df.empty:
        print(f"Stock Data OK: {len(df)} rows")
    else:
        print("Stock Data FAILED")

def test_mf_india():
    print("Testing MutualFundFetcher (India)...")
    mf = MutualFundFetcher()
    df = mf.fetch_historical_data("120716") # HDFC Index
    if df is not None and not df.empty:
        print(f"India MF Data OK: {len(df)} rows")
    else:
        print("India MF Data FAILED")

def test_mf_us():
    print("Testing MutualFundFetcher (US)...")
    mf = MutualFundFetcher()
    df = mf.fetch_historical_data("VFIAX") # Vanguard 500
    if df is not None and not df.empty:
        print(f"US MF Data OK: {len(df)} rows")
    else:
        print("US MF Data FAILED")

if __name__ == "__main__":
    test_stock()
    test_mf_india()
    test_mf_us()
