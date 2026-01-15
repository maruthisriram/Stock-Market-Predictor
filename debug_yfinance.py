import yfinance as yf
import pandas as pd

tickers = ['ADANIENT.NS', 'TCS.NS']
print("Downloading data...")
data = yf.download(tickers, period="2d", group_by='ticker', progress=False)

print("\nData Columns:")
print(data.columns)

print("\nData Head:")
print(data.head())

if isinstance(data.columns, pd.MultiIndex):
    print("\nMultiIndex detected.")
    print("Level 0:", data.columns.get_level_values(0).unique())
    try:
        print("Accessing TCS.NS:")
        print(data['TCS.NS'].head())
    except Exception as e:
        print(f"Error accessing TCS.NS: {e}")
else:
    print("\nSingle Index detected.")
