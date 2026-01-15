import pandas as pd
import requests
from io import StringIO

try:
    print("Fetching Nifty 50 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        tables = pd.read_html(StringIO(response.text))
        print(f"Tables found: {len(tables)}")
        
        for i, t in enumerate(tables[:5]):
            print(f"Table {i} Columns: {t.columns.tolist()}")
            print(f"Table {i} Row 1: {t.iloc[0].values.tolist() if not t.empty else 'Empty'}")
            
    else:
        print("Failed to fetch.")

except Exception as e:
    print(f"Error: {e}")
