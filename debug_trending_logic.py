from data_fetcher import MutualFundFetcher
import pandas as pd

print("Initializing fetcher...")
fetcher = MutualFundFetcher()

print("Calling get_trending_mfs code manually...")
try:
    if not fetcher.mf:
        print("MF Tool not initialized")
    else:
        perf_data = fetcher.mf.get_open_ended_equity_scheme_performance(as_json=False)
        print(f"Perf data type: {type(perf_data)}")
        
        all_funds = []
        if perf_data and isinstance(perf_data, dict):
            for category, funds in perf_data.items():
                if isinstance(funds, list):
                    all_funds.extend(funds)
        
        print(f"Total funds found: {len(all_funds)}")
        
        count = 0
        for i, item in enumerate(all_funds):
            if i >= 5: break # Just check first 5
            
            code = item.get('scheme_code')
            name = item.get('scheme_name')
            print(f"Checking Fund: {name} ({code})")
            
            try:
                # Trace fetch_historical_data logic inline or call it
                print(f"  Fetching history for {code}...")
                df = fetcher.fetch_historical_data(code, period="1mo")
                
                if df is None:
                    print("  -> fetch_historical_data returned None")
                elif df.empty:
                    print("  -> fetch_historical_data returned empty DF")
                else:
                    print(f"  -> Got {len(df)} rows. Last Date: {df.index[-1]}")
                    if len(df) >= 2:
                        latest = df['Close'].iloc[-1]
                        prev = df['Close'].iloc[-2]
                        change = ((latest - prev) / prev) * 100
                        print(f"  -> Change: {change}%")
                    else:
                        print("  -> Not enough rows for change calc")
            except Exception as e:
                print(f"  -> Error: {e}")

except Exception as e:
    print(f"Main Error: {e}")
