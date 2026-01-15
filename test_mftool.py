from mftool import Mftool
import pandas as pd

mf = Mftool()

# Get all available schemes
# schemes = mf.get_available_schemes()
# print(f"Total schemes: {len(schemes)}")

# Search for a scheme (e.g., HDFC Index)
search_results = mf.get_available_schemes("HDFC Index SENSEX")
print("Search Results for 'HDFC Index SENSEX':")
print(search_results)

if search_results:
    scheme_code = list(search_results.keys())[0]
    print(f"Fetching data for code: {scheme_code}")
    
    # Get historical NAV
    try:
        data = mf.get_scheme_historical_nav(scheme_code, as_dataframe=True)
        print("Historical Data (first 5 rows):")
        print(data.head())
    except Exception as e:
        print(f"Error fetching historical NAV: {e}")
