import requests
import pandas as pd
import json

def test_mfapi():
    code = "120716" # HDFC Index Fund Nifty 50 Plan
    url = f"https://api.mfapi.in/mf/{code}"
    print(f"Testing URL: {url}")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Meta Data:")
            print(json.dumps(data.get('meta', {}), indent=2))
            
            nav_data = data.get('data', [])
            print(f"\nNAV Records Found: {len(nav_data)}")
            if nav_data:
                print("Sample Entry:", nav_data[0])
                
                # Verify DataFrame conversion
                df = pd.DataFrame(nav_data)
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df.set_index('date', inplace=True)
                df['nav'] = pd.to_numeric(df['nav'])
                print("\nDataFrame Head:")
                print(df.head())
        else:
            print("Failed to fetch data")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mfapi()
