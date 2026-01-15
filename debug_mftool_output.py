from mftool import Mftool
import json

mf = Mftool()
try:
    print("Fetching open ended equity scheme performance...")
    res = mf.get_open_ended_equity_scheme_performance(as_json=False)
    print(f"Type of result: {type(res)}")
    if isinstance(res, dict):
        print(f"Keys: {list(res.keys())[:5]}")
        first_key = list(res.keys())[0]
        print(f"First Item ({first_key}): {res[first_key]}")
    elif isinstance(res, list):
        print(f"Length of list: {len(res)}")
        if res:
            print(f"First item type: {type(res[0])}")
            print(f"First item: {res[0]}")
    else:
        print(res)
except Exception as e:
    print(f"Error: {e}")
