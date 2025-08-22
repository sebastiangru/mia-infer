# tools/fetch_sectors.py
import os, requests, pandas as pd
API_KEY = os.environ["uUAwMF8XNCgSGpvZWV8zszNaWVd6caDD"]
symbols = [s.strip() for s in open("configs/universe.csv").read().splitlines() if s.strip() and s.strip()!="symbol"][0:]
rows=[]
for s in symbols:
    url = f"https://financialmodelingprep.com/api/v3/profile/{s}?apikey={API_KEY}"
    r = requests.get(url, timeout=10).json()
    if isinstance(r, list) and r:
        d = r[0]
        rows.append({"symbol": s, "sector": d.get("sector"), "industry": d.get("industry")})
pd.DataFrame(rows).to_csv("metadata/sectors.csv", index=False)
print("Saved metadata/sectors.csv")
