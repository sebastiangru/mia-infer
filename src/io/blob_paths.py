import os
from urllib.parse import urljoin

BLOB_BASE = os.getenv("BLOB_BASE", "https://miatradingdata.blob.core.windows.net/market-data")

def market_csv(tf: str, symbol: str) -> str:
    # e.g. .../market-data/15m/AMZN/AMZN_15m.csv
    return f"{BLOB_BASE}/market-data/{tf}/{symbol}/{symbol}_{tf}.csv"

def features_csv(tf: str, symbol: str) -> str:
    # e.g. .../features/15m/AMZN/AMZN_features_15m.csv
    return f"{BLOB_BASE}/features/{tf}/{symbol}/{symbol}_features_{tf}.csv"

def macro_csv(tf: str, macro_sym: str, features=False) -> str:
    # e.g. .../macro/1d/^VIX/^VIX_1d.csv or ^VIX_features_1d.csv
    suffix = "features_" if features else ""
    return f"{BLOB_BASE}/macro/{tf}/{macro_sym}/{macro_sym}_{suffix}{tf}.csv"
