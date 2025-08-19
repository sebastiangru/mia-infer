import pandas as pd
from pathlib import Path

def load_universe(path: str = "configs/universe.csv") -> list[str]:
    return pd.read_csv(path)['symbol'].dropna().astype(str).tolist()
