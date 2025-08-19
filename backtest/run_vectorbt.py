import pandas as pd, numpy as np, vectorbt as vbt
from src.io.blob_paths import market_csv
from src.data.universe import load_universe

def load_close(tf, symbols):
    dfs={}
    for s in symbols:
        df = pd.read_csv(market_csv(tf, s), parse_dates=['time'])
        dfs[s] = df.set_index('time')['close']
    return pd.DataFrame(dfs).sort_index()

# You can store predictions to Blob or compute ad-hoc right after training.
# Here assume we have test-period p_up per symbol/time in a DataFrame p_up_df matching prices index.
def run(prices: pd.DataFrame, p_up_df: pd.DataFrame, thr=0.55):
    entries = p_up_df > thr
    exits = ~entries
    pf = vbt.Portfolio.from_signals(prices, entries=entries, exits=exits,
                                    fees=0.001, slippage=0.0005, init_cash=100000)
    print(pf.stats())
    pf.plot().write_html("reports/equity.html")

if __name__ == "__main__":
    symbols = load_universe()
    prices = load_close("1d", symbols)
    # TODO: build p_up_df from your saved predictions or compute inline
