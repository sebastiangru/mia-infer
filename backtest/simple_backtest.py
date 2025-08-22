"""
Simple daily backtest (hard threshold, equal weight among active names)

Env vars (set via --env-file or -e):
- PRICE_BASE   : e.g. https://miatradingdata.blob.core.windows.net/market-data/market-data
- PRICE_SAS    : <SAS token without leading '?'>
- THRESH       : e.g. 0.565
- COST_PER_SIDE: per-side cost as decimal (e.g., 0.001 for 10 bps)

Inputs (from repo):
- configs/universe.csv        (column: symbol)
- preds/1d/lgbm/preds.csv     (columns: time, symbol, p_up [, yhat])

Outputs:
- reports/1d/equity_series.csv        (date,equity)
- reports/1d/daily_net_returns.csv    (date,ret)
- reports/1d/summary_stats.csv        (key,value)
"""
import os, io, json
import numpy as np
import pandas as pd
from azure.storage.blob import BlobClient


PRICE_BASE = os.getenv(
    "PRICE_BASE", "https://miatradingdata.blob.core.windows.net/market-data/market-data"
)
PRICE_SAS = os.getenv("PRICE_SAS", "")
THRESH = float(os.getenv("THRESH", "0.55"))
COST_PER_SIDE = float(os.getenv("COST_PER_SIDE", "0.001"))


def url_join(base: str, path: str, sas: str | None = None) -> str:
    base = base[:-1] if base.endswith("/") else base
    full = f"{base}/{path.lstrip('/')}"
    if sas:
        sep = "&" if "?" in full else "?"
        full = f"{full}{sep}{sas}"
    return full


def read_csv_url(url: str) -> pd.DataFrame:
    bc = BlobClient.from_blob_url(url)
    data = bc.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


def load_prices(symbols: list[str]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for s in symbols:
        url = url_join(PRICE_BASE, f"1d/{s}/{s}_1d.csv", PRICE_SAS or None)
        try:
            df = read_csv_url(url)
            ser = pd.Series(df["close"].values, index=pd.to_datetime(df["time"]), name=s)
            frames[s] = ser
        except Exception as e:
            print(f"[WARN] failed {s}: {e}")
    px = pd.DataFrame(frames).sort_index()
    return px


def portfolio_stats(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    if len(ret) == 0:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    ar = (1.0 + ret).prod() ** (252.0 / len(ret)) - 1.0
    vol = ret.std() * np.sqrt(252.0)
    sharpe = float(ar / vol) if vol > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"CAGR": float(ar), "Vol": float(vol), "Sharpe": float(sharpe), "MaxDD": float(dd)}


def holding_stats(pos: pd.DataFrame) -> float:
    holds: list[int] = []
    for col in pos.columns:
        x = pos[col].fillna(0).astype(int).values
        run = 0
        for i, v in enumerate(x):
            if v == 1:
                run += 1
            if v == 0 and run > 0:
                holds.append(run)
                run = 0
        if run > 0:
            holds.append(run)
    return float(np.mean(holds)) if holds else 0.0


if __name__ == "__main__":
    # 1) Load universe & predictions
    uni = pd.read_csv("configs/universe.csv")["symbol"].tolist()
    preds = pd.read_csv("preds/1d/lgbm/preds.csv", parse_dates=["time"])

    # 2) Load close prices from Blob
    prices = load_prices(uni)

    # 3) Align predictions to price index
    p_up = (
        preds.pivot(index="time", columns="symbol", values="p_up")
        .sort_index()
        .reindex(prices.index)
        .ffill()
    )

    # 4) Signals → equal-weight positions among active names
    positions = (p_up > THRESH).astype(int)
    active = positions.sum(axis=1).replace(0, np.nan)
    w = positions.div(active, axis=0).fillna(0.0)

    # 5) Returns (no implicit fill); attribute PnL with yesterday's weights
    ret = prices.pct_change(fill_method=None)
    w_prev = w.shift()
    gross = w_prev.mul(ret, fill_value=0.0).sum(axis=1)

    # 6) Turnover & trading costs
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_PER_SIDE).fillna(0.0)
    eq = (1.0 + net).cumprod()

    # 7) Stats
    stats = portfolio_stats(eq)
    stats["HitRate"] = float((net > 0).mean())
    stats["Exposure"] = float(w.sum(axis=1).mean())
    stats["TurnoverYr"] = float(turnover.sum() * (252.0 / len(turnover))) if len(turnover) else 0.0
    stats["AvgHoldDays"] = holding_stats(positions)

    print(stats)

    # 8) Save reports
    os.makedirs("reports/1d", exist_ok=True)
    pd.DataFrame({"date": eq.index, "equity": eq.values}).to_csv(
        "reports/1d/equity_series.csv", index=False
    )
    pd.DataFrame({"date": net.index, "ret": net.values}).to_csv(
        "reports/1d/daily_net_returns.csv", index=False
    )
    pd.Series(stats).to_csv("reports/1d/summary_stats.csv")
    print("Saved reports/1d/{equity_series.csv,daily_net_returns.csv,summary_stats.csv}")
