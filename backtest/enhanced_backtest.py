"""
Enhanced daily backtest with strategy & signal switches

Env vars:
- PRICE_BASE      : e.g. https://miatradingdata.blob.core.windows.net/market-data/market-data
- PRICE_SAS       : <SAS token without leading '?'>
- COST_PER_SIDE   : e.g., 0.001 (10 bps)
- THRESH          : threshold for hard/hybrid (default 0.55)
- STRAT           : hard | conf | hybrid   (default conf)
- SIGNAL          : p_up | yhat            (default p_up)
- SHIFT           : 0|1  (use signal at t or t-1; default 0)

Inputs:
- configs/universe.csv          (column: symbol)
- preds/1d/lgbm/preds.csv       (time,symbol,p_up[,yhat])

Outputs:
- reports/1d/enhanced_stats.json
- reports/1d/enhanced_equity.csv
- reports/1d/enhanced_daily_returns.csv
"""
# backtest/enhanced_backtest.py
import os, io, json
import numpy as np
import pandas as pd
from azure.storage.blob import BlobClient

PRICE_BASE = os.getenv(
    "PRICE_BASE", "https://miatradingdata.blob.core.windows.net/market-data/market-data"
)
PRICE_SAS = os.getenv("PRICE_SAS", "")
COST_PER_SIDE = float(os.getenv("COST_PER_SIDE", "0.001"))
THRESH = float(os.getenv("THRESH", "0.55"))
STRAT = os.getenv("STRAT", "conf")           # "hard" | "conf" | "hybrid"
SIGNAL = os.getenv("SIGNAL", "p_up")         # "p_up" | "yhat"
SHIFT = int(os.getenv("SHIFT", "0"))         # 0 or 1
TOPK = int(os.getenv("TOPK", "0"))           # 0 = disabled
PRED_FILE = os.getenv("PRED_FILE", "preds/1d/lgbm/preds.csv")
HYBRID_RANK = os.getenv("HYBRID_RANK", "yhat")  # "yhat" or "signal"

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
            frames[s] = pd.Series(df["close"].values, index=pd.to_datetime(df["time"]), name=s)
        except Exception as e:
            print(f"[WARN] failed {s}: {e}")
    return pd.DataFrame(frames).sort_index()

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
        for v in x:
            if v == 1: run += 1
            if v == 0 and run > 0:
                holds.append(run); run = 0
        if run > 0: holds.append(run)
    return float(np.mean(holds)) if holds else 0.0

if __name__ == "__main__":
    # 1) Universe & predictions
    uni = pd.read_csv("configs/universe.csv")["symbol"].tolist()
    preds = pd.read_csv(PRED_FILE, parse_dates=["time"])

    # 2) Prices
    prices = load_prices(uni)

    # 3) Build pivoted signals
    pv = preds.pivot(index="time", columns="symbol", values=SIGNAL).sort_index()
    if SHIFT: pv = pv.shift(1)
    pv = pv.reindex(prices.index).ffill()

    # Optional yhat matrix for hybrid ranking
    pv_yhat = None
    if STRAT == "hybrid" and HYBRID_RANK.lower() == "yhat":
        pv_yhat = preds.pivot(index="time", columns="symbol", values="yhat").sort_index()
        if SHIFT: pv_yhat = pv_yhat.shift(1)
        pv_yhat = pv_yhat.reindex(prices.index).ffill()

    # 4) Asset returns (keep NaNs), use lagged weights for attribution
    ret = prices.pct_change(fill_method=None)

    # 5) Weights per strategy
    if STRAT == "hard":
        positions = (pv > THRESH).astype(int)
        active = positions.sum(axis=1).replace(0, np.nan)
        w = positions.div(active, axis=0).fillna(0.0)

    elif STRAT == "conf":
        if SIGNAL == "p_up": conf = (pv - 0.5).clip(lower=0.0)
        else:                 conf = pv.clip(lower=0.0)
        if TOPK > 0:
            ranks = conf.fillna(float("-inf")).rank(axis=1, method="first", ascending=False)
            mask = (ranks <= TOPK) & (conf > 0)
            conf = conf.where(mask, 0.0)
        row_sum = conf.sum(axis=1).replace(0, np.nan)
        w = conf.div(row_sum, axis=0).fillna(0.0)
        positions = (w > 0).astype(int)

    elif STRAT == "hybrid":
        base = (pv > THRESH).astype(int)

        # rank by yhat if available, else by SIGNAL
        if HYBRID_RANK.lower() == "yhat" and pv_yhat is not None:
            conf_src = pv_yhat
            conf = (conf_src.sub(conf_src.min(axis=1), axis=0)).clip(lower=0.0) * base
        else:
            conf_src = pv
            conf = (conf_src - THRESH).clip(lower=0.0) * base

        if TOPK > 0:
            ranks = conf.fillna(float("-inf")).rank(axis=1, method="first", ascending=False)
            mask = (ranks <= TOPK) & (conf > 0)
            conf = conf.where(mask, 0.0)

        row_sum = conf.sum(axis=1).replace(0, np.nan)
        w = conf.div(row_sum, axis=0).fillna(0.0)
        positions = base

    else:
        raise SystemExit(f"Unknown STRAT={STRAT}")

    w_prev = w.shift()
    gross = w_prev.mul(ret, fill_value=0.0).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_PER_SIDE).fillna(0.0)
    eq = (1.0 + net).cumprod()

    stats = portfolio_stats(eq)
    stats.update({
        "HitRate": float((net > 0).mean()),
        "Exposure": float(w.sum(axis=1).mean()),
        "TurnoverYr": float(turnover.sum() * (252.0 / len(turnover))) if len(turnover) else 0.0,
        "AvgHoldDays": holding_stats(positions),
        "STRAT": STRAT, "SIGNAL": SIGNAL, "THRESH": THRESH, "SHIFT": SHIFT, "TOPK": TOPK,
        "HYBRID_RANK": HYBRID_RANK
    })
    print(json.dumps(stats, indent=2))

    os.makedirs("reports/1d", exist_ok=True)
    pd.Series(stats).to_json("reports/1d/enhanced_stats.json", indent=2)
    pd.DataFrame({"date": eq.index, "equity": eq.values}).to_csv(
        "reports/1d/enhanced_equity.csv", index=False
    )
    pd.DataFrame({"date": net.index, "ret": net.values}).to_csv(
        "reports/1d/enhanced_daily_returns.csv", index=False
    )
    print("Saved reports/1d/{enhanced_stats.json,enhanced_equity.csv,enhanced_daily_returns.csv}")
