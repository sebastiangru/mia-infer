# src/io/loaders.py
from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df

def load_universe(universe_csv: str, on_date: str | None = None) -> List[str]:
    """
    Universe CSV supports either:
      - simple list: symbol
      - dated: symbol,start_date,end_date (inclusive)
    """
    u = pd.read_csv(universe_csv)
    if "start_date" in u.columns:
        if on_date is None:
            # Use all symbols active at any time
            symbols = sorted(u["symbol"].unique().tolist())
        else:
            dt = pd.to_datetime(on_date)
            mask = (pd.to_datetime(u["start_date"]) <= dt) & ((u.get("end_date").isna()) | (pd.to_datetime(u.get("end_date")) >= dt))
            symbols = sorted(u.loc[mask, "symbol"].unique().tolist())
    else:
        symbols = sorted(u["symbol"].astype(str).tolist())
    return symbols

def load_macro_panel(cfg: Dict, macro_symbols: List[str]) -> pd.DataFrame:
    """
    Load and horizontally join macro features by date.
    Forward-fill only; no backfill.
    """
    base = cfg["macro_base"]
    out = None
    for sym in macro_symbols:
        p = os.path.join(base, sym, f"{sym}_features_1d.csv")
        df = _read_csv(p).sort_values("time")
        # keep only time + feature cols
        keep = [c for c in df.columns if c not in ("symbol", "timeframe")]
        df = df[keep]
        # prefix macro columns except 'time'
        rename = {c: f"{sym}__{c}" for c in df.columns if c != "time"}
        df = df.rename(columns=rename)
        out = df if out is None else out.merge(df, on="time", how="outer")
    # forward-fill only, then keep as-of features
    out = out.sort_values("time").ffill()
    return out

def load_equity_features_for_symbol(cfg: Dict, symbol: str) -> pd.DataFrame:
    base = cfg["price_base"]
    p = os.path.join(base, symbol, f"{symbol}_features_1d.csv")
    df = _read_csv(p).sort_values("time")
    return df

def load_benchmark_prices(cfg: Dict, benchmark: str) -> pd.DataFrame:
    base = cfg["price_base"]
    p = os.path.join(base, benchmark, f"{benchmark}_1d.csv")
    df = _read_csv(p).sort_values("time")
    if "close" not in df.columns:
        raise ValueError(f"{p} missing 'close'")
    return df[["time", "close"]].rename(columns={"close": f"{benchmark}_close"})

def build_labels(eq_df: pd.DataFrame, bench_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Create y_cont and y_bin on the equity frame using the benchmark series.
    y_cont = log(stock_{t+h}/stock_t) - log(bench_{t+h}/bench_t)
    y_bin   = 1[y_cont > 0]
    """
    df = eq_df.merge(bench_df, on="time", how="left")
    # use stock close from eq_df (must exist)
    stock = df["close"]
    bench = df[bench_df.columns[-1]]

    # future returns (label ONLY): shift(-h). No use in features.
    logret_stock = np.log(stock.shift(-horizon_days) / stock)
    logret_bench = np.log(bench.shift(-horizon_days) / bench)

    y_cont = logret_stock - logret_bench
    y_bin = (y_cont > 0).astype(int)

    df["y_cont"] = y_cont
    df["y_bin"] = y_bin
    return df

def assemble_panel(
    data_cfg_path: str,
    model_cfg: Dict,
    universe_on_date: str | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns a concatenated panel across symbols with features + labels.
    """
    with open(data_cfg_path, "r") as f:
        import yaml
        dcfg = yaml.safe_load(f)

    universe_csv = dcfg["universe_csv"]
    symbols = load_universe(universe_csv, on_date=universe_on_date)

    benchmark = dcfg.get("benchmark", "SPY")
    bench_prices = load_benchmark_prices(dcfg, benchmark)

    # Macro panel
    # If you want to subset, put macro_symbols in model_cfg; otherwise infer from folder listing or a config list
    macro_symbols = model_cfg.get("macro_symbols", [])
    macro_panel = load_macro_panel(dcfg, macro_symbols) if macro_symbols else None

    frames = []
    for sym in symbols:
        eq = load_equity_features_for_symbol(dcfg, sym)
        # left-join macro panel by date, ffill already done
        if macro_panel is not None:
            eq = eq.merge(macro_panel, on="time", how="left")
        # Drop rows with any NA after join (your current rule)
        if dcfg.get("drop_na_after_join", True):
            eq = eq.dropna()

        # Build labels
        eq = build_labels(eq, bench_prices, horizon_days=model_cfg.get("horizon_days", 5))
        # Keep only rows with valid labels
        eq = eq.dropna(subset=["y_cont", "y_bin"])

        # Add symbol column if not present
        if "symbol" not in eq.columns:
            eq["symbol"] = sym
        else:
            eq["symbol"] = eq["symbol"].fillna(sym)

        frames.append(eq)

    panel = pd.concat(frames, axis=0, ignore_index=True).sort_values(["time","symbol"])
    return panel, symbols

def load_panel_for_training(
    data_cfg_path: str,
    model_cfg_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Returns:
      X (np.ndarray), y_bin, y_cont, dates, feature_names, symbols
    """
    import yaml
    with open(model_cfg_path, "r") as f:
        mcfg = yaml.safe_load(f)

    panel, symbols = assemble_panel(data_cfg_path, mcfg)

    # Identify non-feature columns
    non_features = {
        "symbol","time","timeframe","open","high","low","close","volume","y_bin","y_cont"
    }
    feature_cols = [c for c in panel.columns if c not in non_features]

    X = panel[feature_cols].to_numpy(dtype=float)
    y_bin = panel["y_bin"].to_numpy(dtype=int)
    y_cont = panel["y_cont"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["time"]).to_numpy()

    return X, y_bin, y_cont, dates, feature_cols, symbols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_xgb.yaml")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    X, yb, yc, dates, feats, syms = load_panel_for_training(args.data-config, args.model-config)  # noqa
