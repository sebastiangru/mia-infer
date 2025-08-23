# src/io/loaders.py
from __future__ import annotations

import os
import argparse
import json
from typing import Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import yaml


# =========================
# URL/Path helper utilities
# =========================
def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _join_blob_url(base: str, rel_path: str) -> str:
    """
    Join a relative blob path onto a base URL that already contains a SAS query.
    We must insert the path BEFORE the query string.
    """
    parts = urlsplit(base)
    base_path = parts.path.rstrip("/")
    rel_norm = rel_path.lstrip("/")
    new_path = f"{base_path}/{rel_norm}"
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


def _join_any(base: str, subpath: str) -> str:
    """Join subpath onto base which may be a local directory or a SAS URL."""
    if _is_url(base):
        return _join_blob_url(base, subpath)
    return os.path.join(base, subpath)


def _read_csv_any(path_or_url: str) -> pd.DataFrame:
    if _is_url(path_or_url):
        return pd.read_csv(path_or_url)
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Missing file: {path_or_url}")
    return pd.read_csv(path_or_url)


def _read_csv_with_time(path_or_url: str) -> pd.DataFrame:
    df = _read_csv_any(path_or_url)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


# =========================
# Loaders
# =========================
def load_universe(universe_csv: str, on_date: str | None = None) -> List[str]:
    """Universe CSV may be local (recommended). Supports simple or dated format."""
    if _is_url(universe_csv):
        u = pd.read_csv(universe_csv)
    else:
        if not os.path.exists(universe_csv):
            raise FileNotFoundError(f"Universe CSV not found: {universe_csv}")
        u = pd.read_csv(universe_csv)

    u.columns = [c.strip().lower() for c in u.columns]
    if "start_date" in u.columns:
        if on_date is None:
            symbols = sorted(u["symbol"].astype(str).unique().tolist())
        else:
            dt = pd.to_datetime(on_date)
            start = pd.to_datetime(u["start_date"])
            end = pd.to_datetime(u["end_date"]) if "end_date" in u.columns else pd.NaT
            mask = (start <= dt) & ((end.isna()) | (end >= dt))
            symbols = sorted(u.loc[mask, "symbol"].astype(str).unique().tolist())
    else:
        symbols = sorted(u["symbol"].astype(str).tolist())
    return symbols


def _resolve_base(dcfg: Dict, key: str) -> str:
    """
    Return a base that is either a local path or a full SAS URL prefix.
    data.yaml example:
      remote.enabled: true
      remote.blob_base_env: BLOB_BASE
      price_base:   "market-data/market-data/1d"
      feature_base: "market-data/features/1d"
      macro_base:   "market-data/macro/1d"
    """
    rel = dcfg[key]
    remote = dcfg.get("remote", {})
    if remote and remote.get("enabled"):
        env_name = remote.get("blob_base_env", "BLOB_BASE")
        blob_base = os.environ.get(env_name)
        if not blob_base:
            raise RuntimeError(f"Remote mode enabled but env var {env_name} is not set")
        if not _is_url(blob_base):
            raise RuntimeError(f"{env_name} must be an https URL with SAS; got: {blob_base}")
        return _join_blob_url(blob_base, rel)
    # Local
    return rel


def load_benchmark_prices(cfg: Dict, benchmark: str) -> pd.DataFrame:
    """
    Load benchmark close as columns: ['time', 'bench_close'].
    Preferred: raw prices under price_base/{sym}/{sym}_1d.csv (must contain 'close')
    Fallback: macro features under macro_base/{sym}/{sym}_features_1d.csv (must contain 'close')
    """
    price_base = _resolve_base(cfg, "price_base")
    macro_base = _resolve_base(cfg, "macro_base")

    # Preferred: raw prices
    p_price = _join_any(price_base, f"{benchmark}/{benchmark}_1d.csv")
    try:
        df = _read_csv_with_time(p_price).sort_values("time")
        if "close" not in df.columns:
            raise ValueError(f"{p_price} missing 'close' column")
        return df[["time", "close"]].rename(columns={"close": "bench_close"})
    except Exception:
        # Fallback: macro features
        p_macro = _join_any(macro_base, f"{benchmark}/{benchmark}_features_1d.csv")
        try:
            df = _read_csv_with_time(p_macro).sort_values("time")
            if "close" not in df.columns:
                raise ValueError(f"{p_macro} exists but has no 'close' column")
            return df[["time", "close"]].rename(columns={"close": "bench_close"})
        except Exception as e2:
            raise FileNotFoundError(
                "Benchmark not found. Tried:\n"
                f" - {p_price}\n"
                f" - {p_macro}\n"
                "Check configs/data.yaml price_base/macro_base and your BLOB_BASE."
            ) from e2


def load_equity_features_for_symbol(cfg: Dict, symbol: str) -> pd.DataFrame:
    """
    Load per-symbol **equity features** from feature_base/{sym}/{sym}_features_1d.csv
    (NOT from price_base).
    """
    feature_base = _resolve_base(cfg, "feature_base")
    p = _join_any(feature_base, f"{symbol}/{symbol}_features_1d.csv")
    try:
        df = _read_csv_with_time(p).sort_values("time")
    except Exception as e:
        raise FileNotFoundError(f"Equity features not found for {symbol}: {p}") from e
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    else:
        df["symbol"] = df["symbol"].fillna(symbol)
    return df


def load_macro_panel(cfg: Dict, macro_symbols: List[str], ffill_only: bool = True) -> pd.DataFrame:
    """
    Load all macro features and outer-join by time.
    Columns are prefixed with '{sym}__'.
    """
    if not macro_symbols:
        return pd.DataFrame()

    macro_base = _resolve_base(cfg, "macro_base")
    out = None
    for sym in macro_symbols:
        p = _join_any(macro_base, f"{sym}/{sym}_features_1d.csv")
        df = _read_csv_with_time(p).sort_values("time")
        keep = ["time"] + [c for c in df.columns if c not in ("symbol", "timeframe", "time")]
        df = df[keep]
        rename = {c: f"{sym}__{c}" for c in df.columns if c != "time"}
        df = df.rename(columns=rename)
        out = df if out is None else out.merge(df, on="time", how="outer")

    out = out.sort_values("time")
    if ffill_only:
        out = out.ffill()
    return out


def build_labels(eq_df: pd.DataFrame, bench_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    y_cont = log(stock_{t+h}/stock_t) - log(bench_{t+h}/bench_t)
    y_bin  = 1[y_cont > 0]
    """
    df = eq_df.merge(bench_df, on="time", how="left")

    if "bench_close" not in df.columns:
        raise ValueError("Benchmark column 'bench_close' not found after merge. "
                         "Check benchmark loader and 'time' alignment.")

    stock = df["close"].astype(float)
    bench = df["bench_close"].astype(float)

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
    with open(data_cfg_path, "r") as f:
        dcfg = yaml.safe_load(f)

    universe_csv = dcfg["universe_csv"]
    benchmark = dcfg.get("benchmark", "SPY")

    symbols = load_universe(universe_csv, on_date=universe_on_date)
    bench_prices = load_benchmark_prices(dcfg, benchmark)

    macro_symbols = model_cfg.get("macro_symbols", [])
    macro_ffill = bool(dcfg.get("macro_ffill", True))
    macro_panel = load_macro_panel(dcfg, macro_symbols, ffill_only=macro_ffill) if macro_symbols else pd.DataFrame()

    missing_symbols: List[str] = []
    frames: List[pd.DataFrame] = []

    for sym in symbols:
        try:
            eq = load_equity_features_for_symbol(dcfg, sym)
        except FileNotFoundError as e:
            missing_symbols.append(f"{sym} :: {e}")
            continue

        if not macro_panel.empty:
            eq = eq.merge(macro_panel, on="time", how="left")

        if dcfg.get("drop_na_after_join", True):
            eq = eq.dropna()

        horizon = int(model_cfg.get("horizon_days", 5))
        eq = build_labels(eq, bench_prices, horizon_days=horizon)
        eq = eq.dropna(subset=["y_cont", "y_bin"])
        frames.append(eq)

    if not frames:
        raise RuntimeError(
            "No equity frames loaded. All symbols missing or dropped. "
            "Missing details:\n - " + "\n - ".join(missing_symbols)
        )

    panel = pd.concat(frames, axis=0, ignore_index=True).sort_values(["time", "symbol"])

    # Save a small debug report listing missing symbols
    os.makedirs("runs/debug", exist_ok=True)
    with open("runs/debug/missing_symbols.txt", "w") as f:
        for line in missing_symbols:
            f.write(line + "\n")

    # Return only symbols that actually loaded
    loaded_symbols = sorted(panel["symbol"].unique().tolist())
    return panel, loaded_symbols


def load_panel_for_training(
    data_cfg_path: str,
    model_cfg_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    with open(model_cfg_path, "r") as f:
        mcfg = yaml.safe_load(f)

    panel, symbols = assemble_panel(data_cfg_path, mcfg, universe_on_date=None)

    non_features = {
        "symbol", "time", "timeframe", "open", "high", "low", "close", "volume",
        "y_bin", "y_cont"
    }
    feature_cols = [c for c in panel.columns if c not in non_features]

    X = panel[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_bin = panel["y_bin"].to_numpy(dtype=int)
    y_cont = panel["y_cont"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["time"]).to_numpy()

    if np.isnan(X).any():
        raise ValueError("Found NaNs in X after preparation. Check feature generation & join policy.")
    return X, y_bin, y_cont, dates, feature_cols, symbols


# =========================
# Simple audit CLI
# =========================
def _audit(panel: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    by_symbol = panel.groupby("symbol").agg(
        rows=("time", "count"),
        start=("time", "min"),
        end=("time", "max"),
        pos_rate=("y_bin", "mean"),
    ).reset_index()
    class_balance = panel["y_bin"].value_counts(normalize=True).rename_axis("class").reset_index(name="share")
    by_symbol.to_csv(os.path.join(out_dir, "by_symbol.csv"), index=False)
    class_balance.to_csv(os.path.join(out_dir, "class_balance.csv"), index=False)
    summary = {
        "symbols": by_symbol.to_dict(orient="records"),
        "class_balance": class_balance.to_dict(orient="records"),
        "n_rows": int(len(panel)),
        "n_features": int(len([c for c in panel.columns if c not in {"symbol","time","y_bin","y_cont","open","high","low","close","volume","timeframe"}])),
        "time_range": {"start": str(panel["time"].min()), "end": str(panel["time"].max())},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_xgb.yaml")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--out", default="runs/audit")
    args = parser.parse_args()

    X, yb, yc, dates, feats, syms = load_panel_for_training(args.data_config, args.model_config)
    with open(args.model_config, "r") as f:
        mcfg = yaml.safe_load(f)
    panel, _ = assemble_panel(args.data_config, mcfg)

    if args.audit:
        _audit(panel, args.out)
    else:
        print(f"Loaded: X={X.shape}, y_bin={yb.shape}, y_cont={yc.shape}, feats={len(feats)}, symbols={len(syms)}")
