# src/trainers/train_daily_lgbm.py
import os
import io
import sys
import json
import math
import time
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

from urllib.parse import urlsplit, urlunsplit, quote, unquote

# --------------------------- Logging helpers ---------------------------

def log(msg: str):
    print(msg, flush=True)

# --------------------------- URL & path utils --------------------------

def join_url(*parts: str) -> str:
    parts = [str(p).strip("/").replace("\\", "/") for p in parts if p is not None and str(p)]
    if not parts:
        return ""
    return parts[0] + ("/" + "/".join(parts[1:]) if len(parts) > 1 else "")

def sanitize_symbol_for_path(sym: str) -> str:
    out = []
    for ch in sym:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def add_sas(url: str, sas: Optional[str]) -> str:
    if not sas:
        return url
    return url if "?" in url else f"{url}?{sas.lstrip('?')}"

def encode_url_path_only(url: str) -> str:
    """Percent-encode ONLY the path component (keep query/SAS intact)."""
    parts = urlsplit(url)
    path_dec = unquote(parts.path)
    path_enc = quote(path_dec, safe="/-_.~")
    return urlunsplit((parts.scheme, parts.netloc, path_enc, parts.query, parts.fragment))

# --------------------------- Azure / HTTP I/O --------------------------

def azure_or_http_read_csv(url: str, retries: int = 2, backoff: float = 0.8) -> pd.DataFrame:
    last_exc = None
    enc_url = encode_url_path_only(url)
    for attempt in range(retries + 1):
        try:
            from azure.storage.blob import BlobClient
            bc = BlobClient.from_blob_url(enc_url)
            data = bc.download_blob().readall()
            return pd.read_csv(io.BytesIO(data))
        except Exception as e_sdk:
            last_exc = e_sdk
            try:
                return pd.read_csv(enc_url)
            except Exception:
                last_exc = e_sdk
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    raise last_exc

# --------------------------- Small stats helpers -----------------------

def forward_log_return(series: pd.Series, horizon: int) -> pd.Series:
    return np.log(series.shift(-horizon)) - np.log(series)

def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return xr.corr(yr)

def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped, keep = [], []
    for c in df.columns:
        s = df[c]
        if s.dtype.kind in "bifc" and s.notna().any() and s.nunique(dropna=True) > 1:
            keep.append(c)
        else:
            dropped.append(c)
    return df[keep], dropped

def drop_low_variance(df: pd.DataFrame, eps: float = 1e-12) -> Tuple[pd.DataFrame, List[str]]:
    keep, dropped = [], []
    for c in df.columns:
        if df[c].dtype.kind in "bifc":
            if float(df[c].std(skipna=True)) > eps:
                keep.append(c)
            else:
                dropped.append(c)
        else:
            dropped.append(c)
    return df[keep], dropped

# --------------------------- Frame wrangling ---------------------------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["time", "date", "Date", "timestamp"]:
        if c in df.columns:
            df = df.copy()
            df[c] = pd.to_datetime(df[c], utc=False, errors="coerce")
            df = df.set_index(c).sort_index()
            return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    raise ValueError("No date/time column found; columns: " + ", ".join(df.columns))

def pick_price_column(df: pd.DataFrame) -> str:
    for c in ["close", "adj_close", "Adj Close", "last", "price"]:
        if c in df.columns:
            return c
    cands = [c for c in df.columns if "close" in c.lower()]
    if cands:
        return cands[0]
    raise ValueError("No close/price column found.")

# --------------------------- CSV loader (asset or macro) ---------------

def load_symbol_csv(
    sym: str,
    base: str,
    sas_token: Optional[str],
    prefer_unsanitized: bool = False,
    features: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Load one CSV for a symbol-like name (works for both assets and macros).
    - When prefer_unsanitized=True, try the original folder/file (e.g., '^VIX') first.
    - Otherwise, try the sanitized folder/file first (e.g., 'VIX').
    - When features=True, filename is '<folder>_features_1d.csv'; else '<folder>_1d.csv'.
    """
    if not base:
        base = ""

    folder_uns = sym
    folder_san = sanitize_symbol_for_path(sym)

    order = [folder_uns, folder_san] if prefer_unsanitized else [folder_san, folder_uns]
    tried_err = None

    for folder in order:
        fname = f"{folder}_features_1d.csv" if features else f"{folder}_1d.csv"

        # cloud
        if base:
            url = add_sas(join_url(base, "1d", folder, fname), sas_token)
            try:
                df = azure_or_http_read_csv(url)
                return df, folder
            except Exception as e:
                tried_err = f"[read failed] {url} -> {e}"

        # local fallback
        local = os.path.join("data", "1d", folder, fname)
        if os.path.exists(local):
            try:
                df = pd.read_csv(local)
                return df, folder
            except Exception as e:
                tried_err = f"[read failed] {local} -> {e}"

    raise FileNotFoundError(tried_err or f"Could not load {sym} ({'features' if features else 'price'})")

# --------------------------- Macro panel -------------------------------

# Your updated macro list (includes special chars)
DEFAULT_MACROS = [
    "^TNX","HYG","LQD","DXY","EURUSD=X","USDJPY=X","GBPUSD=X",
    "CL=F","BZ=F","GC=F","HG=F","SPY","QQQ","IWM","^STOXX50E","EZU","^VIX","MOVE"
]

def _fallback_macro_from_price(df_raw: pd.DataFrame, folder: str) -> pd.DataFrame:
    """If macro features file is missing, build a small set from price."""
    df_raw = ensure_datetime_index(df_raw)
    price_col = pick_price_column(df_raw)
    m = pd.DataFrame(index=df_raw.index)
    close = df_raw[price_col].astype(float)
    ret1 = close.pct_change()
    ret5 = close.pct_change(5)
    ret21 = close.pct_change(21)
    z20 = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-12)
    vol21 = ret1.rolling(21).std()

    out = pd.DataFrame({
        f"{folder}__close": close,
        f"{folder}__ret1": ret1,
        f"{folder}__ret5": ret5,
        f"{folder}__ret21": ret21,
        f"{folder}__z20": z20,
        f"{folder}__vol21": vol21,
    }, index=m.index)
    return out

def load_macro_panel(macros: List[str], base: Optional[str], sas_token: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Loads macro features from a base like:
      .../market-data/market-data/marco
    preferring:
      marco/1d/<SYMBOL>/<SYMBOL>_features_1d.csv
    else falling back to price:
      marco/1d/<SYMBOL>/<SYMBOL>_1d.csv  -> compute close, ret1, ret5, ret21, z20, vol21
    """
    if not macros:
        return None
    frames = []
    for msym in macros:
        try:
            # Prefer macro features
            try:
                df_feat, folder = load_symbol_csv(msym, base or "", sas_token, prefer_unsanitized=True, features=True)
                df_feat = ensure_datetime_index(df_feat)
                feat_num = df_feat.select_dtypes(include=[np.number])
                # drop any leak-like columns (paranoia)
                leak_like = ("target", "future", "label", "y_", "_target")
                feat_num = feat_num.drop(columns=[c for c in feat_num.columns if any(k in c.lower() for k in leak_like)],
                                         errors="ignore")
                feat_num = feat_num.add_prefix(f"{folder}__")
                frames.append(feat_num)
            except Exception as fe:
                # Fall back to raw price -> minimal features
                df_raw, folder = load_symbol_csv(msym, base or "", sas_token, prefer_unsanitized=True, features=False)
                frames.append(_fallback_macro_from_price(df_raw, folder))
        except Exception as e:
            log(f"[WARN] macro {msym} skipped: {e}")

    if not frames:
        return None

    M = pd.concat(frames, axis=1).sort_index()
    M = M.ffill().bfill()
    M2, dropped = drop_constant_columns(M)
    if dropped:
        log(f"[CLEAN] Dropping constant macro cols: {dropped}")
    return M2

# --------------------------- Per-symbol features -----------------------

def load_equity_df_prefer_features(sym: str, base: str, sas_token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Try <SYM>_features_1d.csv first (contains price + engineered features),
    else fall back to <SYM>_1d.csv.
    """
    try:
        df_feat, folder = load_symbol_csv(sym, base, sas_token, prefer_unsanitized=False, features=True)
        return df_feat, folder
    except Exception:
        df_raw, folder = load_symbol_csv(sym, base, sas_token, prefer_unsanitized=False, features=False)
        return df_raw, folder

def build_symbol_frame(df_in: pd.DataFrame,
                       symbol: str,
                       horizon: int,
                       target_mode: str,
                       bench_close: Optional[pd.Series]) -> pd.DataFrame:
    df = ensure_datetime_index(df_in).copy()
    price_col = pick_price_column(df)
    df["__close__"] = df[price_col].astype(float)

    # forward log return (continuous)
    df["y_cont"] = forward_log_return(df["__close__"], horizon)

    if target_mode == "excess":
        if bench_close is None:
            raise ValueError("target_mode=excess but bench_close is None")
        bench = bench_close.reindex(df.index).ffill()
        df["y_cont"] = df["y_cont"] - forward_log_return(bench, horizon)

    df["y_bin"] = (df["y_cont"] > 0.0).astype(int)

    # numeric features (drop obvious leaks)
    drop_now = {"y_cont", "y_bin", "__close__"}
    leak_like = {"target", "future", "label", "y_", "_target"}

    X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_now if c in df], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if any(k in c.lower() for k in leak_like)], errors="ignore")

    X["y_cont"] = df["y_cont"]
    X["y_bin"] = df["y_bin"]
    X = X.dropna()

    X["symbol__"] = symbol
    return X

def attach_macro(X: pd.DataFrame, macro_panel: Optional[pd.DataFrame]) -> pd.DataFrame:
    if macro_panel is None or macro_panel.empty:
        return X
    out = X.join(macro_panel, how="left")
    macro_cols = [c for c in macro_panel.columns if c in out.columns]
    out[macro_cols] = out[macro_cols].ffill().bfill()
    return out

# --------------------------- Universe & folds --------------------------

def load_universe(symbols_arg: Optional[str]) -> List[str]:
    if symbols_arg:
        return [s.strip() for s in symbols_arg.split(",") if s.strip()]
    uni_txt = os.path.join("data", "universe_1d.txt")
    if os.path.exists(uni_txt):
        with open(uni_txt, "r") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return ["AMZN","GD","NVDA","GOOGL","AMD","ASML","TSM","MRVL","SMCI",
            "MRNA","CRSP","BNTX","ARGX","EXEL","SAGE","SPY"]

def make_time_folds(dates: pd.DatetimeIndex, n_splits: int, embargo_days: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.array(sorted(pd.Index(dates.unique())))
    if n_splits < 2 or len(uniq) < n_splits:
        return [(np.arange(len(uniq)), np.array([], dtype=int))]
    fold_sizes = np.full(n_splits, len(uniq) // n_splits, dtype=int)
    fold_sizes[:len(uniq) % n_splits] += 1
    edges = np.cumsum(fold_sizes)

    folds = []
    start = 0
    for end in edges:
        va_dates = uniq[start:end]
        start = end
        if len(va_dates) == 0:
            continue
        va_min, va_max = va_dates[0], va_dates[-1]
        emb_lo = va_min - pd.Timedelta(days=embargo_days)
        emb_hi = va_max + pd.Timedelta(days=embargo_days)
        tr_dates = uniq[(uniq < emb_lo) | (uniq > emb_hi)]

        tr_idx = np.where(pd.Index(dates).isin(tr_dates))[0]
        va_idx = np.where(pd.Index(dates).isin(va_dates))[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append((np.unique(tr_idx), np.unique(va_idx)))
    return folds

# --------------------------- Benchmark (excess) ------------------------

def load_benchmark_close(bench_symbol: str, price_base: str, sas_token: Optional[str]) -> pd.Series:
    df_bench, _ = load_symbol_csv(bench_symbol, price_base, sas_token, prefer_unsanitized=False, features=False)
    df_bench = ensure_datetime_index(df_bench)
    price_col = pick_price_column(df_bench)
    return df_bench[price_col].astype(float).rename("bench_close")

# --------------------------- Preds assembly ---------------------------

def to_preds_df(index: pd.MultiIndex, proba: np.ndarray) -> pd.DataFrame:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("to_preds_df expects a MultiIndex [symbol, date]")
    syms = index.get_level_values(0)
    dates = index.get_level_values(1)
    if len(dates) != len(proba):
        raise ValueError("Index/proba length mismatch")
    return pd.DataFrame({"date": dates.values, "symbol": syms.values, "p_up": proba.astype(float)})

# --------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data bases & auth
    ap.add_argument("--price_base", type=str, default=os.environ.get("PRICE_BASE", "").strip())
    ap.add_argument("--macro_base", type=str, default=os.environ.get("MACRO_BASE", "").strip(),
                    help="Root for macros (e.g., .../market-data/market-data/marco)")
    ap.add_argument("--azure_sas_token", type=str, default=os.environ.get("AZURE_SAS_TOKEN"),
                    help="SAS token WITHOUT leading '?'. Will be appended to blob URLs.")
    # Universe / macro
    ap.add_argument("--symbols", type=str, default=os.environ.get("SYMBOLS_1D", ""))
    ap.add_argument("--use_macro", type=int, default=int(os.environ.get("USE_MACRO", "1")))
    ap.add_argument("--macros", type=str, default=os.environ.get("MACROS", ""))
    # Target / CV
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--target_mode", type=str, choices=["absolute","excess"], default="excess")
    ap.add_argument("--bench_symbol", type=str, default=os.environ.get("BENCH_SYMBOL", "SPY"))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--embargo_days", type=int, default=5)
    ap.add_argument("--min_rows_per_symbol", type=int, default=800)
    ap.add_argument("--scale_features", type=int, default=1)
    # Feature pruning
    ap.add_argument("--prune_low_variance", type=int, default=1)
    # Output
    ap.add_argument("--preds_out", type=str, default="preds/1d/lgbm/preds.csv")
    ap.add_argument("--preds_oof_out", type=str, default="preds/1d/lgbm/preds_oof.csv")
    ap.add_argument("--out_dir", type=str, default="models/lgbm_daily")
    # LGBM params
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l2", type=float, default=5.0)
    ap.add_argument("--class_weight", type=str, default=None, help="e.g. 'balanced'")
    # Logging silencer
    ap.add_argument("--quiet_lgbm", type=int, default=1, help="Suppress LightGBM eval logs")
    args = ap.parse_args()

    # Dirs
    os.makedirs(os.path.dirname(args.preds_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.preds_oof_out), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Universe
    symbols = load_universe(args.symbols)
    symbols = list(dict.fromkeys(symbols))
    if args.target_mode == "excess" and args.bench_symbol in symbols:
        symbols = [s for s in symbols if s != args.bench_symbol]
        log(f"[CLEAN] Removed benchmark '{args.bench_symbol}' from universe for excess-return target.")

    # Macro base & panel
    macro_list = [s.strip() for s in args.macros.split(",") if s.strip()] if args.macros else DEFAULT_MACROS
    macro_base = args.macro_base if args.macro_base else join_url(args.price_base, "marco")
    macro_panel = load_macro_panel(macro_list if args.use_macro else [], macro_base, args.azure_sas_token)

    # Benchmark (excess target)
    bench_close = None
    if args.target_mode == "excess":
        log("[BENCH] benchmark from PRICE_BASE")
        bench_close = load_benchmark_close(args.bench_symbol, args.price_base, args.azure_sas_token).sort_index()

    # Build symbol frames (prefer features, else price)
    frames = []
    loaded = skipped = 0
    for sym in symbols:
        try:
            df_in, _ = load_equity_df_prefer_features(sym, args.price_base, args.azure_sas_token)
            sf = build_symbol_frame(df_in, sym, args.horizon, args.target_mode, bench_close)
            if macro_panel is not None:
                sf = attach_macro(sf, macro_panel)
            if len(sf) < args.min_rows_per_symbol:
                log(f"[WARN] {sym} skipped: too few rows after cleaning ({len(sf)})")
                skipped += 1
                continue
            Xcols = [c for c in sf.columns if c not in ("y_cont","y_bin","symbol__")]
            log(f"[OK] {sym} -> X:({len(sf)}, {len(Xcols)}) y:({len(sf)},)")
            frames.append(sf)
            loaded += 1
        except Exception as e:
            log(f"[WARN] {sym} skipped: {e}")
            skipped += 1

    if not frames:
        raise RuntimeError("No symbols loaded.")

    stacked = pd.concat(frames, axis=0)
    stacked = stacked.set_index(["symbol__", stacked.index])
    stacked.index.set_names(["symbol", "date"], inplace=True)

    y_cont = stacked["y_cont"].copy()
    y_bin = stacked["y_bin"].copy()
    X = stacked.drop(columns=["y_cont","y_bin"])

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Optional: prune low-variance cols
    if args.prune_low_variance:
        X2, dropped = drop_low_variance(X)
        if dropped:
            log(f"[CLEAN] Dropping low-variance feature cols: {dropped}")
        X = X2

    # Drop rows with NA anywhere
    valid = X.notna().all(axis=1) & y_cont.notna() & y_bin.notna()
    X = X.loc[valid]
    y_cont = y_cont.loc[valid]
    y_bin = y_bin.loc[valid]

    log(f"[LOAD] symbols loaded: {loaded} | skipped: {skipped}")

    # Time folds
    dates = X.index.get_level_values("date")
    folds = make_time_folds(pd.DatetimeIndex(dates), n_splits=args.n_splits, embargo_days=args.embargo_days)
    if not folds:
        raise RuntimeError("No valid folds produced. Try lowering embargo or n_splits.")

    # OOF containers
    oof_p = pd.Series(index=X.index, dtype=float)
    oof_reg = pd.Series(index=X.index, dtype=float)  # calibrated proxy for RMSE vs standardized target
    y_std = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-12)

    # LGBM callbacks to quiet logs
    callbacks = [lgb.early_stopping(stopping_rounds=100)]
    if args.quiet_lgbm:
        callbacks.insert(0, lgb.log_evaluation(period=0))

    # Per-fold train
    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[np.sort(tr_idx)]
        X_va = X.iloc[np.sort(va_idx)]
        y_tr_bin = y_bin.iloc[np.sort(tr_idx)]
        y_va_bin = y_bin.iloc[np.sort(va_idx)]
        y_va_cont = y_cont.iloc[np.sort(va_idx)]
        y_va_std = y_std.iloc[np.sort(va_idx)]

        # scaling
        if args.scale_features:
            scaler = StandardScaler()
            X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_va_sc = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
        else:
            X_tr_sc, X_va_sc = X_tr, X_va

        clf = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            reg_lambda=args.lambda_l2,
            class_weight=args.class_weight,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr_sc, y_tr_bin, eval_set=[(X_va_sc, y_va_bin)], callbacks=callbacks)

        p_va = clf.predict_proba(X_va_sc)[:, 1]
        oof_p.loc[X_va_sc.index] = p_va

        # simple foldwise linear calibration to y_std: solve a + b*p = y_std
        A = np.vstack([np.ones_like(p_va), p_va]).T
        sol, _, _, _ = np.linalg.lstsq(A, y_va_std.values, rcond=None)
        reg_hat_va = A @ sol
        oof_reg.loc[X_va_sc.index] = reg_hat_va

        rmse = math.sqrt(mean_squared_error(y_va_std, reg_hat_va))
        ic = spearman_ic(pd.Series(p_va, index=X_va_sc.index), y_va_cont)
        try:
            auc = roc_auc_score(y_va_bin, p_va)
        except ValueError:
            auc = np.nan
        try:
            brier = brier_score_loss(y_va_bin, p_va)
        except ValueError:
            brier = np.nan

        log(f"[Fold {i}] RMSE={rmse:.6f} IC={ic:.4f} AUC={auc:.4f} Brier={brier:.6f}")

    # ---------------- OOF summary -----------------
    vm = oof_p.notna()
    mask = vm & oof_reg.notna() & y_std.notna()
    oof_rmse = math.sqrt(mean_squared_error(y_std.loc[mask], oof_reg.loc[mask])) if mask.any() else float("nan")
    # For classifier path: IC from probabilities vs y_cont
    oof_ic = spearman_ic(oof_p.loc[vm], y_cont.loc[vm]) if vm.any() else float("nan")
    oof_auc = roc_auc_score(y_bin.loc[vm], oof_p.loc[vm]) if vm.any() else float("nan")
    oof_brier = brier_score_loss(y_bin.loc[vm], oof_p.loc[vm]) if vm.any() else float("nan")
    log(f"[OOF] RMSE={oof_rmse:.6f} IC={oof_ic:.4f} AUC={oof_auc:.4f} Brier={oof_brier:.6f}")

    # Save OOF
    preds_oof = to_preds_df(oof_p.loc[vm].index, oof_p.loc[vm].values).sort_values(["date","symbol"])
    preds_oof.to_csv(args.preds_oof_out, index=False)

    # Final fit on all
    if args.scale_features:
        scaler_full = StandardScaler()
        X_full = pd.DataFrame(scaler_full.fit_transform(X), index=X.index, columns=X.columns)
    else:
        X_full = X

    clf_full = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        reg_lambda=args.lambda_l2,
        class_weight=args.class_weight,
        n_jobs=-1,
        verbose=-1,
    )
    clf_full.fit(X_full, y_bin, callbacks=[lgb.log_evaluation(period=0)] if args.quiet_lgbm else None)

    p_full = clf_full.predict_proba(X_full)[:, 1]
    preds_full = to_preds_df(X_full.index, p_full).sort_values(["date","symbol"])
    preds_full.to_csv(args.preds_out, index=False)

    # Artifacts
    try:
        clf_full.booster_.save_model(os.path.join(args.out_dir, "model.txt"))
    except Exception:
        pass
    try:
        fi = pd.DataFrame({
            "feature": X.columns,
            "gain": clf_full.booster_.feature_importance(importance_type="gain"),
            "split": clf_full.booster_.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        fi.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False)
    except Exception:
        pass

    summary = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "symbols_loaded": int(loaded),
        "symbols_skipped": int(skipped),
        "n_splits": int(args.n_splits),
        "embargo_days": int(args.embargo_days),
        "oof": {"rmse": float(oof_rmse), "ic": float(oof_ic), "auc": float(oof_auc), "brier": float(oof_brier)},
        "params": {
            "horizon": args.horizon,
            "target_mode": args.target_mode,
            "bench_symbol": args.bench_symbol,
            "scale_features": bool(args.scale_features),
            "prune_low_variance": bool(args.prune_low_variance),
            "lgbm": {
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "bagging_freq": args.bagging_freq,
                "lambda_l2": args.lambda_l2,
                "class_weight": args.class_weight,
            },
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
