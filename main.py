# --- add/imports near the top of main.py ---
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI

# MUST be before any @app.get/@app.post decorators
app = FastAPI(title="MIA Infer", version="v4.0.2")

from fastapi import Body, HTTPException

# Keep your existing BLOB loading and BUNDLE wiring.
# We assume you already have something like:
# BUNDLE = {"model": booster, "version": "v4.0.1", "features": [...]}
# TIMEFRAME = os.getenv("TIMEFRAME", "1d")

def _get_model_and_features():
    """Return (booster, features_list). Try to read features from model."""
    if BUNDLE is None or "model" not in BUNDLE or BUNDLE["model"] is None:
        raise RuntimeError("Model not loaded")
    booster = BUNDLE["model"]

    # Prefer model feature names if available
    feats = None
    try:
        if hasattr(booster, "feature_name"):
            feats = booster.feature_name()
            # LightGBM can return None if not set; ensure list
            if feats is None:
                feats = []
    except Exception:
        feats = []

    # Fall back to features saved in bundle, or a conservative default list
    if not feats:
        feats = BUNDLE.get("features") or [
            # Safest guess based on your training:
            "ret1","logret1",
            "sma20","sma50","sma200",
            "ema12","ema26",
            "macd","macd_signal","macd_hist",
            "rsi14","atr14","vol20","bb_width",
            # add any other engineered features you trained with
        ]
    return booster, list(feats)

def _to_frame_from_rows(rows: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """Normalize incoming payload (list of dicts or dict) -> DataFrame with lowercase columns."""
    if isinstance(rows, dict):
        # accept { ... } and wrap as single row
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty JSON array or object")

    df = pd.DataFrame(rows)
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows found in payload")

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Keep time/symbol/timeframe if present (for echoing back), but not required for prediction
    return df

def _coerce_numeric(df: pd.DataFrame, use_features: List[str]) -> pd.DataFrame:
    """Ensure numeric dtype for features; add missing features as NaN; order columns."""
    # Make sure the requested feature set exists as columns
    df2 = df.copy()

    # Add any missing model features as NaN
    for c in use_features:
        if c not in df2.columns:
            df2[c] = np.nan

    # Convert features to numeric
    for c in use_features:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Final matrix in the model's feature order
    X = df2[use_features]
    return X

@app.post("/predict", tags=["inference"])
def predict(
    rows: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(..., example=[{"symbol":"AMZN","time":"2025-03-25T13:30:00Z", "ret1":0.01, "ema12":205.7, "ema26":205.7, "macd":0}]),
    thr: float = 0.50,          # optional: classification threshold in response
):
    """
    Accepts either:
      - a JSON array of candle dicts (what n8n sends), or
      - a single candle dict.
    Returns per-row probabilities and a 'latest' summary (last row).
    """
    try:
        booster, feats = _get_model_and_features()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    df = _to_frame_from_rows(rows)

    # Prepare feature matrix in exact model order
    X = _coerce_numeric(df, feats)

    # LightGBM binary models return class probabilities by default
    try:
        yhat = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Build row-wise results
    # Echo back symbol/time if present
    echo_cols = [c for c in ["symbol","time","timeframe","open","high","low","close","volume"] if c in df.columns]
    results = []
    for i, p in enumerate(yhat):
        row = {k: (None if pd.isna(df.at[i, k]) else df.at[i, k]) for k in echo_cols}
        row.update({
            "p_up": float(p),
            "signal": "BUY" if p >= thr else "HOLD/SELL",
        })
        results.append(row)

    # Latest = last row in the payload
    latest = results[-1] if results else None

    return {
        "ok": True,
        "version": BUNDLE.get("version", "unknown"),
        "timeframe": os.getenv("TIMEFRAME", "1d"),
        "features_used": feats,
        "n_rows": len(results),
        "threshold": thr,
        "latest": latest,
        "rows": results,
    }
