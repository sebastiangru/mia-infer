# main.py
import os
import io
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from azure.storage.blob import BlobClient

# Optional: load .env locally during dev
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")

# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default

def resolve_model_url(env_url_name: str, env_file_name: str, base_env: str = "MODELS_BLOB") -> Optional[str]:
    """
    Resolve a model location from either a full URL (e.g. LGBM_REG_URL)
    or a filename combined with MODELS_BLOB (e.g. LGBM_REG_FILE).
    """
    direct = _env(env_url_name)
    if direct:
        return direct
    base = _env(base_env)
    fname = _env(env_file_name)
    if base and fname:
        if base.endswith("/"):
            return base + fname
        return base + "/" + fname
    return None

# ---------------------------------------------------------------------
# I/O: load joblib from Blob URL (works for public or SAS-protected blobs)
# ---------------------------------------------------------------------

def load_joblib_from_url(url: str):
    try:
        bc = BlobClient.from_blob_url(url)
        data = bc.download_blob().readall()
        return joblib.load(io.BytesIO(data))
    except Exception as e:
        # Fallback: if azure client fails, try plain HTTP GET if accessible
        try:
            import requests  # lazy import
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return joblib.load(io.BytesIO(r.content))
        except Exception as e2:
            raise RuntimeError(f"Failed to load joblib from URL: {url}\nPrimary error: {e}\nHTTP fallback error: {e2}")

# ---------------------------------------------------------------------
# Models & feature name helpers
# ---------------------------------------------------------------------

class LoadedModels(BaseModel):
    lgbm_reg: Any | None = None
    lgbm_clf: Any | None = None
    xgb_reg: Any | None = None
    xgb_clf: Any | None = None
    lgbm_q10: Any | None = None   # optional quantile regressor
    lgbm_q90: Any | None = None

models = LoadedModels()

def _feature_names_from_model(m) -> Optional[List[str]]:
    """
    Try to pull the expected feature names list from LightGBM or XGBoost models.
    """
    if m is None:
        return None
    # LightGBM sklearn wrapper
    fn = getattr(m, "feature_name_", None)
    if fn:
        return list(fn)
    booster = getattr(m, "booster_", None)
    if booster is not None:
        try:
            return list(booster.feature_name())
        except Exception:
            pass
    # XGBoost sklearn wrapper
    try:
        booster = m.get_booster()
        if booster is not None and booster.feature_names:
            return list(booster.feature_names)
    except Exception:
        pass
    return None

def _align_features(x: pd.DataFrame, expected: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reindex incoming features to expected names; drop extras; fill missing with 0.
    Returns (X_aligned, warnings).
    """
    warnings: List[str] = []
    if expected is None:
        warnings.append("No feature_name info on model; using provided features as-is.")
        return x, warnings
    # Extra cols
    extra = [c for c in x.columns if c not in expected]
    if extra:
        warnings.append(f"Dropping unexpected features: {extra}")
    # Missing cols
    missing = [c for c in expected if c not in x.columns]
    if missing:
        warnings.append(f"Missing features not provided; filling with 0: {missing}")
    X = x.reindex(columns=expected, fill_value=0.0)
    return X, warnings

# ---------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------

class ScoreRequest(BaseModel):
    symbol: str = Field(..., description="Ticker (e.g., AMZN)")
    tf: str = Field("1d", description="Timeframe like 15m, 1h, 1d")
    features: Dict[str, float] = Field(..., description="Flat dict of feature_name -> value")
    timestamp: Optional[str] = Field(None, description="ISO time of the bar (optional)")

class ScoreResponse(BaseModel):
    symbol: str
    tf: str
    yhat_lgbm: Optional[float] = None
    yhat_xgb: Optional[float] = None
    yhat_ens: Optional[float] = None
    p_up_lgbm: Optional[float] = None
    p_up_xgb: Optional[float] = None
    p_up_ens: Optional[float] = None
    q_lo: Optional[float] = None
    q_hi: Optional[float] = None
    meta: Dict[str, Any]

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(title="MIA Inference Service", version=APP_VERSION)

def _load_all_models():
    # Resolve URLs (either *_URL or *_FILE + MODELS_BLOB)
    urls = {
        "lgbm_reg": resolve_model_url("LGBM_REG_URL", "LGBM_REG_FILE"),
        "lgbm_clf": resolve_model_url("LGBM_CLF_URL", "LGBM_CLF_FILE"),
        "xgb_reg":  resolve_model_url("XGB_REG_URL",  "XGB_REG_FILE"),
        "xgb_clf":  resolve_model_url("XGB_CLF_URL",  "XGB_CLF_FILE"),
        "lgbm_q10": resolve_model_url("LGBM_Q10_URL", "LGBM_Q10_FILE"),
        "lgbm_q90": resolve_model_url("LGBM_Q90_URL", "LGBM_Q90_FILE"),
    }
    loaded = {}
    for k, url in urls.items():
        if not url:
            loaded[k] = None
            continue
        loaded[k] = load_joblib_from_url(url)
    return LoadedModels(**loaded)

@app.on_event("startup")
def _startup():
    global models
    models = _load_all_models()

@app.get("/health")
def health():
    ok = (models.lgbm_reg is not None) or (models.xgb_reg is not None)
    return {"ok": ok, "version": APP_VERSION}

@app.get("/ready")
def ready():
    ok = all([
        (models.lgbm_reg is not None),
        (models.lgbm_clf is not None),
        (models.xgb_reg  is not None),
        (models.xgb_clf  is not None),
    ])
    return {"ready": ok, "version": APP_VERSION}

@app.get("/modelinfo")
def modelinfo():
    info = {}
    for name in ["lgbm_reg","lgbm_clf","xgb_reg","xgb_clf","lgbm_q10","lgbm_q90"]:
        m = getattr(models, name)
        if m is None:
            info[name] = {"loaded": False}
        else:
            info[name] = {
                "loaded": True,
                "feature_names": _feature_names_from_model(m)
            }
    return {"version": APP_VERSION, "models": info}

@app.post("/reload")
def reload_models():
    global models
    try:
        models = _load_all_models()
        return {"ok": True, "version": APP_VERSION}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest = Body(...)):
    if not req.features:
        raise HTTPException(400, "Missing 'features'")

    # Convert to single-row DataFrame
    x_in = pd.DataFrame([req.features])

    warnings: List[str] = []
    yhat_l = yhat_x = None
    p_l = p_x = None
    q_lo = q_hi = None

    # Regression heads
    if models.lgbm_reg is not None:
        exp = _feature_names_from_model(models.lgbm_reg)
        Xl, w = _align_features(x_in, exp)
        warnings.extend(w)
        yhat_l = float(models.lgbm_reg.predict(Xl)[0])
    if models.xgb_reg is not None:
        exp = _feature_names_from_model(models.xgb_reg)
        Xx, w = _align_features(x_in, exp)
        warnings.extend(w)
        yhat_x = float(models.xgb_reg.predict(Xx)[0])

    # Classification heads (prob up)
    if models.lgbm_clf is not None:
        exp = _feature_names_from_model(models.lgbm_clf)
        Xlc, w = _align_features(x_in, exp)
        warnings.extend(w)
        try:
            p_l = float(models.lgbm_clf.predict_proba(Xlc)[:,1][0])
        except Exception:
            # Some calibrated wrappers are stored directly as 'calibrated' model
            p_l = float(models.lgbm_clf.predict_proba(Xlc)[:,1][0])
    if models.xgb_clf is not None:
        exp = _feature_names_from_model(models.xgb_clf)
        Xxc, w = _align_features(x_in, exp)
        warnings.extend(w)
        p_x = float(models.xgb_clf.predict_proba(Xxc)[:,1][0])

    # Optional quantile heads (q10/q90)
    if models.lgbm_q10 is not None:
        exp = _feature_names_from_model(models.lgbm_q10)
        Xq10, w = _align_features(x_in, exp)
        warnings.extend(w)
        q_lo = float(models.lgbm_q10.predict(Xq10)[0])
    if models.lgbm_q90 is not None:
        exp = _feature_names_from_model(models.lgbm_q90)
        Xq90, w = _align_features(x_in, exp)
        warnings.extend(w)
        q_hi = float(models.lgbm_q90.predict(Xq90)[0])

    # Simple ensemble
    W_LGBM = float(_env("W_LGBM", "0.5"))
    W_XGB  = float(_env("W_XGB",  "0.5"))
    yhat_ens = None
    if (yhat_l is not None) or (yhat_x is not None):
        a = yhat_l if yhat_l is not None else 0.0
        b = yhat_x if yhat_x is not None else 0.0
        yhat_ens = W_LGBM * a + W_XGB * b

    # Noisy-OR for combining probabilities
    p_up_ens = None
    if (p_l is not None) and (p_x is not None):
        p_up_ens = 1.0 - (1.0 - p_l) * (1.0 - p_x)
    elif p_l is not None:
        p_up_ens = p_l
    elif p_x is not None:
        p_up_ens = p_x

    meta = {
        "version": APP_VERSION,
        "warnings": warnings or None
    }

    return ScoreResponse(
        symbol=req.symbol,
        tf=req.tf,
        yhat_lgbm=yhat_l,
        yhat_xgb=yhat_x,
        yhat_ens=yhat_ens,
        p_up_lgbm=p_l,
        p_up_xgb=p_x,
        p_up_ens=p_up_ens,
        q_lo=q_lo,
        q_hi=q_hi,
        meta=meta
    )
