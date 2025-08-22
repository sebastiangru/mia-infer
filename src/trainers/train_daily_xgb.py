# src/trainers/train_daily_xgb.py
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from joblib import dump
import yaml

from src.trainers.utils_cv import PurgedKFold
from src.io.loaders import load_panel_for_training

def fit_calibrator(y_true: np.ndarray, p_raw: np.ndarray, method: str = "isotonic"):
    """Return (fitted_calibrator, transform_fn)."""
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_raw, y_true)
        return cal, (lambda x: cal.transform(x))
    elif method == "platt":
        lr = LogisticRegression(max_iter=200)
        lr.fit(p_raw.reshape(-1,1), y_true)
        return lr, (lambda x: lr.predict_proba(x.reshape(-1,1))[:,1])
    else:
        return None, (lambda x: x)

def spearman_ic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if mask.sum() < 3:
        return np.nan
    return spearmanr(y_true[mask], y_score[mask]).correlation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model_xgb.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.model_config, "r") as f:
        mcfg: Dict[str, Any] = yaml.safe_load(f)

    # Load panel
    X, y_bin, y_cont, dates, feature_names, symbols = load_panel_for_training(args.data_config, args.model_config)

    # CV
    n_splits = int(mcfg.get("n_splits", 5))
    embargo = int(mcfg.get("embargo_days", 5))
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo)

    # XGB params
    xgb_params = dict(mcfg.get("params", {}))
    early_stop = int(mcfg.get("early_stopping_rounds", 100))
    calibration = str(mcfg.get("calibration", "isotonic"))

    # OOF containers
    oof_raw = np.full(len(y_bin), np.nan, dtype=float)
    fold_metrics = []

    for fold, (tr, va) in enumerate(cv.split(dates), start=1):
        model = XGBClassifier(**xgb_params)
        model.fit(
            X[tr], y_bin[tr],
            eval_set=[(X[va], y_bin[va])],
            eval_metric="auc",
            verbose=False,
            early_stopping_rounds=early_stop
        )
        p = model.predict_proba(X[va])[:, 1]
        oof_raw[va] = p

        auc = roc_auc_score(y_bin[va], p)
        # IC uses continuous target to judge rank quality
        ic  = spearman_ic(y_cont[va], p)
        br  = brier_score_loss(y_bin[va], p)

        fold_metrics.append({"fold": fold, "auc": float(auc), "ic": float(ic), "brier": float(br),
                             "best_iter": int(model.get_booster().best_iteration or 0)})

        # Save per-fold model
        dump(model, os.path.join(args.out, f"model_fold{fold}.joblib"))

    # Calibrate on OOF
    mask = np.isfinite(oof_raw)
    cal, cal_fn = fit_calibrator(y_bin[mask], oof_raw[mask], calibration)
    oof_cal = np.full_like(oof_raw, np.nan)
    oof_cal[mask] = cal_fn(oof_raw[mask])

    # Aggregate metrics
    auc_oof = roc_auc_score(y_bin[mask], oof_raw[mask])
    ic_oof  = spearman_ic(y_cont[mask], oof_raw[mask])
    br_oof  = brier_score_loss(y_bin[mask], oof_raw[mask])
    auc_oof_cal = roc_auc_score(y_bin[mask], oof_cal[mask])
    br_oof_cal  = brier_score_loss(y_bin[mask], oof_cal[mask])

    summary = {
        "fold_metrics": fold_metrics,
        "oof": {
            "auc_raw": float(auc_oof),
            "ic_raw": float(ic_oof),
            "brier_raw": float(br_oof),
            "auc_cal": float(auc_oof_cal),
            "brier_cal": float(br_oof_cal),
        },
        "params": xgb_params,
        "n_splits": n_splits,
        "embargo": embargo,
        "calibration": calibration,
        "features": feature_names,
        "symbols": symbols,
    }

    # Save artifacts
    pd.DataFrame({
        "date": pd.to_datetime(dates),
        "p_raw": oof_raw,
        "p_cal": oof_cal,
        "y_bin": y_bin,
        "y_cont": y_cont
    }).to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save calibrator
    if cal is not None:
        dump(cal, os.path.join(args.out, f"calibrator_{calibration}.joblib"))

    print(json.dumps(summary["oof"], indent=2))

if __name__ == "__main__":
    main()
