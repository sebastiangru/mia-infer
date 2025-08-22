import os, io, argparse, warnings
import numpy as np
import pandas as pd
import joblib

from azure.storage.blob import BlobClient
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

def url_join(base: str, path: str, sas: str | None) -> str:
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

def pick_time_col(df: pd.DataFrame) -> str:
    for c in ["time", "timestamp", "date", "Datetime", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column found (expected one of: time, timestamp, date).")

def build_target(df: pd.DataFrame) -> pd.Series:
    if "close" in df.columns:
        return np.log(df["close"]).diff().shift(-1)  # next-bar log return
    for c in ["logret", "logret1", "ret1"]:
        if c in df.columns:
            return df[c].shift(-1)
    raise ValueError("Could not derive target: need 'close' or a return column like 'logret'/'ret1'.")

def load_universe(path: str | None) -> list[str]:
    if path and os.path.exists(path):
        return pd.read_csv(path)["symbol"].dropna().astype(str).tolist()
    return ['AMZN','GD','NVDA','GOOGL','RHM.DE','PLTR','HIA1.F','RBI.VI',
            'AMD','ASML','TSM','MRVL','SMCI','MRNA','CRSP','BNTX',
            'ARGX','EXEL','SAGE','LDO.MI','BA.L','AIR.PA','HO.PA','HAG.DE']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d", choices=["15m","1h","1d"])
    ap.add_argument("--algo", default="lgbm", choices=["lgbm"])
    ap.add_argument("--output", default="models")
    ap.add_argument("--universe_csv", default="configs/universe.csv")
    args = ap.parse_args()

    FEATURES_BASE = os.getenv("FEATURES_BASE", "https://miatradingdata.blob.core.windows.net/market-data/features")
    FEATURES_SAS  = os.getenv("FEATURES_SAS", "")

    symbols = load_universe(args.universe_csv)

    # -------- Load & concatenate --------
    frames = []
    for sym in symbols:
        path = f"{args.tf}/{sym}/{sym}_features_{args.tf}.csv"
        url = url_join(FEATURES_BASE, path, FEATURES_SAS or None)
        try:
            df = read_csv_url(url)
            df["symbol"] = sym
            frames.append(df)
            print(f"[OK] {sym} -> {len(df)} rows")
        except Exception as e:
            print(f"[WARN] {sym} skipped: {e}")

    if not frames:
        raise SystemExit("No feature files loaded. Check FEATURES_BASE/SAS and folder layout.")

    data = pd.concat(frames, ignore_index=True)
    tcol = pick_time_col(data)
    data[tcol] = pd.to_datetime(data[tcol])

    # -------- Target & features --------
    y = build_target(data)

    drop_cols = {tcol, "symbol"}
    X = data.drop(columns=[c for c in data.columns if c in drop_cols], errors="ignore")
    # Keep only numeric & force float32 (robust with NumPy 2.0)
    X = X.select_dtypes(include=[np.number]).astype(np.float32)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # Chronological split (last 20% validation)
    order = np.argsort(data.loc[valid, tcol].values)
    X = X.iloc[order]
    y = y.iloc[order]

    n = len(X)
    cut = int(n * 0.8)
    X_train, X_val = X.iloc[:cut], X.iloc[cut:]
    y_train, y_val = y.iloc[:cut], y.iloc[cut:]

    # ---- Convert to NumPy (fix for NumPy 2.0 + LightGBM label path) ----
    X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
    X_val_np   = X_val.to_numpy(dtype=np.float32, copy=False)
    y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
    y_val_np   = y_val.to_numpy(dtype=np.float32, copy=False)

    # ---- Regressor ----
    reg = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
    # Regressor
    reg.fit(
    X_train, y_train_np,                              # <— X as DF, y as np
    eval_set=[(X_val, y_val_np)],                     # <— X_val as DF, y_val as np
    eval_metric="l2",
    callbacks=[lgb.early_stopping(stopping_rounds=100),
               lgb.log_evaluation(period=0)]
)

    # ---- Classifier (calibrated) ----
    y_clf = (y > 0).astype(int)
    y_clf_train, y_clf_val = y_clf.iloc[:cut], y_clf.iloc[cut:]
    y_clf_train_np = y_clf_train.to_numpy(dtype=np.int32, copy=False)

    base_clf = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
    clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3)
    clf.fit(X_train, y_clf_train_np)                      # <— X as DF, y as np

    feature_list = X.columns.tolist()
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"features_{args.tf}.json"), "w") as f:
        import json; json.dump(feature_list, f)

    os.makedirs(args.output, exist_ok=True)
    reg_path = os.path.join(args.output, f"lgbm_{args.tf}_reg.pkl")
    clf_path = os.path.join(args.output, f"lgbm_{args.tf}_clf_cal.pkl")
    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)

    # Quick validation report
    yhat = reg.predict(X_val_np)
    p_up = clf.predict_proba(X_val_np)[:, 1]
    mae = float(mean_absolute_error(y_val_np, yhat))
    auc = float(roc_auc_score(y_clf_val.to_numpy(), p_up))
    brier = float(brier_score_loss(y_clf_val.to_numpy(), p_up))
    print({"val_mae": mae, "val_auc": auc, "val_brier": brier})
    print(f"Saved: {reg_path}, {clf_path}")

if __name__ == "__main__":
    main()
