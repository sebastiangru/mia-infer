import os, joblib, mlflow, argparse, pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from src.io.blob_paths import features_csv
from src.data.universe import load_universe

def load_features(tf: str, symbols: list[str]) -> pd.DataFrame:
    dfs = []
    for s in symbols:
        try:
            df = pd.read_csv(features_csv(tf, s))
            df['symbol'] = s
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] {s}: {e}")
    return pd.concat(dfs, ignore_index=True).sort_values(['symbol','time'])

def split_chrono(df: pd.DataFrame, dtcol='time', train_end=None, val_end=None):
    df[dtcol] = pd.to_datetime(df[dtcol])
    if train_end is None: train_end = df[dtcol].quantile(0.7)
    if val_end   is None: val_end   = df[dtcol].quantile(0.85)
    tr = df[df[dtcol] <= train_end]
    va = df[(df[dtcol] > train_end) & (df[dtcol] <= val_end)]
    te = df[df[dtcol] > val_end]
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d", choices=["15m","1h","1d"])
    ap.add_argument("--algo", default="lgbm", choices=["lgbm","xgb"])
    ap.add_argument("--output", default="models")
    ap.add_argument("--target", default="logret1", help="next-bar return column name in your CSV")
    args = ap.parse_args()

    mlflow.set_experiment(f"mia_{args.tf}_{args.algo}")
    mlflow.autolog()

    symbols = load_universe()
    df = load_features(args.tf, symbols)

    # Select features (all numeric except leaks and ids)
    drop_cols = {"time","symbol","timeframe"}
    y = df[args.target].shift(-1)  # next period target (as in your files)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.loc[y.index]
    y = y.dropna()
    X = X.loc[y.index]

    tr, va, te = split_chrono(pd.concat([X, y.rename("__y")], axis=1))
    X_tr, y_tr = tr.drop(columns="__y"), tr["__y"]
    X_va, y_va = va.drop(columns="__y"), va["__y"]
    X_te, y_te = te.drop(columns="__y"), te["__y"]

    if args.algo == "lgbm":
        reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=63)
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05)
    else:
        reg = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist")
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="logloss")

    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    yhat = reg.predict(X_te)

    # Prob-up head (sign)
    y_sign_tr = (y_tr > 0).astype(int)
    clf.fit(X_tr, y_sign_tr)
    # Optionally calibrate
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    cal.fit(X_va, (y_va > 0).astype(int))
    p_up = cal.predict_proba(X_te)[:,1]

    rmse = float(mean_squared_error(y_te, yhat, squared=False))
    auc  = float(roc_auc_score((y_te>0).astype(int), p_up))
    brier= float(brier_score_loss((y_te>0).astype(int), p_up))

    mlflow.log_metric("rmse_te", rmse)
    mlflow.log_metric("auc_te", auc)
    mlflow.log_metric("brier_te", brier)

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(reg, f"{args.output}/{args.algo}_{args.tf}_reg.pkl")
    joblib.dump(cal, f"{args.output}/{args.algo}_{args.tf}_clf_cal.pkl")
    print({"rmse":rmse,"auc":auc,"brier":brier})

if __name__ == "__main__":
    main()
