import os, pandas as pd, numpy as np, io, joblib, json
from azure.storage.blob import BlobClient

def read_csv_url(url):
    bc = BlobClient.from_blob_url(url)
    return pd.read_csv(io.BytesIO(bc.download_blob().readall()))

def url_join(base, path, sas=None):
    base = base[:-1] if base.endswith("/") else base
    full = f"{base}/{path.lstrip('/')}"
    if sas:
        sep = "&" if "?" in full else "?"
        full = f"{full}{sep}{sas}"
    return full

if __name__ == "__main__":
    FEATURES_BASE = os.getenv("FEATURES_BASE", "https://miatradingdata.blob.core.windows.net/market-data/features")
    FEATURES_SAS  = os.getenv("FEATURES_SAS","")  # no leading ?
    UNIVERSE_CSV  = os.getenv("UNIVERSE_CSV", "configs/universe.csv")

    reg_url = os.environ["LGBM_REG_URL"]
    clf_url = os.environ["LGBM_CLF_URL"]
    reg = joblib.load(io.BytesIO(BlobClient.from_blob_url(reg_url).download_blob().readall()))
    clf = joblib.load(io.BytesIO(BlobClient.from_blob_url(clf_url).download_blob().readall()))

    uni = pd.read_csv(UNIVERSE_CSV)["symbol"].tolist()
    rows = []
    for sym in uni:
        url = url_join(FEATURES_BASE, f"1d/{sym}/{sym}_features_1d.csv", FEATURES_SAS or None)
        try:
            df = read_csv_url(url)
        except Exception as e:
            print(f"[WARN] {sym}: {e}"); continue

        # pick time column
        tcol = next((c for c in ["time","timestamp","date","Datetime","datetime"] if c in df.columns), None)
        if not tcol: raise SystemExit("No time-like column in features")
        time = pd.to_datetime(df[tcol])

        # build X = numeric only
        X = df.drop(columns=[tcol,"symbol"], errors="ignore").select_dtypes(include=[np.number]).astype(np.float32)

        yhat = reg.predict(X)
        p_up = clf.predict_proba(X)[:,1]

        out = pd.DataFrame({"time": time, "symbol": sym, "yhat": yhat, "p_up": p_up})
        rows.append(out)

    preds = pd.concat(rows).sort_values(["symbol","time"])
    os.makedirs("preds/1d/lgbm", exist_ok=True)
    fn = f"preds/1d/lgbm/preds.csv"
    preds.to_csv(fn, index=False)
    print(f"saved {fn} with {len(preds)} rows")
