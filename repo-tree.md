```
.
├── .devcontainer
│   └── devcontainer.json
├── backtest
│   ├── enhanced_backtest.py
│   ├── run_vectorbt.py
│   ├── simple_backtest.py
│   └── sweep_thresholds.py
├── configs
│   ├── backtest.yaml
│   ├── data.yaml
│   ├── model_xgb.yaml
│   └── universe.csv
├── Makefile
├── models
│   ├── lgbm_daily
│   │   ├── feature_importance.csv
│   │   ├── model.txt
│   │   └── summary.json
│   ├── lgbm_daily_h10
│   │   ├── calibration_oof.csv
│   │   ├── feature_importance.csv
│   │   ├── model.txt
│   │   └── summary.json
│   ├── lgbm_daily_h5
│   │   ├── calibration_oof.csv
│   │   ├── calibration.csv
│   │   ├── feature_importance.csv
│   │   ├── metrics_by_fold.csv
│   │   ├── model.txt
│   │   └── summary.json
│   ├── lgbm_daily_h5_excess
│   ├── lgbm_daily_v1
│   │   ├── lgbm_clf_calibrated.pkl
│   │   ├── lgbm_reg.pkl
│   │   └── metadata.json
│   ├── features_1d.json
│   ├── lgbm_1d_clf_cal.pkl
│   └── lgbm_1d_reg.pkl
├── preds
│   └── 1d
│       └── lgbm
├── reports
│   └── 1d
│       ├── daily_net_returns.csv
│       ├── enhanced_daily_returns.csv
│       ├── enhanced_equity.csv
│       ├── enhanced_stats.json
│       ├── equity_series.csv
│       ├── summary_stats.csv
│       └── sweep_thresholds.csv
├── runs
│   └── 2025-08-22
│       └── xgb_v1
├── src
│   ├── data
│   │   └── universe.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── reports.py
│   ├── feature_engineering
│   │   ├── __init__.py
│   │   └── enhanced_features.py
│   ├── io
│   │   ├── __init__.py
│   │   ├── azure_blob.py
│   │   ├── blob_paths.py
│   │   └── loaders.py
│   ├── portfolio
│   │   ├── __init__.py
│   │   ├── construct.py
│   │   └── costs.py
│   ├── predict
│   │   └── save_preds.py
│   ├── trainers
│   │   ├── __init__.py
│   │   ├── optimization.py
│   │   ├── train_daily_lgbm.py
│   │   ├── train_daily_xgb.py
│   │   ├── train_tabular.py
│   │   └── utils_cv.py
│   └── __init__.py
├── tools
│   ├── fetch_sectors.py
│   ├── run_backtest.sh
│   └── run_train.sh
├── Dockerfile
├── main.py
├── payload.json
├── repo-tree.md
└── requirements.txt

28 directories, 64 files
```
