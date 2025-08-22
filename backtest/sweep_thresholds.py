# backtest/sweep_thresholds.py
import os, pandas as pd, numpy as np
from simple_backtest import load_prices, portfolio_stats, holding_stats

THRESHOLDS = [round(x,3) for x in np.linspace(0.50, 0.65, 31)]
COST_PER_SIDE = float(os.getenv("COST_PER_SIDE","0.001"))
PRICE_BASE = os.getenv("PRICE_BASE","")
PRICE_SAS  = os.getenv("PRICE_SAS","")

if __name__ == "__main__":
    uni = pd.read_csv("configs/universe.csv")["symbol"].tolist()
    preds = pd.read_csv("preds/1d/lgbm/preds.csv", parse_dates=["time"])
    prices = load_prices(uni)

    p_up = preds.pivot(index="time", columns="symbol", values="p_up").reindex(prices.index).ffill()
    ret = prices.pct_change(fill_method=None).fillna(0.0)

    rows=[]
    for th in THRESHOLDS:
        positions = (p_up > th).astype(int)
        active = positions.sum(axis=1).replace(0, np.nan)
        w = positions.div(active, axis=0).fillna(0.0)

        turnover = w.diff().abs().sum(axis=1).fillna(0.0)
        gross = (w.shift().fillna(0.0) * ret).sum(axis=1)
        net = gross - turnover * COST_PER_SIDE
        eq = (1+net).cumprod()

        st = portfolio_stats(eq)
        st.update({
            "THRESH": th,
            "HitRate": float((net>0).mean()),
            "Exposure": float(w.sum(axis=1).mean()),
            "TurnoverYr": float(turnover.sum() * (252/len(turnover))) if len(turnover)>0 else 0.0,
            "AvgHoldDays": holding_stats(positions)
        })
        rows.append(st)

    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    os.makedirs("reports/1d", exist_ok=True)
    df.to_csv("reports/1d/sweep_thresholds.csv", index=False)
    print(df.head(10))
