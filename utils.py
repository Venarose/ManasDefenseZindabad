import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.covariance import EmpiricalCovariance

def parse_dates_safe(series):
    return pd.to_datetime(series, errors="coerce", utc=False)

def winsorize(series, lower=0.01, upper=0.99):
    if not np.issubdtype(series.dtype, np.number):
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def clean_prices(df):
    out = df.copy()
    out["date"] = parse_dates_safe(out["date"])
    out["close"] = coerce_numeric(out["close"])
    out["volume"] = coerce_numeric(out["volume"])
    out["market_cap"] = coerce_numeric(out["market_cap"])
    out = out[~out["ticker"].isin(["???", None, "N/A"])]
    out = out[out["date"] <= pd.Timestamp.today().normalize()]
    for col in ["close","volume","market_cap"]:
        out[col] = winsorize(out[col])
        out[col] = out[col].interpolate().fillna(method="bfill").fillna(method="ffill")
    out = out.drop_duplicates(subset=["date","ticker"])
    out = out.sort_values(["ticker","date"])
    return out

def clean_fundamentals(df):
    out = df.copy()
    out["quarter"] = parse_dates_safe(out["quarter"])
    for col in ["revenue","ebit_margin","ebit","rnd_spend","debt_to_equity"]:
        out[col] = coerce_numeric(out[col])
        out[col] = winsorize(out[col])
        out[col] = out[col].interpolate().fillna(method="bfill").fillna(method="ffill")
    out = out.drop_duplicates(subset=["quarter","ticker"])
    out = out.sort_values(["ticker","quarter"])
    return out

def clean_contracts(df):
    out = df.copy()
    out["award_date"] = parse_dates_safe(out["award_date"])
    out["value_usd"] = coerce_numeric(out["value_usd"])
    out["duration_months"] = coerce_numeric(out["duration_months"])
    for col in ["value_usd","duration_months"]:
        out[col] = winsorize(out[col])
        out[col] = out[col].interpolate().fillna(method="bfill").fillna(method="ffill")
    out = out.drop_duplicates(subset=["contract_id"])
    out = out.sort_values(["ticker","award_date"])
    return out

def clean_news(df):
    out = df.copy()
    out["date"] = parse_dates_safe(out["date"])
    out["sentiment"] = coerce_numeric(out["sentiment"]).clip(-1,1)
    out["url"] = out["url"].where(out["url"].str.startswith("http"), np.nan)
    out = out.drop_duplicates(subset=["ticker","date","headline"])
    out = out.sort_values(["ticker","date"])
    return out

def returns_from_prices(df):
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = wide.pct_change().dropna(how="all")
    return rets

def drawdown(series):
    cum = (1+series).cumprod()
    peak = cum.cummax()
    dd = (cum/peak) - 1.0
    return dd

def sharpe(returns, rf=0.0):
    mu = returns.mean()*252 - rf
    sigma = returns.std()*np.sqrt(252)
    return mu / (sigma.replace(0, np.nan))

def cov_matrix(returns):
    return returns.cov()*252

def monte_carlo_portfolios(returns, n=5000, rf=0.0, seed=42):
    np.random.seed(seed)
    tickers = returns.columns.tolist()
    mean_ret = returns.mean()*252
    cov = returns.cov()*252
    all_weights = []
    all_ret = []
    all_vol = []
    all_sharpe = []
    for _ in range(n):
        w = np.random.random(len(tickers))
        w = w / w.sum()
        port_ret = np.dot(w, mean_ret)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        s = (port_ret - rf) / (port_vol if port_vol != 0 else np.nan)
        all_weights.append(w)
        all_ret.append(port_ret)
        all_vol.append(port_vol)
        all_sharpe.append(s)
    res = pd.DataFrame({"return": all_ret, "vol": all_vol, "sharpe": all_sharpe})
    weights_df = pd.DataFrame(all_weights, columns=tickers)
    return res, weights_df

def scenario_impact(base_revenue, budget_growth=0.03, conflict_intensity=0.0, delay_months=0, fx_delta=0.0):
    """
    Simple toy model: 
    - revenue grows with defense budget growth, elasticity ~ 0.6
    - conflict intensity (0-1) adds up to +10% demand at 1.0
    - procurement delays reduce revenue recognition proportional to delay
    - FX delta affects 40% of revenue (exports)
    """
    elasticity = 0.6
    conflict_boost = 0.10 * conflict_intensity
    delay_penalty = min(0.25, delay_months/60.0)
    fx_effect = 0.4 * fx_delta
    factor = (1 + elasticity*budget_growth) * (1 + conflict_boost) * (1 - delay_penalty) * (1 + fx_effect)
    return base_revenue * factor
