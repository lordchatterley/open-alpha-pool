# chain_of_alpha_nasdaq/analytics/features.py
import pandas as pd
import numpy as np

def features_from_factors(factors_wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Input: dict of name -> DataFrame (index=date, columns=tickers) for each factor's per-ticker signal.
    Output: long-form features with columns: date, ticker, feature_name, value
    """
    frames = []
    for name, df in factors_wide.items():
        tmp = df.stack().reset_index()
        tmp.columns = ["date", "ticker", "value"]
        tmp["feature_name"] = name
        frames.append(tmp[["date", "ticker", "feature_name", "value"]])
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "feature_name", "value"])
    feats = pd.concat(frames, ignore_index=True)
    feats["date"] = pd.to_datetime(feats["date"])
    feats["ticker"] = feats["ticker"].astype(str)
    return feats

def cheap_price_features(prices_long: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a few stable technicals per (date,ticker) to bootstrap ML:
      - intraday_range = (high - low) / (close + 1e-9)
      - close_to_open   = (close - open) / (open + 1e-9)
      - volume_z        = z-score of volume cross-sectionally per day
    Expect columns: ['date','ticker','open','high','low','close','volume'] (lowercase).
    Returns long-form rows appended with feature_name and value.
    """
    df = prices_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df["intraday_range"] = (df["high"] - df["low"]) / (df["close"].abs() + 1e-9)
    df["close_to_open"]  = (df["close"] - df["open"]) / (df["open"].abs() + 1e-9)

    # cross-sectional z-score of volume per date
    def _z(g):
        return (g - g.mean()) / (g.std(ddof=0) + 1e-9)
    df["volume_z"] = df.groupby("date")["volume"].transform(_z)

    melted = df.melt(
        id_vars=["date","ticker"],
        value_vars=["intraday_range","close_to_open","volume_z"],
        var_name="feature_name",
        value_name="value"
    )
    return melted[["date","ticker","feature_name","value"]]
