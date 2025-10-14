# chain_of_alpha_nasdaq/analytics/labels.py
import pandas as pd
import numpy as np

def make_forward_returns(prices_long: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    Expect prices_long with columns: date,ticker,close (lowercase).
    Returns long-form labels: date,ticker,horizon,fwd_return
      where fwd_return refers to (t -> t+h) using close-to-close simple return.
    """
    df = prices_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot(index="date", columns="ticker", values="close").sort_index()
    fwd = piv.shift(-horizon) / piv - 1.0
    lab = fwd.stack().reset_index()
    lab.columns = ["date","ticker","fwd_return"]
    lab["horizon"] = horizon
    # we align label to the *anchor* date t (predicting t->t+h)
    return lab[["date","ticker","horizon","fwd_return"]].dropna()
