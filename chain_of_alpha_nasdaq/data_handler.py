import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# TEMP: Direct Polygon fetch helper (for debugging / dev use)
# -------------------------------------------------------------------
def fetch_polygon(symbol: str, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Directly fetch daily OHLCV data from Polygon API for a single ticker.
    Returns a clean DataFrame with ['date', 'open', 'high', 'low', 'close', 'volume'].
    """
    api_key = api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("No Polygon API key found in environment.")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    print(f"📡 Fetching {symbol} from Polygon...")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"⚠️ Polygon fetch failed for {symbol}: {resp.status_code}")
        return pd.DataFrame()

    data = resp.json().get("results", [])
    if not data:
        print(f"⚠️ No results for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
        inplace=True,
    )

    df = df[["date", "open", "high", "low", "close", "volume"]]
    print(f"✅ Got {len(df)} rows for {symbol} ({df['date'].min().date()} → {df['date'].max().date()})")
    return df

def fetch_data(tickers, start="2020-01-01", end="2024-01-01"):
    import os
    import pandas as pd

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found in environment")

    all_dfs = []

    for ticker in tickers:
        print(f"📡 Fetching {ticker} data...")
        df = fetch_polygon(ticker, start, end, api_key)
        if df is None or df.empty:
            print(f"⚠️ Could not fetch data for {ticker}, skipping.")
            continue

        # ✅ normalize column case & ensure date index
        df.columns = [c.capitalize() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # prefix columns for multi-ticker clarity
        df = df.add_prefix(f"{ticker}_")
        all_dfs.append(df)

        print(f"✅ Got {len(df)} rows from Polygon for {ticker}")

    if not all_dfs:
        raise RuntimeError("No market data available from any source.")

    # ✅ Outer join to preserve all dates
    combined = pd.concat(all_dfs, axis=1, join="outer").sort_index()

    # ✅ Trim to range
    mask = (combined.index >= pd.to_datetime(start)) & (combined.index <= pd.to_datetime(end))
    combined = combined.loc[mask]

    print(f"📊 Final dataset shape: {combined.shape[0]} rows, {len(tickers)} tickers.")
    return combined

# -------------------------------------------------------------------
# Alpha Vantage fallback
# -------------------------------------------------------------------
def _fetch_alphavantage(ticker, api_key):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}&outputsize=full&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json().get("Time Series (Daily)", {})
        if not data:
            return None
        df = pd.DataFrame(data).T.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "6. volume": "volume",
            }
        )
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        df.sort_values("date", inplace=True)
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None
