import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded each run

CACHE_PATH = Path("data/nasdaq_tickers.parquet")

def fetch_nasdaq_tickers(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the full list of NASDAQ-listed tickers from Polygon.
    Returns a DataFrame with columns: symbol, name, type, active.
    Caches locally for reuse.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    print(f"🔑 POLYGON_API_KEY detected: {bool(api_key)} length={len(api_key) if api_key else 0}")
    if not api_key:
        raise RuntimeError("Missing POLYGON_API_KEY in environment")

    if CACHE_PATH.exists() and not force_refresh:
        print(f"📁 Loaded cached NASDAQ tickers: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print("📡 Fetching full NASDAQ ticker list from Polygon...")
    base_url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "exchange": "XNAS",
        "active": "true",
        "limit": 1000,
        "apiKey": api_key,
    }

    all_tickers = []
    next_url = base_url

    while next_url:
        print(f"🔗 Requesting: {next_url[:120]}...")
        # If next_url is the base endpoint, pass params (includes apiKey); if next_url is full URL from API, call it directly.
        if next_url.startswith(base_url):
            resp = requests.get(next_url, params=params)
        else:
            resp = requests.get(next_url)

        print(resp.status_code, resp.text[:120])
        if resp.status_code != 200:
            raise RuntimeError(f"Polygon ticker fetch failed: {resp.status_code}")

        data = resp.json()
        results = data.get("results", [])
        all_tickers.extend(results)
        next_url = data.get("next_url")

    df = pd.DataFrame(all_tickers)
    if not df.empty and "type" in df.columns:
        df = df[df["type"] == "CS"]  # common stock only
    df = df[["ticker", "name", "active"]].reset_index(drop=True)
    df = df.sort_values("ticker")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f"✅ Saved {len(df)} NASDAQ tickers → {CACHE_PATH}")
    return df


if __name__ == "__main__":
    df = fetch_nasdaq_tickers(force_refresh=True)
    print(df.head(20))
    print(f"Total active NASDAQ tickers: {len(df)}")
