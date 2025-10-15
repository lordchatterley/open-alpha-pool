import os
from pathlib import Path
import importlib
import io
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
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

STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame with a DateTime index and canonical columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    frame = df.copy()

    # Normalise column names to lower case for matching
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        idx = pd.to_datetime(frame.pop("date"), errors="coerce")
    else:
        idx = pd.to_datetime(frame.index, errors="coerce")

    frame.index = idx
    frame = frame.sort_index()

    canonical = pd.DataFrame(index=frame.index)
    column_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    for raw_name, canonical_name in column_map.items():
        if raw_name in frame.columns:
            series = pd.to_numeric(frame[raw_name], errors="coerce")
        elif canonical_name.lower() in frame.columns:
            series = pd.to_numeric(frame[canonical_name.lower()], errors="coerce")
        elif canonical_name in frame.columns:
            series = pd.to_numeric(frame[canonical_name], errors="coerce")
        else:
            series = pd.Series(np.nan, index=frame.index)
        canonical[canonical_name] = series

    canonical = canonical.dropna(how="all")
    canonical = canonical[STANDARD_COLUMNS]
    canonical.index.name = "Date"
    canonical = canonical[~canonical.index.duplicated(keep="first")]
    return canonical


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download data via yfinance if the package is available."""
    if importlib.util.find_spec("yfinance") is None:
        return pd.DataFrame()

    yf = importlib.import_module("yfinance")
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.reset_index()
    data.rename(columns={"Adj Close": "Close"}, inplace=True)
    return data[[col for col in data.columns if col.lower() in {"date", "open", "high", "low", "close", "volume"}]]


def _load_local_fixture(ticker: str, directory: str | os.PathLike[str] | None) -> pd.DataFrame:
    if directory is None:
        return pd.DataFrame()

    path = Path(directory) / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


def _load_nasdaq_directory() -> list[str]:
    """Fallback loader for NASDAQ symbol directory (no API key required)."""
    url = "https://ftp.nasdaqtrader.com/dynamic/SYMBOLDirectory/nasdaqlisted.txt"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        df = pd.read_csv(io.StringIO(resp.text), sep="|")
        if "Symbol" not in df.columns:
            raise RuntimeError("Invalid NASDAQ directory schema")

        df = df[df.get("Test Issue", "N") == "N"]
        symbols = [sym.strip().upper() for sym in df["Symbol"].dropna().unique().tolist() if sym]
        return symbols
    except Exception as exc:
        print(f"⚠️ Fallback NASDAQ directory download failed: {exc}")
        return []


def _resolve_tickers(tickers: Sequence[str] | str | Iterable[str]) -> list[str]:
    """Accept a list of tickers or a named universe."""

    if isinstance(tickers, str):
        universe = tickers.strip().lower()
        if universe != "nasdaq":
            raise ValueError(f"Unsupported ticker universe: {tickers}")

        try:
            from chain_of_alpha_nasdaq.utils import ticker_fetcher

            df = ticker_fetcher.fetch_nasdaq_tickers()
            if "ticker" in df.columns:
                symbols = df["ticker"].astype(str).str.strip()
            else:
                symbols = df.iloc[:, 0].astype(str).str.strip()
            resolved = [sym.upper() for sym in symbols.tolist() if sym]
            if resolved:
                return resolved
        except Exception as exc:
            print(f"⚠️ Polygon NASDAQ fetch unavailable: {exc}")

        fallback = _load_nasdaq_directory()
        if fallback:
            return fallback
        raise RuntimeError("Unable to resolve NASDAQ ticker universe")

    if isinstance(tickers, Iterable):
        resolved = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not resolved:
            raise ValueError("Ticker list is empty")
        return resolved

    raise TypeError("tickers must be a sequence or a supported universe name")


def fetch_data(
    tickers: Sequence[str] | str | Iterable[str],
    start="2020-01-01",
    end="2024-01-01",
    source: str | None = None,
    local_dir: str | os.PathLike[str] | None = None,
):
    """Fetch daily OHLCV data for one or more tickers.

    Args:
        tickers: Sequence of tickers or the string "nasdaq" to request the full
            NASDAQ universe (resolved via Polygon when available, falling back to
            the public NASDAQ Trader symbol directory).
        start: ISO date string for the inclusive start of the history window.
        end: ISO date string for the inclusive end of the history window.
        source: Force a specific data source ("polygon", "yfinance", "local").
            If omitted the loader chooses Polygon when an API key is present and
            otherwise falls back to yfinance and finally local fixtures.
        local_dir: Optional directory containing ``{ticker}.csv`` fixtures used
            for regression tests or offline development.

    Returns:
        ``pd.DataFrame`` indexed by ``Date`` with a ``MultiIndex`` over columns
        ``["Ticker", "Field"]`` providing the canonical OHLCV schema for each
        successfully loaded symbol.
    """
    api_key = os.getenv("POLYGON_API_KEY")

    if source is None:
        source = "polygon" if api_key else "yfinance"

    source = source.lower()
    if source not in {"polygon", "yfinance", "local"}:
        raise ValueError("source must be 'polygon', 'yfinance', or 'local'")

    def _source_sequence():
        if source == "polygon":
            if api_key:
                yield "polygon"
            yield "yfinance"
            yield "local"
        elif source == "yfinance":
            yield "yfinance"
            yield "local"
        else:
            yield "local"

    all_dfs: list[pd.DataFrame] = []
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    resolved_tickers = _resolve_tickers(tickers)

    for ticker in resolved_tickers:
        print(f"📡 Fetching {ticker} data...")
        frame = pd.DataFrame()

        for src in _source_sequence():
            if src == "polygon":
                try:
                    frame = fetch_polygon(ticker, start, end, api_key)
                except Exception as exc:
                    print(f"⚠️ Polygon fetch failed for {ticker}: {exc}")
                    frame = pd.DataFrame()
            elif src == "yfinance":
                frame = _fetch_yfinance(ticker, start, end)
            else:
                frame = _load_local_fixture(ticker, local_dir)

            frame = _standardize_ohlcv(frame)
            if not frame.empty:
                break

        if frame.empty:
            print(f"⚠️ No market data retrieved for {ticker}.")
            continue

        frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
        frame.columns = pd.MultiIndex.from_arrays(
            [[ticker] * len(frame.columns), frame.columns], names=["Ticker", "Field"]
        )
        all_dfs.append(frame)

        print(f"✅ {ticker}: {len(frame)} rows loaded after harmonisation.")

    if not all_dfs:
        raise RuntimeError("No market data available from any source.")

    combined = pd.concat(all_dfs, axis=1, join="outer").sort_index()
    combined = combined.loc[(combined.index >= start_ts) & (combined.index <= end_ts)]
    combined.index.name = "Date"

    print(f"📊 Final dataset shape: {combined.shape[0]} rows, {len(all_dfs)} tickers with data.")
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
