import os
import math
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

from chain_of_alpha_nasdaq.broker.paper_broker import PaperBroker
from chain_of_alpha_nasdaq.analytics.performance_tracker import PerformanceTracker
from chain_of_alpha_nasdaq import data_handler
from chain_of_alpha_nasdaq.agents.live_agent import LiveAlphaAgent
from chain_of_alpha_nasdaq.utils.html_reporter import save_html_report
from chain_of_alpha_nasdaq.sqlite_store import SQLiteStore

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
TICKER_FILE = "data/nasdaq_tickers.parquet"
CACHE_DIR = Path("data/cache/ohlcv")
LOOKBACK_DAYS = 90
BATCH_SIZE = 200
MAX_WORKERS = 4
SLEEP_BETWEEN_BATCHES = 2
RESUME_LOG = CACHE_DIR / "completed_batches.txt"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def fetch_batch(tickers, start, end):
    """Fetches a single batch of tickers and caches to Parquet."""
    batch_id = f"{tickers[0]}_{tickers[-1]}_{start}_{end}".replace("/", "-")
    cache_path = CACHE_DIR / f"batch_{batch_id}.parquet"

    if cache_path.exists():
        print(f"📁 Using cached batch {cache_path.name}")
        return pd.read_parquet(cache_path)

    try:
        print(f"📡 Fetching {len(tickers)} tickers ({tickers[0]} → {tickers[-1]})")
        df = data_handler.fetch_data(tickers, start, end)
        if df is None or df.empty:
            print(f"⚠️ Empty data for batch {batch_id}")
            return pd.DataFrame()
        df.to_parquet(cache_path)
        with open(RESUME_LOG, "a") as log:
            log.write(batch_id + "\n")
        print(f"💾 Cached {len(tickers)} tickers → {cache_path.name}")
        return df
    except Exception as e:
        print(f"⚠️ Error fetching batch {batch_id}: {e}")
        return pd.DataFrame()


def get_completed_batches():
    if not RESUME_LOG.exists():
        return set()
    with open(RESUME_LOG) as f:
        return set(line.strip() for line in f if line.strip())


# ----------------------------------------------------------
# Main NASDAQ Scan
# ----------------------------------------------------------
def run_full_scan(limit=None, lookback_days=None, verbose=False):
    if verbose:
        print("🔍 Verbose logging enabled")

    if lookback_days:
        print(f"⏱️ Lookback override: {lookback_days} days")
    else:
        lookback_days = LOOKBACK_DAYS

    print("🚀 Starting multi-threaded NASDAQ scan")

    if not os.path.exists(TICKER_FILE):
        raise FileNotFoundError("⚠️ Run utils/ticker_fetcher.py first!")

    df_tickers = pd.read_parquet(TICKER_FILE)
    tickers = df_tickers[df_tickers["active"] == True]["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]
    print(f"✅ Loaded {len(tickers)} active NASDAQ tickers")

    end = datetime.now(UTC).date()
    start = end - timedelta(days=lookback_days)

    num_batches = math.ceil(len(tickers) / BATCH_SIZE)
    completed = get_completed_batches()
    print(f"🧩 Total batches: {num_batches}, already completed: {len(completed)}")

    all_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(num_batches):
            batch = tickers[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            if not batch:
                continue
            batch_id = f"{batch[0]}_{batch[-1]}_{start}_{end}".replace("/", "-")
            if batch_id in completed:
                print(f"⏭️ Skipping completed batch {batch[0]} → {batch[-1]}")
                continue
            futures.append(executor.submit(fetch_batch, batch, str(start), str(end)))
            time.sleep(SLEEP_BETWEEN_BATCHES)

        for i, f in enumerate(as_completed(futures), 1):
            df = f.result()
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"✅ Completed batch {i}/{num_batches} ({df.shape[0]} rows)")
            else:
                print(f"⚠️ Skipped batch {i}/{num_batches} (no data)")

    # Combine all data
    if not all_data:
        cached = list(CACHE_DIR.glob("batch_*.parquet"))
        if cached:
            print(f"📁 Using {len(cached)} cached batches from previous runs.")
            all_data = [pd.read_parquet(f) for f in cached]
        else:
            raise RuntimeError("No data fetched.")

        # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined.columns = [c.strip().lower() for c in combined.columns]

    print(f"📊 Combined NASDAQ dataset shape: {combined.shape}")
    print(f"🧩 Combined columns: {combined.columns.tolist()}")

    # Ensure expected schema
    rename_map = {}
    if "ticker" not in combined.columns:
        for col in combined.columns:
            if col.lower() in ["symbol", "security", "asset"]:
                rename_map[col] = "ticker"
    if "date" not in combined.columns:
        for col in combined.columns:
            if col.lower() in ["datetime", "time", "timestamp"]:
                rename_map[col] = "date"
    if rename_map:
        combined = combined.rename(columns=rename_map)
        print(f"🧭 Renamed columns for compatibility: {rename_map}")

    # Detect long vs wide format
    if "ticker" not in combined.columns or "date" not in combined.columns:
    # Check if it looks like wide-format OHLCV
        sample_cols = [c.lower() for c in combined.columns[:10]]
        if any("_close" in c or "_open" in c for c in sample_cols):
            print("📐 Detected wide-format OHLCV — skipping (date,ticker) validation.")
            # create synthetic date column if missing
            if "date" not in combined.columns:
                combined["date"] = pd.date_range(
                    end=pd.Timestamp.now(), periods=len(combined)
                )
        else:
            raise RuntimeError(
                f"Missing required columns for long-format data: {combined.columns.tolist()}"
            )

    # Connect LiveAlphaAgent
    agent = LiveAlphaAgent(ticker_limit=len(tickers))
    agent.tickers = tickers
    factors = agent.load_active_factors()

    # Normalize factor formulas to lowercase so they match column names
    factors = {k: v.lower() for k, v in factors.items()}
    print(f"🔧 Normalized factor formulas: {factors}")


    # ----------------------------------------------------------
    # Pivot for signals (all OHLCV)
    # ----------------------------------------------------------
    print("🔄 Pivoting copy to wide for agent signals (all OHLCV fields)...")
    # Drop duplicate (date, ticker) rows if in long format
    if "ticker" in combined.columns and "date" in combined.columns:
        if combined.duplicated(subset=["date", "ticker"]).any():
            n_dupes = combined.duplicated(subset=["date", "ticker"]).sum()
            print(f"⚠️ Found {n_dupes} duplicate (date, ticker) rows — dropping duplicates.")
            combined = combined.drop_duplicates(subset=["date", "ticker"])
    else:
        print("📐 Wide-format OHLCV detected — skipping duplicate (date, ticker) check.")


    # ----------------------------------------------------------
    # Normalize case and detect format
    # ----------------------------------------------------------
    combined.columns = [c.strip().lower() for c in combined.columns]
    print(f"🧩 Combined columns (lowercase): {combined.columns[:10].tolist()}")

    # Wide-format detection
    wide_detected = any("_close" in c or "_open" in c for c in combined.columns)
    if wide_detected:
        print("📐 Detected wide-format OHLCV — skipping pivot step.")
        combined_wide = combined.copy()
        if "date" not in combined_wide.columns:
            combined_wide["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(combined_wide))
    else:
        print("🔄 Detected long-format OHLCV — pivoting for agent signals.")
        # Drop potential duplicates and pivot
        if "ticker" not in combined.columns or "date" not in combined.columns:
            raise RuntimeError(f"Missing 'ticker' or 'date' for pivot: {combined.columns.tolist()}")
        if combined.duplicated(subset=["date", "ticker"]).any():
            n_dupes = combined.duplicated(subset=["date", "ticker"]).sum()
            print(f"⚠️ Found {n_dupes} duplicate (date,ticker) rows — dropping duplicates.")
            combined = combined.drop_duplicates(subset=["date", "ticker"])
        combined_wide = (
            combined.pivot(index="date", columns="ticker")[["open", "high", "low", "close", "volume"]]
            .sort_index(axis=1)
            .reset_index()
        )

    print(f"📊 combined_wide shape: {combined_wide.shape}")


    # ----------------------------------------------------------
    # Compute signals
    # ----------------------------------------------------------
    signals = agent.compute_signals(combined_wide, factors)
    trades = agent.generate_trade_table(signals)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    store = agent.db_store

    # Ensure 'combined' is long-form before inserting
    if not {"ticker", "date"}.issubset(combined.columns):
        print("🔄 Converting wide-format combined data to long-form for DB storage...")
        # melt only OHLCV columns
        value_vars = [c for c in combined.columns if "_" in c and any(x in c for x in ["open", "high", "low", "close", "volume"])]
        long_frames = []
        for field in ["open", "high", "low", "close", "volume"]:
            cols = [c for c in value_vars if c.endswith(f"_{field}")]
            if not cols:
                continue
            tmp = combined.melt(id_vars=["date"], value_vars=cols, var_name="ticker_field", value_name=field)
            tmp["ticker"] = tmp["ticker_field"].str.replace(f"_{field}", "", regex=False)
            tmp = tmp.drop(columns=["ticker_field"])
            long_frames.append(tmp)
        combined_long = long_frames[0]
        for f in long_frames[1:]:
            combined_long = combined_long.merge(f, on=["date", "ticker"], how="outer")
        combined = combined_long[["ticker", "date", "open", "high", "low", "close", "volume"]]
        print(f"✅ Converted to long-form shape: {combined.shape}")


    # ----------------------------------------------------------
    # Persist data
    # ----------------------------------------------------------
    store.insert_prices(combined, run_id)

    if signals is not None and not signals.empty:
        signals["date"] = pd.Timestamp.now(UTC)
        store.insert_signals(signals, run_id)
    else:
        print("⚠️ No valid signals to insert.")

    if trades is not None and not trades.empty:
        store.insert_trades(trades, run_id)
    else:
        print("⚠️ No trades generated — skipping insert_trades().")

    # ----------------------------------------------------------
    # Simulated paper broker
    # ----------------------------------------------------------
    broker = PaperBroker(store, run_id)
    broker.summary()

    # ----------------------------------------------------------
    # Generate reports
    # ----------------------------------------------------------
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"nasdaq_scan_{run_id}.csv"
    html_path = reports_dir / f"nasdaq_scan_{run_id}.html"
    trades.to_csv(csv_path, index=False)
    save_html_report(run_id, trades, str(html_path))

    print(f"\n🧾 CSV report → {csv_path}")
    print(f"🌐 HTML report → {html_path}")

    # ----------------------------------------------------------
    # Performance summary
    # ----------------------------------------------------------
    print("\n🏁 Running post-scan performance report...")
    try:
        tracker = PerformanceTracker(store, run_id)
        tracker.summarize_run()
    except Exception as e:
        print(f"⚠️ Post-scan report failed: {e}")



    print(f"\n✅ NASDAQ scan complete. run_id={run_id}")
    print(f"🧱 Data persisted to SQLite (data/alpha_live.sqlite)")
    return trades


# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NASDAQ scan with resume and reporting.")
    parser.add_argument("--limit", type=int, help="Limit number of tickers", default=None)
    parser.add_argument("--since", type=str, help="Lookback period in days (e.g. 30d)", default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    lookback_days = None
    if args.since and args.since.endswith("d"):
        lookback_days = int(args.since[:-1])

    try:
        run_full_scan(limit=args.limit, lookback_days=lookback_days, verbose=args.verbose)
    except RuntimeError as e:
        print(f"❌ Scan aborted: {e}")
