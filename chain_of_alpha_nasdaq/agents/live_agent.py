import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from chain_of_alpha_nasdaq import data_handler
from chain_of_alpha_nasdaq.sqlite_store import SQLiteStore as DuckDBStore
from chain_of_alpha_nasdaq.alpha_db import AlphaDB
from chain_of_alpha_nasdaq.portfolio import Portfolio
from chain_of_alpha_nasdaq.broker.paper_broker import PaperBroker


# --------------------------------------------------------
# Config
# --------------------------------------------------------
TICKER_FILE = "data/nasdaq_tickers.parquet"
DEFAULT_TICKER_LIMIT = 100
LOOKBACK_DAYS = 90


# --------------------------------------------------------
# LiveAlphaAgent
# --------------------------------------------------------
class LiveAlphaAgent:
    def __init__(
        self,
        db_path: str = "alphas.db",
        duck_path: str = "market.duckdb",
        ticker_limit: int = DEFAULT_TICKER_LIMIT,
        tickers: list[str] | None = None,
    ):
        self.db = AlphaDB(db_path)
        self.db_store = DuckDBStore(duck_path)
        self.portfolio = Portfolio(top_k=20)
        self.ticker_limit = ticker_limit
        self.tickers = tickers or self._load_tickers()
        self.broker = self.MockBroker()

    # --------------------------------------------------------
    # Load NASDAQ tickers
    # --------------------------------------------------------
    def _load_tickers(self):
        if not os.path.exists(TICKER_FILE):
            print(f"⚠️ {TICKER_FILE} not found. Please run utils/ticker_fetcher.py first.")
            return ["AAPL", "MSFT"]

        df = pd.read_parquet(TICKER_FILE)
        df = df[df["active"] == True].sort_values("ticker").head(self.ticker_limit)
        tickers = df["ticker"].tolist()
        print(f"✅ Loaded {len(tickers)} active tickers from {TICKER_FILE}")
        return tickers

    # --------------------------------------------------------
    # Step 1: Fetch recent OHLCV data
    # --------------------------------------------------------
    def fetch_latest_data(self, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
        end = datetime.now(UTC).date()
        start = end - timedelta(days=lookback_days)
        print(f"📡 Fetching data from {start} to {end} ...")

        df = data_handler.fetch_data(self.tickers, str(start), str(end))
        print(f"✅ Data shape: {df.shape}")
        return df

    # --------------------------------------------------------
    # Step 2: Load active factors from DB
    # --------------------------------------------------------
    def load_active_factors(self) -> dict:
        cur = self.db.conn.execute("SELECT name, formula FROM factors WHERE active = 1")
        rows = cur.fetchall()
        factors = {r[0]: r[1] for r in rows}
        print(f"🧮 Loaded {len(factors)} active factors from DB.")
        return factors

    # --------------------------------------------------------
    # Step 3: Compute factor signals (with QA)
    # --------------------------------------------------------
    def compute_signals(self, data: pd.DataFrame, factors: dict[str, str]) -> pd.DataFrame:
        """Compute per-ticker factor signals with QA and NaN filtering."""
        data.columns = [c.strip().lower() for c in data.columns]
        print("🔍 DEBUG compute_signals() columns:", data.columns[:15].tolist())

        valid_fields = ["open", "high", "low", "close", "volume"]
        tickers = sorted({c.split("_")[0] for c in data.columns if c.endswith("_close")})
        if not tickers:
            print("⚠️ No tickers detected in data — check pivot step.")
            return pd.DataFrame()

        detailed = {}
        for name, formula in factors.items():
            formula_clean = formula.lower().strip()
            print(f"🧮 Evaluating factor '{name}' → {formula_clean}")

            for t in tickers:
                local = {f: data[f"{t}_{f}"] for f in valid_fields if f"{t}_{f}" in data.columns}
                if len(local) < 2:
                    continue

                try:
                    s = pd.eval(formula_clean, local_dict=local, engine="numexpr")
                except Exception as e:
                    print(f"⚠️ Factor {name} failed for {t}: {e}")
                    continue

                detailed[f"{t}_{name}"] = s

        if not detailed:
            print("⚠️ No valid factor signals computed (check formulas or missing fields).")
            return pd.DataFrame(index=data.index, columns=tickers)

        detailed_df = pd.DataFrame(detailed, index=data.index)

        # Combine signals per ticker
        combined = pd.DataFrame(index=detailed_df.index, columns=tickers, dtype=float)
        for t in tickers:
            t_cols = [c for c in detailed_df.columns if c.startswith(f"{t}_")]
            if t_cols:
                combined[t] = detailed_df[t_cols].mean(axis=1)
            else:
                combined[t] = np.nan

        # -----------------------------
        # QA Filtering: drop NaN & flat
        # -----------------------------
        valid_tickers = []
        for t in tickers:
            series = combined[t]
            if series.isna().all():
                print(f"⚠️ Ticker {t} dropped (all NaN values).")
                continue
            if series.nunique(dropna=True) <= 1:
                print(f"⚠️ Ticker {t} dropped (constant signal).")
                continue
            valid_tickers.append(t)

        combined = combined[valid_tickers]
        if combined.empty:
            print("⚠️ All signals filtered out during QA — check data alignment.")
        else:
            print(f"✅ {len(valid_tickers)} tickers passed QA filtering.")

        # Drop all-zero rows
        combined = combined.loc[(combined.abs().sum(axis=1) != 0)]
        print(f"✅ Computed signals for {len(valid_tickers)} tickers × {len(factors)} factors")
        return combined

    # --------------------------------------------------------
    # Step 4: Build trade table
    # --------------------------------------------------------
    def generate_trade_table(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert latest signal snapshot to trades."""
        if signals is None or signals.empty:
            return pd.DataFrame(columns=["Ticker", "Signal", "Action", "Weight"])

        latest = signals.iloc[-1]
        trades = []
        for ticker in latest.index:
            signal_val = float(latest.get(ticker, 0.0))
            if np.isnan(signal_val):
                continue
            if signal_val > 0.5:
                action = "BUY"
            elif signal_val < -0.5:
                action = "SELL"
            else:
                action = "HOLD"
            weight = round(abs(signal_val), 4)
            trades.append({"Ticker": ticker, "Signal": signal_val, "Action": action, "Weight": weight})

        if not trades:
            print("⚠️ No actionable trades generated.")
            return pd.DataFrame(columns=["Ticker", "Signal", "Action", "Weight"])

        trade_table = pd.DataFrame(trades).sort_values("Weight", ascending=False).reset_index(drop=True)
        print("\n📊 Trade Table (top 25 by |signal|):")
        print(trade_table.head(25).to_string(index=False))
        return trade_table

    # --------------------------------------------------------
    # Step 5: Run one live cycle
    # --------------------------------------------------------
    def run_once(self):
        print("🚀 Running LiveAlphaAgent cycle...")
        data = self.fetch_latest_data()
        factors = self.load_active_factors()
        signals = self.compute_signals(data, factors)
        trades = self.generate_trade_table(signals)
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Persist
        self.db_store.insert_prices(data, run_id)
        self.db_store.insert_signals(signals, run_id)
        self.db_store.insert_trades(trades, run_id)
        self.db_store.trim_old_data(days=60)

        # Execute
        self.broker.execute_trades(trades, data, run_id)
        print(f"🦆 Data persisted to alpha_live.duckdb with run_id={run_id}")
        print("✅ Cycle complete.")
        return trades

    # --------------------------------------------------------
    # Mock broker (for testing)
    # --------------------------------------------------------
    class MockBroker:
        """Simulated broker for paper trading."""
        def execute_trades(self, trades, data, run_id):
            print(f"💰 Executing {len(trades)} simulated trades for run_id={run_id}")
            for t in trades.itertuples():
                print(f"   → {t.Ticker}: {t.Action} ({t.Weight:.2f})")


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------
if __name__ == "__main__":
    agent = LiveAlphaAgent(tickers=["AAPL", "MSFT"])
    agent.run_once()
