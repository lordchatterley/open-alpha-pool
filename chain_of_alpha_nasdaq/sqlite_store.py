import os
import sqlite3
import pandas as pd
from datetime import datetime


class SQLiteStore:
    def __init__(self, db_path: str = "data/alpha_live.sqlite"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
        print(f"🧱 Connected to SQLite → {db_path}")

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            run_id TEXT,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            run_id TEXT,
            ticker TEXT,
            date TEXT,
            factor_name TEXT,
            signal REAL
        );

        CREATE TABLE IF NOT EXISTS trades (
            run_id TEXT,
            ticker TEXT,
            signal REAL,
            action TEXT,
            weight REAL
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            run_id TEXT,
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            action TEXT,
            quantity REAL,
            price REAL,
            timestamp TEXT
        );
                          
        CREATE TABLE IF NOT EXISTS factors (
            factor_name TEXT PRIMARY KEY,
            formula TEXT,
            active INTEGER DEFAULT 1
        );
                          
        -- added tables for features, labels, models, predictions
        CREATE TABLE IF NOT EXISTS features (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            feature_name TEXT,
            value REAL
        );

        CREATE TABLE IF NOT EXISTS labels (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            horizon INTEGER,
            fwd_return REAL
        );

        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            trained_at TEXT,
            horizon INTEGER,
            params_json TEXT,
            artifact_path TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            horizon INTEGER,
            score REAL,
            model_id TEXT
        );

        """)
        self.conn.commit()

    # --------------------------------------------------------------
    # Prices
    # --------------------------------------------------------------
    def insert_prices(self, df: pd.DataFrame, run_id: str):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Convert wide OHLCV to long if necessary
        if "date" not in df.columns:
            print("📐 Detected wide-format OHLCV — unpivoting by ticker...")
            df_long = []
            for col in df.columns:
                if "_" in col:
                    ticker, field = col.split("_", 1)
                    temp = pd.DataFrame({
                        "date": df.index if "date" not in df.columns else df["date"],
                        "ticker": ticker.upper(),
                        field: df[col],
                    })
                    df_long.append(temp)
            df = pd.concat(df_long)
            df = df.pivot_table(index=["ticker", "date"], values=["open", "high", "low", "close", "volume"]).reset_index()

        df["run_id"] = run_id
        df.to_sql("prices", self.conn, if_exists="append", index=False)
        print(f"🧾 Inserted {len(df)} price rows into SQLite.")

    # --------------------------------------------------------------
    # Signals
    # --------------------------------------------------------------
    def insert_signals(self, df: pd.DataFrame, run_id: str, signal_col: str = "signal"):
        """
        Insert wide-format signals DataFrame into the SQLite 'signals' table.
        Expected format: one column per ticker, plus a 'date' column.
        """

        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure 'date' exists as a column
        if "date" not in df.columns:
            if df.index.name and "date" in df.index.name.lower():
                df = df.reset_index().rename(columns={df.index.name: "date"})
            else:
                print(f"⚠️ insert_signals(): no 'date' column found. Columns = {df.columns.tolist()}")
                return

        # Melt tickers into long format
        # Avoid name conflict with existing columns
        if signal_col in df.columns:
            df = df.rename(columns={signal_col: f"{signal_col}_val"})

        # Melt tickers into long format
        df_long = df.melt(id_vars=["date"], var_name="ticker", value_name=signal_col)

        df_long["run_id"] = run_id
        df_long["factor_name"] = "default"

        # Rename the user-specified signal column to 'signal' for DB consistency
        if signal_col != "signal":
            df_long.rename(columns={signal_col: "signal"}, inplace=True)

        # Insert into DB
        df_long.to_sql("signals", self.conn, if_exists="append", index=False)
        print(f"📈 Inserted {len(df_long)} signal rows into SQLite (as column '{signal_col}').")


    # --------------------------------------------------------------
    # Trades
    # --------------------------------------------------------------
    def insert_trades(self, df: pd.DataFrame, run_id: str):
        df = df.copy()
        df["run_id"] = run_id
        df.to_sql("trades", self.conn, if_exists="append", index=False)
        print(f"💼 Inserted {len(df)} trades into SQLite.")

    # --------------------------------------------------------------
    # Paper Trades
    # --------------------------------------------------------------
    def insert_paper_trade(self, ticker: str, action: str, quantity: float, price: float, run_id: str):
        ts = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO paper_trades (run_id, ticker, action, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, ticker, action, quantity, price, ts))
        self.conn.commit()
        print(f"🧾 Recorded paper trade {action} {quantity} {ticker} @ {price}")

    # --------------------------------------------------------------
    # Query utilities
    # --------------------------------------------------------------
    def get_recent_trades(self, limit: int = 10):
        return pd.read_sql_query(f"SELECT * FROM trades ORDER BY rowid DESC LIMIT {limit}", self.conn)

    def get_recent_paper_trades(self, limit: int = 10):
        return pd.read_sql_query(f"SELECT * FROM paper_trades ORDER BY trade_id DESC LIMIT {limit}", self.conn)

    def close(self):
        self.conn.close()
