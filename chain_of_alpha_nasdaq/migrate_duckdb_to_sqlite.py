# migrate_duckdb_to_sqlite.py
"""
Enhanced migration helper for Chain of Alpha NASDAQ:
- Validates schemas against the SQLiteStore expected structure.
- Supports incremental migration (skips rows already present).
"""

import os
import duckdb
import sqlite3
import pandas as pd

DUCKDB_PATH = "data/market.duckdb"
SQLITE_PATH = "data/alpha_live.sqlite"

# ---------------------------------------------------------------------
# Expected schema definition (matches SQLiteStore._init_tables)
# ---------------------------------------------------------------------
EXPECTED_SCHEMAS = {
    "prices": ["run_id", "ticker", "date", "open", "high", "low", "close", "volume"],
    "signals": ["run_id", "ticker", "date", "factor_name", "signal"],
    "trades": ["run_id", "ticker", "signal", "action", "weight"],
    "paper_trades": ["run_id", "trade_id", "ticker", "action", "quantity", "price", "timestamp"],
}

# ---------------------------------------------------------------------
def validate_schema(sqlite_con):
    """
    Verify that all expected tables exist with matching columns.
    """
    cur = sqlite_con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = {r[0] for r in cur.fetchall()}

    missing = set(EXPECTED_SCHEMAS.keys()) - existing_tables
    if missing:
        print(f"⚠️ Missing tables in SQLite: {missing} — they will be created automatically.")
        for t in missing:
            if t == "paper_trades":
                sqlite_con.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    run_id TEXT,
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    timestamp TEXT
                );
                """)
        sqlite_con.commit()

    # Column validation
    for table, expected_cols in EXPECTED_SCHEMAS.items():
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1;", sqlite_con)
            current_cols = list(df.columns)
            missing_cols = set(expected_cols) - set(current_cols)
            if missing_cols:
                print(f"⚠️ Table '{table}' missing columns: {missing_cols}")
        except Exception as e:
            print(f"⚠️ Skipping validation for '{table}': {e}")


# ---------------------------------------------------------------------
def migrate_duckdb_to_sqlite(duckdb_path=DUCKDB_PATH, sqlite_path=SQLITE_PATH, incremental=True):
    if not os.path.exists(duckdb_path):
        raise FileNotFoundError(f"❌ DuckDB file not found at {duckdb_path}")

    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)

    print(f"🚀 Starting migration from {duckdb_path} → {sqlite_path}")
    con_duck = duckdb.connect(duckdb_path)
    con_sqlite = sqlite3.connect(sqlite_path)

    validate_schema(con_sqlite)

    duck_tables = [t[0] for t in con_duck.execute("SHOW TABLES").fetchall()]
    print(f"📋 Found {len(duck_tables)} DuckDB tables: {duck_tables}")

    for table in duck_tables:
        if table not in EXPECTED_SCHEMAS:
            print(f"⏭️ Skipping unrecognized table '{table}'")
            continue

        print(f"\n🔄 Migrating table: {table}")
        df_duck = con_duck.execute(f"SELECT * FROM {table}").fetchdf()
        df_duck.columns = [c.lower() for c in df_duck.columns]

        if incremental:
            # Try to detect overlap by the 'date' column if present
            sqlite_cols = pd.read_sql_query(f"PRAGMA table_info({table});", con_sqlite)["name"].tolist()
            if "date" in df_duck.columns and "date" in sqlite_cols:
                try:
                    last_date_row = con_sqlite.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
                    if last_date_row:
                        before = len(df_duck)
                        df_duck = df_duck[df_duck["date"] > last_date_row]
                        print(f"🕐 Incremental mode: {before - len(df_duck)} old rows skipped.")
                except Exception as e:
                    print(f"⚠️ Incremental mode fallback for {table}: {e}")

        if df_duck.empty:
            print(f"⏹️ No new rows to migrate for {table}")
            continue

        # Align columns with SQLite schema (drop extras)
        df_duck = df_duck[[c for c in df_duck.columns if c in EXPECTED_SCHEMAS[table]]]
        for col in EXPECTED_SCHEMAS[table]:
            if col not in df_duck.columns:
                df_duck[col] = None

        df_duck.to_sql(table, con_sqlite, if_exists="append", index=False)
        print(f"✅ {len(df_duck)} new rows appended to {table}")

    con_sqlite.commit()
    con_duck.close()
    con_sqlite.close()
    print("\n🎉 Migration completed successfully!")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    migrate_duckdb_to_sqlite()
