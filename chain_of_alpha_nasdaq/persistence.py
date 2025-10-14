import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/factors.db")
LOG_PATH = Path("data/logs/")
LOG_PATH.mkdir(parents=True, exist_ok=True)


class Persistence:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS factors (
            name TEXT PRIMARY KEY,
            expression TEXT,
            metrics TEXT,
            status TEXT,
            created_at TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP,
            portfolio_metrics TEXT
        )
        """)
        conn.commit()
        conn.close()

    def save_factor(self, name, expression, metrics, status="effective"):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO factors (name, expression, metrics, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (name, expression, json.dumps(metrics), status, datetime.utcnow()))
        conn.commit()
        conn.close()

    def load_factors(self, status="effective"):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, expression, metrics FROM factors WHERE status=?", (status,))
        rows = cur.fetchall()
        conn.close()
        return {name: {"expression": expr, "metrics": json.loads(metrics)} for name, expr, metrics in rows}

    def save_run(self, portfolio_metrics):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO runs (run_date, portfolio_metrics) VALUES (?, ?)",
                    (datetime.utcnow(), json.dumps(portfolio_metrics)))
        conn.commit()
        conn.close()

    def export_log(self, run_id, data):
        """Save raw JSON log for debugging/auditing."""
        log_file = LOG_PATH / f"run_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return log_file
