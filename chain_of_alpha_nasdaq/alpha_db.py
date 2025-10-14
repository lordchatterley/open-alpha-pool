import sqlite3
from datetime import datetime

class AlphaDB:
    def __init__(self, db_path="alpha.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self._migrate_schema()

    def _create_tables(self):
        """
        Initialize the SQLite schema for factors and metrics.
        """
        # Table: factors
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            formula TEXT,
            created_at DATETIME,
            active INTEGER DEFAULT 1
        )
        """)

        # Table: metrics
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id INTEGER,
            run_id TEXT,
            created_at DATETIME,
            ic REAL,
            rankic REAL,
            icir REAL,
            rankicir REAL,
            sharpe REAL,
            return REAL,
            turnover REAL,
            FOREIGN KEY(factor_id) REFERENCES factors(id)
        )
        """)

        self.conn.commit()


    def _migrate_schema(self):
        """Ensure columns 'active' and 'rank_ic' exist for backward compatibility."""
        cur = self.conn.execute("PRAGMA table_info(factors)")
        factor_cols = [r[1] for r in cur.fetchall()]
        if "active" not in factor_cols:
            self.conn.execute("ALTER TABLE factors ADD COLUMN active INTEGER DEFAULT 1")
            self.conn.commit()

        cur = self.conn.execute("PRAGMA table_info(metrics)")
        metric_cols = [r[1] for r in cur.fetchall()]
        for col, sql_type in [
            ("rank_ic", "REAL"),
            ("ir", "REAL"),
            ("cagr", "REAL"),
            ("max_drawdown", "REAL"),
        ]:
            if col not in metric_cols:
                self.conn.execute(f"ALTER TABLE metrics ADD COLUMN {col} {sql_type}")
                self.conn.commit()

    def insert_factor(self, name, formula):
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO factors (name, formula, created_at, active) VALUES (?, ?, ?, 1)",
            (name, formula, datetime.utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid or self.get_factor_id(name)

    def get_factor_id(self, name):
        cur = self.conn.execute("SELECT id FROM factors WHERE name = ?", (name,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_active_factors(self):
        cur = self.conn.execute("SELECT id, name, formula FROM factors WHERE active = 1")
        return [{"id": r[0], "name": r[1], "formula": r[2]} for r in cur.fetchall()]

    def insert_metrics(self, factor_name=None, run_id=None, metrics=None, **kwargs):
        """
        Insert performance metrics for a given factor into the metrics table.
        Supports both dict-style and keyword-style calls.
        """
        # Handle case where tests call insert_metrics(IC=..., RankIC=..., etc.)
        if metrics is None and kwargs:
            # Convert test’s keyword-style input into metrics dict
            metrics = {k.lower(): v for k, v in kwargs.items()}
        elif metrics is None:
            metrics = {}

        # Normalize keys to lowercase
        metrics = {k.lower(): v for k, v in metrics.items()}

        # Ensure factor name was provided
        if not factor_name:
            raise ValueError("factor_name must be provided")

        # Verify factor exists
        cur = self.conn.execute("SELECT id FROM factors WHERE name = ?", (factor_name,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Factor '{factor_name}' not found in DB")
        factor_id = row[0]

        # Fill defaults for missing metrics
        expected_keys = ["ic", "rankic", "icir", "rankicir", "sharpe", "return", "turnover"]
        norm_metrics = {k: metrics.get(k, None) for k in expected_keys}

        # Insert record
        self.conn.execute("""
            INSERT INTO metrics (
                factor_id, run_id, created_at,
                ic, rankic, icir, rankicir, sharpe, return, turnover
            ) VALUES (?, ?, datetime('now'),
                      ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_id, run_id,
            norm_metrics["ic"], norm_metrics["rankic"], norm_metrics["icir"],
            norm_metrics["rankicir"], norm_metrics["sharpe"], norm_metrics["return"],
            norm_metrics["turnover"]
        ))
        self.conn.commit()



    def get_metrics(self, factor_name):
        """
        Retrieve the most recent performance metrics for a factor.
        Returns a list with one dict, as the test expects.
        """
        cur = self.conn.execute("""
            SELECT m.ic, m.rankic, m.icir, m.rankicir, m.sharpe, m.return, m.turnover
            FROM metrics m
            JOIN factors f ON m.factor_id = f.id
            WHERE f.name = ?
            ORDER BY m.created_at DESC
            LIMIT 1
        """, (factor_name,))
        row = cur.fetchone()
        if row:
            return [{
                "ic": row[0],
                "rank_ic": row[1],
                "icir": row[2],
                "rankicir": row[3],
                "sharpe": row[4],
                "return": row[5],
                "turnover": row[6],
            }]
        return []



    def deactivate_factor(self, name):
        self.conn.execute("UPDATE factors SET active = 0 WHERE name = ?", (name,))
        self.conn.commit()

    # ✅ Alias for compatibility with older tests
    def deprecate_factor(self, name):
        self.deactivate_factor(name)

    def close(self):
        self.conn.close()
