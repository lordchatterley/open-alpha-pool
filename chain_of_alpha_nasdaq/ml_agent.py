# chain_of_alpha_nasdaq/agents/ml_agent.py
import numpy as np
import pandas as pd
from datetime import datetime, UTC, timedelta

from chain_of_alpha_nasdaq.sqlite_store import SQLiteStore
from chain_of_alpha_nasdaq.analytics.features import features_from_factors, cheap_price_features
from chain_of_alpha_nasdaq.analytics.labels import make_forward_returns
from chain_of_alpha_nasdaq.analytics.model import assemble_panel, train_lgbm, save_model_sqlite, predict_panel

class MLAlphaAgent:
    def __init__(self, store: SQLiteStore, horizon: int = 10):
        self.store = store
        self.horizon = horizon

    def _load_prices_recent(self, days: int = 120) -> pd.DataFrame:
        # Pull prices from SQLite (long-form) — you already write here.
        q = f"""
        SELECT date, ticker, open, high, low, close, volume
        FROM prices
        WHERE date >= date('now','-{days} days')
        """
        df = pd.read_sql_query(q, self.store.conn)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_factor_signals(self, days: int = 120) -> dict[str, pd.DataFrame]:
        # Use stored signals (model_signal or factor-specific) if available; else return empty dict
        try:
            sig = pd.read_sql_query(f"""
                SELECT date, ticker, factor_name, signal
                FROM signals
                WHERE date >= date('now','-{days} days')
            """, self.store.conn)
        except Exception:
            return {}

        if sig.empty:
            return {}

        sig["date"] = pd.to_datetime(sig["date"])
        out = {}
        for name, g in sig.groupby("factor_name"):
            w = g.pivot(index="date", columns="ticker", values="signal").sort_index()
            out[name] = w
        return out

    def fit_and_predict(self, run_id: str) -> pd.DataFrame:
        prices = self._load_prices_recent(days=180)
        if prices.empty:
            print("⚠️ No prices found to build features/labels.")
            return pd.DataFrame()

        # Labels (forward returns, paper uses h=10 by default)
        labels = make_forward_returns(prices, horizon=self.horizon)

        # Factor features (if any) + cheap technicals to stabilize training
        factor_wide = self._load_factor_signals(days=180)
        feats_factors = features_from_factors(factor_wide)
        feats_prices  = cheap_price_features(prices)
        feats = pd.concat([feats_factors, feats_prices], ignore_index=True).dropna(subset=["value"])

        if feats.empty:
            print("⚠️ No features available; aborting ML step.")
            return pd.DataFrame()

        # Assemble panel
        X, y, dates, tickers = assemble_panel(feats, labels)
        if X.empty:
            print("⚠️ Feature/label join is empty; check coverage and horizons.")
            return pd.DataFrame()

        # Train LGBM using time split
        model = train_lgbm(X, y, dates)
        model_id, path = save_model_sqlite(self.store, model, horizon=self.horizon)

        # Predict for the latest date
        latest_date = dates.max()
        mask_latest = dates == latest_date
        X_latest = X[mask_latest]
        tick_latest = tickers[mask_latest]
        scores = predict_panel(model, X_latest)

        pred = pd.DataFrame({
            "run_id": run_id,
            "date": latest_date,
            "ticker": tick_latest.values,
            "horizon": self.horizon,
            "score": scores,
            "model_id": model_id
        })

        pred.to_sql("predictions", self.store.conn, if_exists="append", index=False)
        return pred

    def trades_from_predictions(self, preds: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        if preds is None or preds.empty:
            return pd.DataFrame(columns=["Ticker","Signal","Action","Weight"])
        # Long-only top_k by score
        df = preds.sort_values("score", ascending=False).head(top_k)
        df["Ticker"] = df["ticker"]
        df["Signal"] = df["score"]
        df["Action"] = "BUY"
        # normalize weights to sum 1
        w = df["score"].clip(lower=0)
        w = w / (w.sum() + 1e-9)
        df["Weight"] = w
        return df[["Ticker","Signal","Action","Weight"]].reset_index(drop=True)
