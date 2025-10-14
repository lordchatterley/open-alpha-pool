# chain_of_alpha_nasdaq/analytics/performance_tracker.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PerformanceTracker:
    """
    Evaluates paper trading results using trade history and price data.
    """

    def __init__(self, store, run_id: str, initial_cash: float = 100_000.0):
        self.store = store
        self.run_id = run_id
        self.initial_cash = initial_cash

    # ------------------------------------------------------------------
    def load_data(self):
        """
        Loads paper trades and prices for this run_id from SQLite.
        """
        trades = pd.read_sql_query(
            f"SELECT * FROM paper_trades WHERE run_id = '{self.run_id}' ORDER BY timestamp ASC",
            self.store.conn,
            parse_dates=["timestamp"]
        )
        prices = pd.read_sql_query(
            f"SELECT ticker, date, close FROM prices WHERE run_id = '{self.run_id}'",
            self.store.conn,
            parse_dates=["date"]
        )
        return trades, prices

    # ------------------------------------------------------------------
    def build_equity_curve(self):
        """
        Reconstructs equity over time using executed paper trades.
        """
        trades, prices = self.load_data()
        if trades.empty:
            print("⚠️ No trades found for this run_id.")
            return None

        equity = self.initial_cash
        positions = {}
        curve = []

        # Convert to datetime index for time-based equity
        trades = trades.sort_values("timestamp").reset_index(drop=True)

        for _, row in trades.iterrows():
            tkr, act, qty, price, ts = row["ticker"], row["action"], row["quantity"], row["price"], row["timestamp"]

            if act == "BUY":
                cost = qty * price
                equity -= cost
                positions[tkr] = positions.get(tkr, 0) + qty
            elif act == "SELL":
                equity += qty * price
                positions[tkr] = max(0, positions.get(tkr, 0) - qty)

            # Compute holdings value at that timestamp
            value = 0.0
            for t, q in positions.items():
                recent = prices.loc[(prices["ticker"] == t) & (prices["date"] <= ts)]
                if not recent.empty:
                    value += q * recent.iloc[-1]["close"]

            total_equity = equity + value
            curve.append({"timestamp": ts, "equity": total_equity})

        eq_df = pd.DataFrame(curve).set_index("timestamp")
        eq_df["returns"] = eq_df["equity"].pct_change().fillna(0)
        return eq_df

    # ------------------------------------------------------------------
    def compute_metrics(self, eq_df: pd.DataFrame):
        """
        Computes common performance metrics from an equity curve.
        """
        if eq_df is None or eq_df.empty:
            return {}

        daily_returns = eq_df["returns"]
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        max_dd = (eq_df["equity"].cummax() - eq_df["equity"]).max()
        total_return = (eq_df["equity"].iloc[-1] / self.initial_cash) - 1

        metrics = {
            "final_equity": eq_df["equity"].iloc[-1],
            "total_return_%": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_$": max_dd,
        }

        return metrics

    # ------------------------------------------------------------------
    def plot_equity(self, eq_df: pd.DataFrame, show: bool = True):
        """
        Plots the equity curve with Matplotlib.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(eq_df.index, eq_df["equity"], label="Equity", linewidth=2)
        plt.title("📈 Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        return plt

    # ------------------------------------------------------------------
    def run_full_report(self):
        """
        Convenience function: builds equity curve, computes metrics, prints summary.
        """
        eq_df = self.build_equity_curve()
        if eq_df is None:
            return None

        metrics = self.compute_metrics(eq_df)
        print("\n📊 PERFORMANCE SUMMARY")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:,.2f}")

        return eq_df, metrics
