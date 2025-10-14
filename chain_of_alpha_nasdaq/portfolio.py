import numpy as np
import pandas as pd


def construct_portfolio(signals: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """
    Construct a portfolio by selecting top-K assets each day.
    Returns a DataFrame of weights (same shape as signals).
    """
    weights = pd.DataFrame(index=signals.index, columns=signals.columns, dtype=float)

    for date, row in signals.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        k = min(top_k, len(ranked))
        if k == 0:
            continue

        top_assets = ranked.head(k)
        w = pd.Series(0.0, index=signals.columns)
        w.loc[top_assets.index] = 1.0 / float(k)  # equal weights
        weights.loc[date] = w

    # Fill forward for missing days and ensure consistent dtype
    weights = (
        weights.ffill().fillna(0.0).infer_objects(copy=False)
    )

    return weights


def backtest_portfolio(prices: pd.DataFrame, signals: pd.DataFrame, top_k: int = 50):
    """
    Run a simple backtest using top-K signals.
    Returns (equity_curve, metrics_dict)
    """
    # --- Defensive shape handling ---
    if isinstance(prices, pd.Series):
        prices = prices.to_frame("price")

    # If columns are datetime (instead of tickers), transpose
    if np.issubdtype(np.array(prices.columns).dtype, np.datetime64):
        prices = prices.T

    # If index is datetime but data are actually tickers, transpose again
    if prices.shape[0] < prices.shape[1] and np.issubdtype(prices.index.dtype, np.datetime64):
        pass  # good shape already (dates as index)
    elif np.issubdtype(prices.columns.dtype, np.datetime64):
        prices = prices.T

    # Coerce all values to numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # --- Proceed with backtest ---
    weights = construct_portfolio(signals, top_k=top_k)

    # Align shapes
    prices, weights = prices.align(weights, join="inner", axis=0)

    # Compute daily returns per asset
    returns = prices.pct_change().fillna(0.0)
    returns, weights = returns.align(weights, join="inner", axis=0)

    # Portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Cumulative performance
    equity_curve = (1 + portfolio_returns).cumprod()

    # Compute key metrics
    if len(equity_curve) > 1:
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        cagr = float((equity_curve.iloc[-1] ** (252 / len(equity_curve))) - 1)
        sharpe = float(
            portfolio_returns.mean() / (portfolio_returns.std() + 1e-9) * np.sqrt(252)
        )
        max_dd = float(
            ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max()
        )
    else:
        total_return, cagr, sharpe, max_dd = [np.nan] * 4

    metrics = {
        "Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
    }

    return equity_curve, metrics



def performance_metrics(backtest_output):
    """
    Extract performance metrics from the result of backtest_portfolio().
    Accepts either (equity_curve, metrics) or metrics directly.
    """
    if isinstance(backtest_output, tuple) and len(backtest_output) == 2:
        _, metrics = backtest_output
    else:
        metrics = backtest_output

    return metrics
# ... (all your imports, functions, and the Portfolio class) ...

class Portfolio:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def run_backtest(self, signal_df: pd.DataFrame, prices: pd.DataFrame):
        """
        Run a simple backtest for given signals and prices.
        Returns (equity_curve, metrics).
        """
        if signal_df is None or prices is None or signal_df.empty or prices.empty:
            print("⚠️ No data or signals provided to Portfolio.run_backtest().")
            return None

        equity_curve, metrics = backtest_portfolio(prices, signal_df, top_k=self.top_k)

        print("📊 Portfolio Backtest Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

        return equity_curve, metrics

    def construct_portfolio(self, signals: pd.DataFrame, prices: pd.DataFrame):
        """
        Construct and return weights DataFrame (wrapper).
        """
        return construct_portfolio(signals, top_k=self.top_k)

    def performance_metrics(self, backtest_results):
        """
        Extract and return performance metrics.
        """
        return performance_metrics(backtest_results)


# ✅ Backward compatibility alias — must come *after* the class definition
PortfolioConstructor = Portfolio


