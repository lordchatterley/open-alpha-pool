import pandas as pd
import numpy as np
from chain_of_alpha_nasdaq import utils


class Backtester:
    def __init__(self, prices: pd.DataFrame, signals: pd.DataFrame, top_k=50, cost=0.001):
        """
        prices: DataFrame (Date x Ticker) of adjusted close prices
        signals: DataFrame (Date x Ticker) of factor values (higher = more bullish)
        top_k: number of stocks to hold each rebalance
        cost: transaction cost per trade (fraction)
        """
        self.prices = prices
        self.signals = signals.reindex_like(prices)
        self.top_k = top_k
        self.cost = cost

    def compute_forward_returns(self, horizon=1):
        """Compute forward returns over given horizon."""
        return self.prices.pct_change(horizon).shift(-horizon)

    def run(self, horizon=1):
        """
        Run backtest and compute performance metrics.
        Returns dict with returns and evaluation metrics.
        """
        forward_returns = self.compute_forward_returns(horizon)
        portfolio_returns = []

        ic_list, rankic_list = [], []

        for date in self.signals.index[:-horizon]:
            scores = self.signals.loc[date].dropna()
            rets = forward_returns.loc[date].dropna()
            common = scores.index.intersection(rets.index)
            if len(common) < self.top_k:
                continue

            scores = scores[common]
            rets = rets[common]

            # IC and RankIC via utils
            ic_list.append(utils.compute_ic(scores, rets))
            rankic_list.append(utils.compute_rankic(scores, rets))

            # Portfolio = top_k equally weighted
            top_assets = scores.sort_values(ascending=False).head(self.top_k).index
            daily_ret = rets[top_assets].mean() - self.cost
            portfolio_returns.append(daily_ret)

        returns_series = pd.Series(portfolio_returns, index=self.signals.index[: len(portfolio_returns)])

        # Aggregate metrics
        metrics = {
            "IC": np.nanmean(ic_list) if ic_list else np.nan,
            "RankIC": np.nanmean(rankic_list) if rankic_list else np.nan,
            "ICIR": np.nanmean(ic_list) / np.nanstd(ic_list) if np.nanstd(ic_list) else np.nan,
            "RankICIR": np.nanmean(rankic_list) / np.nanstd(rankic_list) if np.nanstd(rankic_list) else np.nan,
        }
        metrics.update(utils.compute_portfolio_metrics(returns_series))

        return {"returns": returns_series, "metrics": metrics}