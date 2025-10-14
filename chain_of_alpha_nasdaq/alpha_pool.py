import numpy as np
from chain_of_alpha_nasdaq import utils


class AlphaPool:
    def __init__(self, thresholds=None, max_size=100):
        """
        Manage effective and deprecated alpha factors.
        """
        self.effective = {}   # {name: {"expression": str, "metrics": dict}}
        self.deprecated = {}  # {name: {"expression": str, "metrics": dict}}
        self.max_size = max_size

        self.thresholds = thresholds or {
            "RankIC": 0.015,
            "RankICIR": 0.2,
            "Turnover": 1.5,
            "Diversity": 0.2,
        }

    def evaluate_factor(self, factor_name, signals, returns, existing_effective=None):
        """
        Evaluate a factor using IC/RankIC, turnover, and diversity metrics.
        Always returns a dict, even if metrics are NaN.
        """
        ic_series, rankic_series = utils.compute_ic_series(signals, returns)

        if ic_series.empty or rankic_series.empty:
            return {
                "RankIC": np.nan,
                "RankICIR": np.nan,
                "Turnover": self._compute_turnover(signals),
                "Diversity": 1.0,
            }

        rankic = rankic_series.mean()
        rankicir = rankic / rankic_series.std() if rankic_series.std() else np.nan
        turnover = self._compute_turnover(signals)

        diversity = 1.0
        if existing_effective and len(existing_effective) > 0:
            diversities = []
            for eff_signal in existing_effective:
                corr = signals.corrwith(eff_signal, axis=1).mean()
                diversities.append(1 - abs(corr))
            diversity = min(diversities) if diversities else 1.0

        return {
            "RankIC": rankic,
            "RankICIR": rankicir,
            "Turnover": turnover,
            "Diversity": diversity,
        }

    def add_factor(self, factor_name, expression, metrics):
        """Add factor to effective or deprecated pool based on thresholds."""
        is_effective = all(
            metrics[k] >= v if k != "Turnover" else metrics[k] <= v
            for k, v in self.thresholds.items()
        )

        if is_effective:
            self.effective[factor_name] = {"expression": expression, "metrics": metrics}
            if len(self.effective) > self.max_size:
                # Drop worst by RankIC
                worst = min(self.effective.items(), key=lambda x: x[1]["metrics"]["RankIC"])[0]
                self.deprecated[worst] = self.effective.pop(worst)
        else:
            self.deprecated[factor_name] = {"expression": expression, "metrics": metrics}

        return is_effective

    def _compute_turnover(self, signals):
        """Compute turnover as avg fraction of assets changing between top-k sets."""
        top_k = max(1, int(len(signals.columns) * 0.1))  # at least 1
        turnovers = []
        prev_top = None
        for date in signals.index:
            ranked = signals.loc[date].dropna().sort_values(ascending=False)
            top_assets = set(ranked.head(top_k).index)
            if prev_top is not None:
                change = len(top_assets.symmetric_difference(prev_top)) / top_k
                turnovers.append(change)
            prev_top = top_assets
        return np.mean(turnovers) if turnovers else 0.0