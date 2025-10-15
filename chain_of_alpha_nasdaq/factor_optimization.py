# chain_of_alpha_nasdaq/factor_optimization.py

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FactorEvaluationResult:
    """Structured output for an evaluated factor formula."""

    formula: str
    signals: pd.DataFrame
    forward_returns: pd.DataFrame
    ic_series: pd.Series
    rankic_series: pd.Series
    metrics: dict


class FactorOptimizer:
    def __init__(self, generator=None, max_steps: int = 5, top_k_ratio: float = 0.1, horizon: int = 1):
        self.generator = generator
        self.max_steps = max_steps
        self.top_k_ratio = top_k_ratio
        self.horizon = horizon

    def sanitize_formula(self, formula: str) -> str:
        """Clean LLM-generated formulas (remove numbering, markdown fences, etc)."""
        if not formula:
            return ""
        # Remove markdown fences
        formula = re.sub(r"```.*?```", "", formula, flags=re.DOTALL)
        formula = formula.replace("```python", "").replace("```", "")
        # Remove numbering like '1. ' at start
        formula = re.sub(r"^\s*\d+\.\s*", "", formula)
        return formula.strip()

    def _extract_panel(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("FactorOptimizer expects MultiIndex columns [Ticker, Field]")
        if "Field" not in data.columns.names:
            raise ValueError("Price panel must include a 'Field' level")
        panel = data.xs(field, axis=1, level="Field")
        panel = panel.sort_index()
        panel.index = pd.to_datetime(panel.index)
        return panel

    def _compute_forward_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        close_prices = self._extract_panel(data, "Close")
        forward_returns = close_prices.pct_change(periods=self.horizon).shift(-self.horizon)
        return forward_returns

    def evaluate_formula(self, formula: str, data: pd.DataFrame) -> pd.DataFrame | None:
        """Evaluate a factor formula for every ticker, returning a cross-sectional signal frame."""
        formula = self.sanitize_formula(formula)
        if not formula:
            return None

        tickers = data.columns.get_level_values("Ticker").unique().tolist()
        signals: dict[str, pd.Series] = {}

        for ticker in tickers:
            slice_df = data.xs(ticker, axis=1, level="Ticker")
            slice_df = slice_df.sort_index()
            slice_df.index = pd.to_datetime(slice_df.index)

            context = {
                "Open": slice_df.get("Open"),
                "High": slice_df.get("High"),
                "Low": slice_df.get("Low"),
                "Close": slice_df.get("Close"),
                "Volume": slice_df.get("Volume"),
                "np": np,
                "pd": pd,
            }

            try:
                raw_signals = eval(formula, {"__builtins__": {}}, context)
            except Exception as exc:
                print(f"⚠️ Formula evaluation error for {ticker} [{formula}]: {exc}")
                return None

            if isinstance(raw_signals, pd.DataFrame):
                if raw_signals.empty:
                    continue
                series = raw_signals.iloc[:, 0]
            elif isinstance(raw_signals, pd.Series):
                series = raw_signals
            elif np.isscalar(raw_signals):
                series = pd.Series(raw_signals, index=slice_df.index)
            else:
                print(f"⚠️ Unsupported signal output for {ticker}: {type(raw_signals)}")
                return None

            series = pd.to_numeric(series, errors="coerce")
            if series.isna().all():
                continue
            signals[ticker] = series

        if not signals:
            print(f"⚠️ Formula produced no valid signals [{formula}]")
            return None

        frame = pd.DataFrame(signals).sort_index()
        frame.index = pd.to_datetime(frame.index)
        return frame

    def _compute_turnover(self, signals: pd.DataFrame) -> float:
        if signals.empty:
            return float("nan")
        top_k = max(1, int(len(signals.columns) * self.top_k_ratio))
        prev_top: set[str] | None = None
        turnovers: list[float] = []
        for _, row in signals.sort_index().iterrows():
            ranked = row.dropna().sort_values(ascending=False)
            if ranked.empty:
                continue
            top_assets = set(ranked.head(top_k).index)
            if prev_top is not None and top_assets:
                change = len(top_assets.symmetric_difference(prev_top)) / top_k
                turnovers.append(change)
            prev_top = top_assets
        return float(np.mean(turnovers)) if turnovers else float("nan")

    def _compute_diversity(self, signals: pd.DataFrame, existing_signals: Iterable[pd.DataFrame] | None) -> float:
        if not existing_signals:
            return 1.0

        diversities: list[float] = []
        candidate_flat = signals.stack().rename("candidate").dropna()
        if candidate_flat.empty:
            return float("nan")

        for frame in existing_signals:
            if frame is None or frame.empty:
                continue
            aligned = candidate_flat.to_frame().join(frame.stack().rename("existing"), how="inner").dropna()
            if len(aligned) < 2:
                continue
            corr = aligned.corr().iloc[0, 1]
            diversities.append(1 - abs(corr))

        return float(min(diversities)) if diversities else 1.0

    def _summarise_metrics(
        self,
        formula: str,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        existing_signals: Iterable[pd.DataFrame] | None,
    ) -> FactorEvaluationResult | None:
        if signals.empty:
            return None

        aligned_index = signals.index.intersection(forward_returns.index)
        if aligned_index.empty:
            return None

        signals = signals.loc[aligned_index]
        returns = forward_returns.loc[aligned_index]

        ic_values: list[float] = []
        rankic_values: list[float] = []
        ic_dates: list[pd.Timestamp] = []

        for date in aligned_index:
            signal_row = signals.loc[date]
            return_row = returns.loc[date]
            mask = signal_row.notna() & return_row.notna()
            if mask.sum() < 2:
                continue
            ic_dates.append(date)
            ic_values.append(signal_row[mask].corr(return_row[mask], method="pearson"))
            rankic_values.append(signal_row[mask].corr(return_row[mask], method="spearman"))

        ic_series = pd.Series(ic_values, index=ic_dates, name="IC").dropna()
        rankic_series = pd.Series(rankic_values, index=ic_dates, name="RankIC").dropna()

        ic_mean = float(ic_series.mean()) if not ic_series.empty else float("nan")
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
        rankic_mean = float(rankic_series.mean()) if not rankic_series.empty else float("nan")
        rankic_std = float(rankic_series.std(ddof=1)) if len(rankic_series) > 1 else float("nan")

        ic_ir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std != 0.0 else float("nan")
        rankic_ir = (
            rankic_mean / rankic_std if np.isfinite(rankic_std) and rankic_std != 0.0 else float("nan")
        )

        coverage = float(signals.notna().sum().sum() / signals.size) if signals.size else 0.0

        metrics = {
            "Formula": formula,
            "IC": ic_mean,
            "ICStd": ic_std,
            "ICIR": ic_ir,
            "RankIC": rankic_mean,
            "RankICStd": rankic_std,
            "RankICIR": rankic_ir,
            "Turnover": self._compute_turnover(signals),
            "Diversity": self._compute_diversity(signals, existing_signals),
            "Coverage": coverage,
            "Observations": int(len(ic_series)),
        }

        return FactorEvaluationResult(
            formula=formula,
            signals=signals,
            forward_returns=returns,
            ic_series=ic_series,
            rankic_series=rankic_series,
            metrics=metrics,
        )

    def optimize_factor(
        self,
        factor_name: str,
        formula: str,
        data: pd.DataFrame,
        existing_signals: Iterable[pd.DataFrame] | None = None,
    ) -> FactorEvaluationResult | None:
        """Iteratively evaluate and optionally refine a factor formula."""

        forward_returns = self._compute_forward_returns(data)
        current_formula = formula
        best_result: FactorEvaluationResult | None = None

        for step in range(self.max_steps):
            print(f"🔄 Optimization step {step + 1}/{self.max_steps} for {factor_name}")
            signals = self.evaluate_formula(current_formula, data)
            if signals is None:
                print("⚠️ Formula evaluation failed. Skipping.")
                break

            evaluation = self._summarise_metrics(current_formula, signals, forward_returns, existing_signals)
            if evaluation is None:
                print("⚠️ Unable to compute metrics for formula. Aborting optimization.")
                break

            if best_result is None or (
                abs(evaluation.metrics.get("IC", float("nan")))
                > abs(best_result.metrics.get("IC", float("nan")))
            ):
                best_result = evaluation

            if not self.generator:
                break

            current_formula = self.generator.refine_formula(current_formula, evaluation.metrics)

        return best_result
