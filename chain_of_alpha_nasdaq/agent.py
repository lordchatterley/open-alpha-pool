import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from chain_of_alpha_nasdaq import (
    data_handler,
    factor_generation,
    factor_optimization,
    portfolio,
)
from chain_of_alpha_nasdaq.alpha_db import AlphaDB
from chain_of_alpha_nasdaq.alpha_pool import AlphaPool


class ChainOfAlphaAgent:
    """
    Main orchestrator for the alpha generation, optimization, and backtesting pipeline.

    Steps:
      1. Fetch historical market data.
      2. Generate seed alpha factors (text -> formula).
      3. Optimize each factor using data-driven metrics.
      4. Save optimized factors and metrics to the database.
      5. Construct a portfolio and backtest it.
    """

    def __init__(
        self,
        tickers,
        db_path="alphas.db",
        model="gpt-4o-mini",
        signal_store_dir: str | os.PathLike[str] = "artifacts/factor_signals",
    ):
        self.tickers = tickers
        self.db = AlphaDB(db_path)
        self.generator = factor_generation.FactorGenerator(model=model)
        self.optimizer = factor_optimization.FactorOptimizer()
        self.pool = AlphaPool()
        self.signal_store = Path(signal_store_dir)
        self.signal_store.mkdir(parents=True, exist_ok=True)

    def _persist_signals(
        self,
        factor_name: str,
        signals: pd.DataFrame,
        category: str = "candidates",
    ) -> Path:
        directory = self.signal_store / category
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{factor_name}.csv"
        signals.to_csv(path)
        return path

    def _load_effective_signals(self) -> dict[str, pd.DataFrame]:
        stored: dict[str, pd.DataFrame] = {}
        effective_dir = self.signal_store / "effective"
        if not effective_dir.exists():
            return stored
        for csv_path in effective_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                stored[csv_path.stem] = df
            except Exception:
                continue
        return stored

    def run_pipeline(self, start="2020-01-01", end="2024-01-01", num_new_factors=3):
        """
        Run the full alpha discovery and evaluation pipeline.

        Args:
            start (str): Start date for historical data.
            end (str): End date for historical data.
            num_new_factors (int): Number of new factors to generate.

        Returns:
            dict: Contains factor signals, combined signal, portfolio results, and metrics.
        """
        print("📡 Fetching market data...")
        data = data_handler.fetch_data(self.tickers, start, end)

        print("🧪 Generating seed factors...")
        seed_factors = self.generator.generate_factors(num_new_factors=num_new_factors)

        # Normalize generator output
        if isinstance(seed_factors, list):
            seed_factors = {f"alpha_{i+1}": f for i, f in enumerate(seed_factors)}

        factor_signals: dict[str, pd.DataFrame] = {}
        effective_signals = self._load_effective_signals()

        # === Optimize and store each factor ===
        for name, formula in seed_factors.items():
            print(f"\n🔧 Optimizing {name}: {formula}")
            try:
                evaluation = self.optimizer.optimize_factor(
                    factor_name=name,
                    formula=formula,
                    data=data,
                    existing_signals=effective_signals.values(),
                )

                if evaluation is None:
                    print(f"⚠️ Optimization failed for {name}: no metrics produced.")
                    continue

                print(f"✅ {name} optimized → {evaluation.formula}")

                self.db.insert_factor(name, evaluation.formula)

                metrics = evaluation.metrics.copy()
                metrics_lower = {k.lower(): v for k, v in metrics.items()}

                self.db.insert_metrics(
                    factor_name=name,
                    run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    metrics={
                        "ic": metrics_lower.get("ic"),
                        "rankic": metrics_lower.get("rankic"),
                        "icir": metrics_lower.get("icir"),
                        "rankicir": metrics_lower.get("rankicir"),
                        "turnover": metrics_lower.get("turnover"),
                    },
                )

                factor_signals[name] = evaluation.signals
                candidate_path = self._persist_signals(name, evaluation.signals, "candidates")

                is_effective = self.pool.add_factor(name, evaluation.formula, metrics)
                if is_effective:
                    effective_signals[name] = evaluation.signals
                    persisted_path = self._persist_signals(name, evaluation.signals, "effective")
                    print(f"💾 Stored signals for {name} at {persisted_path}")
                else:
                    print(f"🚧 {name} did not meet pool thresholds; candidate saved at {candidate_path}")

            except Exception as e:
                print(f"⚠️ Optimization failed for {name}: {e}")

        if not factor_signals:
            print("⚠️ No factor signals were produced. Exiting.")
            return None

        print("🔗 Combining factor signals into aggregate alpha...")
        selected_signals = effective_signals if effective_signals else factor_signals
        combined_sum: pd.DataFrame | None = None
        for frame in selected_signals.values():
            combined_sum = frame if combined_sum is None else combined_sum.add(frame, fill_value=0.0)
        combined_signal = combined_sum.divide(len(selected_signals)) if combined_sum is not None else pd.DataFrame()
        combined_signal.index.name = "Date"

        print("📈 Constructing portfolio and running backtest...")
        port = portfolio.Portfolio(top_k=20)
        result = port.run_backtest(combined_signal, data)

        print("📊 Portfolio Backtest Metrics:")
        for k, v in result[1].items():
            print(f"  {k}: {v:.4f}")

        # ✅ Normalize metrics to lowercase for consistency
        metrics_lower = {k.lower(): v for k, v in result[1].items()}

        # ✅ Portfolio results as a DataFrame (for test compatibility)
        portfolio_results = pd.DataFrame({"equity_curve": result[0]})

        # ✅ Structured return (tests expect this exact format)
        return {
            "factor_signals": factor_signals,
            "combined_signal": combined_signal,
            "portfolio_results": portfolio_results,
            "metrics": metrics_lower,
        }
