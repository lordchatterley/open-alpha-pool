# chain_of_alpha_nasdaq/factor_optimization.py

import pandas as pd
import numpy as np
import re

class FactorOptimizer:
    def __init__(self, generator=None, max_steps=5, top_k=50):
        self.generator = generator
        self.max_steps = max_steps
        self.top_k = top_k

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

    def evaluate_formula(self, formula: str, data: pd.DataFrame) -> pd.Series | None:
        """Evaluate a factor formula against OHLCV data, return signals or None."""
        formula = self.sanitize_formula(formula)

        try:
            # Ensure DataFrame index is Date for alignment
            df = data.copy()
            df.index = pd.to_datetime(df.index)

            signals = eval(
                formula,
                {"__builtins__": {}},
                {
                    "Open": df["Open"],
                    "High": df["High"],
                    "Low": df["Low"],
                    "Close": df["Close"],
                    "Volume": df["Volume"],
                    "np": np,
                    "pd": pd,
                },
            )
            if isinstance(signals, (pd.Series, pd.DataFrame)):
                return signals
            else:
                return None
        except Exception as e:
            print(f"⚠️ Formula evaluation error [{formula}]: {e}")
            return None

    def optimize_factor(self, factor_name: str, formula: str, data: pd.DataFrame, returns: pd.DataFrame):
        """Naive optimization loop: try formula, measure IC/RankIC, refine via generator if provided."""
        current_formula = formula
        best_formula = formula
        best_metrics = None

        for step in range(self.max_steps):
            print(f"🔄 Optimization step {step+1}/{self.max_steps} for {factor_name}")

            signals = self.evaluate_formula(current_formula, data)
            if signals is None:
                print("⚠️ Formula evaluation failed. Skipping.")
                break

            # Compute IC and RankIC
            aligned = pd.concat([signals, returns.squeeze()], axis=1).dropna()
            if aligned.empty:
                print("⚠️ No aligned data for IC computation.")
                break

            ic = aligned.corr().iloc[0, 1]
            rankic = aligned.corr(method="spearman").iloc[0, 1]

            metrics = {"IC": ic, "RankIC": rankic}

            # Save best
            if best_metrics is None or abs(ic) > abs(best_metrics["IC"]):
                best_formula = current_formula
                best_metrics = metrics

            # Try refinement via generator (if available)
            if self.generator:
                current_formula = self.generator.refine_formula(current_formula, metrics)
            else:
                break

        return best_formula, best_metrics
