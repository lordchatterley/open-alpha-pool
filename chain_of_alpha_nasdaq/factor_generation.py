# chain_of_alpha_nasdaq/factor_generation.py

from openai import OpenAI
import os
import pandas as pd


class FactorGenerator:
    """
    Uses an LLM (via OpenAI API) to generate candidate formulaic alpha factors
    and evaluate them on OHLCV data.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_factors(self, num_factors: int = 3):
        """
        Generate a list of candidate alpha factor formulas.
        """
        prompt = f"""
You are a quantitative researcher.
Generate {num_factors} simple, interpretable alpha factor formulas
using common price/volume operators (Open, High, Low, Close, Volume).
Formulas should be concise and computable with pandas.
Output one formula per line, no explanation, just the formula.
Example:
Close / Open
Volume / mean(Volume, 5)
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )

        text = completion.choices[0].message.content.strip()
        formulas = [line.strip() for line in text.splitlines() if line.strip()]

        return formulas

    def evaluate_formula(self, formula: str, data: pd.DataFrame) -> pd.Series | None:
        """
        Evaluate a formulaic alpha expression on OHLCV data.

        Args:
            formula (str): the formula string (e.g., "Close - Open").
            data (pd.DataFrame): must include columns [Open, High, Low, Close, Volume].

        Returns:
            pd.Series with Date index and factor values, or None if evaluation fails.
        """
        try:
            # Clean up any stray markdown fences
            formula = formula.replace("```", "").replace("python", "").strip()

            # Evaluate formula using pandas eval
            signals = pd.eval(formula, local_dict=data)
            signals.index = data.index
            return signals

        except Exception as e:
            print(f"⚠️ Failed to evaluate formula '{formula}': {e}")
            return None
