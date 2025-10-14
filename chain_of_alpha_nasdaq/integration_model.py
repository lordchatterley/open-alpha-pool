import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


class IntegrationModel:
    def __init__(self, model_type="linear"):
        """
        model_type: 'linear', 'rf', 'lgbm'
        """
        self.model_type = model_type
        self.model = None

    def _init_model(self):
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "rf":
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == "lgbm":
            return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def prepare_training_data(self, factor_signals: dict, returns: pd.DataFrame, horizon=1):
        """
        factor_signals: {factor_name: DataFrame (Date x Ticker)}
        returns: DataFrame of forward returns (Date x Ticker)
        horizon: forward return horizon (days)

        Returns: X (features), y (targets)
        """
        # Stack factors into a single panel
        features = []
        for fname, df in factor_signals.items():
            melted = df.stack().rename(fname)
            features.append(melted)
        X = pd.concat(features, axis=1)

        # Align targets
        y = returns.shift(-horizon).stack().rename("target")
        data = pd.concat([X, y], axis=1).dropna()

        return data.drop(columns="target"), data["target"]

    def fit(self, factor_signals: dict, returns: pd.DataFrame, horizon=1):
        """
        Train integration model.
        """
        X, y = self.prepare_training_data(factor_signals, returns, horizon)
        self.model = self._init_model()
        self.model.fit(X, y)
        return self

    def predict(self, factor_signals: dict):
        """
        Predict stock rankings using trained model.
        factor_signals: {factor_name: DataFrame (Date x Ticker)}

        Returns: DataFrame of predictions (Date x Ticker)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first.")

        # Build same feature matrix
        features = []
        for fname, df in factor_signals.items():
            melted = df.stack().rename(fname)
            features.append(melted)
        X = pd.concat(features, axis=1).dropna()

        preds = pd.Series(self.model.predict(X), index=X.index, name="pred")
        return preds.unstack()  # reshape back to (Date x Ticker)