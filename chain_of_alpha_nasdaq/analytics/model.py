# chain_of_alpha_nasdaq/analytics/model.py
import os, json, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, UTC

def assemble_panel(features_long: pd.DataFrame, labels_long: pd.DataFrame):
    """
    features_long: date,ticker,feature_name,value
    labels_long:   date,ticker,horizon,fwd_return
    Returns:
      X (DataFrame: rows=(date,ticker), cols=features), 
      y (Series), dates (Index of dates), tickers (Index aligned)
    """
    X = features_long.pivot_table(index=["date","ticker"], columns="feature_name", values="value")
    y = labels_long.set_index(["date","ticker"])["fwd_return"]
    # align
    m = X.join(y, how="inner").dropna()
    y = m["fwd_return"].astype(float)
    X = m.drop(columns=["fwd_return"]).astype(float)
    dates = X.index.get_level_values("date")
    tickers = X.index.get_level_values("ticker")
    return X, y, dates, tickers

def train_lgbm(X: pd.DataFrame, y: pd.Series, dates: pd.Index, val_days: int = 20, params: dict | None = None):
    """
    Simple time split: last `val_days` unique dates as validation.
    """
    uniq_dates = pd.Index(sorted(pd.unique(dates)))
    if len(uniq_dates) <= val_days:
        val_days = max(1, len(uniq_dates)//5)
    split_date = uniq_dates[-val_days]
    train_mask = dates < split_date
    valid_mask = dates >= split_date

    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[valid_mask], y[valid_mask]

    if params is None:
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.005,
            num_leaves=24,
            max_depth=8,
            n_estimators=2000,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        valid_sets=[dtr, dva],
        valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )
    return model

def save_model_sqlite(store, model, horizon: int, model_id: str | None = None, params: dict | None = None, base_dir="data/models"):
    os.makedirs(base_dir, exist_ok=True)
    model_id = model_id or datetime.now(UTC).strftime("lgbm_%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{model_id}.pkl")
    joblib.dump(model, path)
    cur = store.conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO models (model_id, trained_at, horizon, params_json, artifact_path) VALUES (?, ?, ?, ?, ?)",
        (model_id, datetime.now(UTC).isoformat(), horizon, json.dumps(params or {}), path)
    )
    store.conn.commit()
    return model_id, path

def predict_panel(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration or model.current_iteration())
