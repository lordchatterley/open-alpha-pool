import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# ----------------------------
# Logging
# ----------------------------

def setup_logger(name="chain_of_alpha", level=logging.INFO, log_file=None):
    """
    Create a logger with console + optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
            logger.addHandler(fh)

    return logger


# ----------------------------
# Config Loader
# ----------------------------

def load_config(dotenv_path=".env"):
    """
    Load environment variables into os.environ (used in config.py).
    """
    load_dotenv(dotenv_path)
    return {k: v for k, v in os.environ.items()}


# ----------------------------
# Metrics
# ----------------------------

def compute_ic(signals: pd.Series, returns: pd.Series):
    """Pearson IC between signals and returns."""
    if signals.std() == 0 or returns.std() == 0:
        return np.nan
    return pearsonr(signals, returns)[0]


def compute_rankic(signals: pd.Series, returns: pd.Series):
    """Spearman RankIC between signals and returns."""
    if len(set(signals)) < 2 or len(set(returns)) < 2:
        return np.nan
    return spearmanr(signals, returns)[0]


def compute_ic_series(signals: pd.DataFrame, returns: pd.DataFrame):
    """
    Compute IC and RankIC time series across dates.
    """
    ic_list, rankic_list = [], []
    for date in signals.index:
        s = signals.loc[date].dropna()
        r = returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 5:
            continue
        ic_list.append(compute_ic(s[common], r[common]))
        rankic_list.append(compute_rankic(s[common], r[common]))
    return pd.Series(ic_list), pd.Series(rankic_list)


def compute_portfolio_metrics(returns: pd.Series):
    """
    Compute standard portfolio metrics:
    - Annualized Return
    - Information Ratio
    - Max Drawdown
    """
    if returns.empty:
        return {"Annualized Return": np.nan, "IR": np.nan, "Max Drawdown": np.nan}

    ar = (1 + returns).prod() ** (252 / len(returns)) - 1
    ir = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() else np.nan
    cum = (1 + returns).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    mdd = dd.max()

    return {
        "Annualized Return": ar,
        "IR": ir,
        "Max Drawdown": mdd,
    }