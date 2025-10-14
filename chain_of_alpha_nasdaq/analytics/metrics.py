# chain_of_alpha_nasdaq/analytics/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def rank_ic(scores_cs: pd.DataFrame, returns_cs: pd.DataFrame) -> pd.DataFrame:
    """
    scores_cs/returns_cs: index=date, columns=ticker (aligned).
    Returns DataFrame with RankIC per date and summary (mean, std, IR).
    """
    assert scores_cs.index.equals(returns_cs.index)
    vals = []
    for dt in scores_cs.index:
        s = scores_cs.loc[dt]
        r = returns_cs.loc[dt]
        m = pd.concat([s, r], axis=1).dropna()
        if len(m) < 5:
            vals.append(np.nan); continue
        vals.append(spearmanr(m.iloc[:,0], m.iloc[:,1])[0])
    ic = pd.Series(vals, index=scores_cs.index, name="RankIC")
    summary = pd.DataFrame({
        "RankIC_mean":[ic.mean()],
        "RankIC_std":[ic.std(ddof=0)],
        "RankICIR":[ic.mean()/(ic.std(ddof=0)+1e-9)]
    })
    return ic.to_frame(), summary
