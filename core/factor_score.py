import numpy as np

def aggregate_score(sharpe: float, ann_ret: float, D: float) -> float:
    vals = [sharpe, ann_ret, D]
    if any(np.isnan(v) for v in vals): return np.nan
    return float((sharpe + ann_ret + D) / 3.0)
