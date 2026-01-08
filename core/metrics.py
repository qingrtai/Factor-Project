# core/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd


def _to_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric and drop +/- inf (keep NaNs)."""
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def coverage(factor: pd.Series) -> float:
    """
    简单覆盖度：非空占比（全局）。
    注：验证段逐日横截面 coverage 已在 evaluator 中单独计算，这里保持基础定义即可。
    """
    x = _to_numeric(factor)
    return float(1.0 - x.isna().mean()) if len(x) else np.nan


def ann_ret(ret: pd.Series, periods_per_year: int = 4) -> float:
    """
    年化收益率（越大越好）。
    首选几何年化：G = (∏(1+r))^(ppy/n) - 1
    若 ∏(1+r) ≤ 0（极端/脏值）或样本太少，则回退为算术年化：mean(r) * ppy
    """
    r = _to_numeric(ret).dropna()
    n = len(r)
    if n == 0:
        return np.nan
    # 尝试几何年化（常规情况下 evaluator 已做过收益截断，几何应安全）
    gprod = float(np.prod(1.0 + r))
    if np.isfinite(gprod) and gprod > 0.0 and n > 0:
        try:
            return gprod ** (periods_per_year / n) - 1.0
        except Exception:
            pass
    # 回退：算术年化
    return float(r.mean() * periods_per_year)


def sharpe(ret: pd.Series, periods_per_year: int = 4, eps: float = 1e-12) -> float:
    """
    年化夏普比率： (mean(r) * ppy) / (std(r) * sqrt(ppy))
    当样本过少或波动过小（<eps）返回 NaN。
    """
    r = _to_numeric(ret).dropna()
    if len(r) < 3:
        return np.nan
    mu = float(r.mean()) * periods_per_year
    sd = float(r.std(ddof=1)) * (periods_per_year ** 0.5)
    if not np.isfinite(sd) or sd < eps:
        return np.nan
    val = mu / sd
    return float(val) if np.isfinite(val) else np.nan


def max_dd(nav: pd.Series) -> float:
    """
    最大回撤（输出为正值，越小越好）。
    传入的是净值曲线（例如 (1 + ret).cumprod()）。
    """
    x = _to_numeric(nav).dropna()
    if x.empty:
        return np.nan
    peak = x.cummax()
    dd = x / peak - 1.0
    mdd = dd.min()  # 最小值是负数
    if not np.isfinite(mdd):
        return np.nan
    return float(abs(mdd))


def D_from_maxdd(maxdd: float) -> float:
    """
    定义 D = 1 - MaxDD（越大越好），并限幅到 [0, 1] 防溢出。
    """
    if maxdd is None or not np.isfinite(maxdd):
        return np.nan
    d = 1.0 - float(maxdd)
    # 保险起见做限幅
    if np.isnan(d):
        return np.nan
    return float(min(1.0, max(0.0, d)))
