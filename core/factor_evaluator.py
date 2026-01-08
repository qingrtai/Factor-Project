# core/factor_evaluator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from core.factor_exec import safe_execute
from core.metrics import coverage as _coverage_simple, ann_ret, sharpe, max_dd, D_from_maxdd
from core.factor_score import aggregate_score
import re
from common.column_desc import COLUMN_DESC

# —— 与 factor_score.py 对齐的稳健化参数（可按需微调）——
FACTOR_CLIP_Q = (0.005, 0.995)     # 对原始因子值轻度去极值（仅影响排序的稳定性）
# 异常检测门槛（参考之前的项目）
MIN_DD_OBS = 12      # 最大回撤计算至少需要12个观测点（3年季度数据）
MIN_NEG_OBS = 3      # 至少需要3个负收益观测点（否则信号太弱）
NEUTRAL_MDD = 0.5    # 异常情况下的中性最大回撤值
NEUTRAL_D = 0.5      # 对应的中性D值
LS_WINSOR_Q   = (0.01, 0.99)       # 对多空收益序列分位去极值
LS_CLAMP      = (-0.5, 0.5)        # 对多空收益的幅度截断，抑制离群
MIN_GROUP     = 5                  # 单侧最少样本
QUANTILES     = (0.10, 0.20, 0.25, 0.30)  # 分层回退路径

def _extract_fields_used(code: str) -> tuple[int, str]:
    """
    从因子代码中提取使用的白名单字段
    
    支持两种格式：
    1. data['xxx'] 或 data["xxx"]
    2. data.get('xxx', ...) 或 data.get("xxx", ...)
    
    Returns:
        (字段数量, 字段列表字符串)
    """
    # 模式1: data['xxx'] 或 data["xxx"]
    pattern1 = r"data\[['\"](.+?)['\"]\]"
    matches1 = re.findall(pattern1, code)
    
    # 模式2: data.get('xxx', ...) 或 data.get("xxx", ...)
    pattern2 = r"data\.get\(['\"](.+?)['\"]"
    matches2 = re.findall(pattern2, code)
    
    # 合并两种匹配
    all_matches = matches1 + matches2
    
    # 过滤：只保留在白名单中的字段
    whitelist = set(COLUMN_DESC.keys())
    used_fields = []
    for field in all_matches:
        field_clean = field.strip()
        # 排除 factor_score 和中间变量，只保留白名单字段
        if field_clean in whitelist and field_clean not in used_fields:
            used_fields.append(field_clean)
    
    # 排序（保持一致性）
    used_fields = sorted(used_fields)
    
    n_fields = len(used_fields)
    fields_str = ",".join(used_fields) if used_fields else ""
    
    return n_fields, fields_str

# 评估输出固定列（失败行也要齐全，避免 KeyError）
COLS = [
    "factor_id", "code",
    "n_fields", "fields_used",  # ← 新增这两列
    # Train指标（带 train_ 前缀）
    "train_score", "train_coverage", "train_ann_ret", "train_sharpe", 
    "train_max_dd", "train_D", "train_diversity", "train_autocorr", "train_skew",
    # Val指标（带 val_ 前缀）
    "val_score", "val_coverage", "val_ann_ret", "val_sharpe",
    "val_max_dd", "val_D", "val_diversity", "val_autocorr", "val_skew",
    "status",
]

def _empty_metrics() -> dict:
    return dict(
        n_fields=0,          # ← 新增
        fields_used="",      # ← 新增
        # Train指标
        train_score=np.nan, train_coverage=np.nan, train_ann_ret=np.nan, 
        train_sharpe=np.nan, train_max_dd=np.nan, train_D=np.nan, 
        train_diversity=np.nan, train_autocorr=np.nan, train_skew=np.nan,
        # Val指标
        val_score=np.nan, val_coverage=np.nan, val_ann_ret=np.nan,
        val_sharpe=np.nan, val_max_dd=np.nan, val_D=np.nan,
        val_diversity=np.nan, val_autocorr=np.nan, val_skew=np.nan,
    )

def _soft_clip(x: pd.Series, qlo: float, qhi: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return x
    lo, hi = x.quantile([qlo, qhi])
    return x.clip(lo, hi)

def _sanitize_ls(ls: pd.Series) -> pd.Series:
    """对多空收益序列做 winsorize + clamp，增强稳健性（与 factor_score 保持一致思想）"""
    ls = pd.to_numeric(pd.Series(ls), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ls.empty:
        return ls
    lo, hi = ls.quantile([LS_WINSOR_Q[0], LS_WINSOR_Q[1]])
    ls = ls.clip(lo, hi)
    return ls.clip(LS_CLAMP[0], LS_CLAMP[1])

def _lag1_autocorr(s: pd.Series) -> float:
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    try:
        ac1 = float(s.autocorr(lag=1))
        return ac1 if np.isfinite(ac1) else np.nan
    except Exception:
        return np.nan

def _coverage_val(val_factor: pd.Series, val_df: pd.DataFrame, date_col: str, ret_col: str) -> float:
    """
    验证集 coverage 口径：逐日横截面里“因子与收益均非空”的样本占比的均值。
    这与 factor_score 的 coverage 度量一致，优于简单全局非空比例。
    """
    f = pd.to_numeric(val_factor, errors="coerce")
    r = pd.to_numeric(val_df[ret_col], errors="coerce")
    mask = f.notna() & r.notna()
    by_day = mask.groupby(val_df[date_col]).mean()
    return float(by_day.mean()) if not by_day.empty else np.nan

def _long_short_once(df: pd.DataFrame, factor: pd.Series, date_col: str, ret_col: str,
                     q: float, min_group: int) -> pd.Series:
    x = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "score": pd.to_numeric(factor, errors="coerce"),
        "ret":   pd.to_numeric(df[ret_col], errors="coerce"),
    }).dropna(subset=["date", "score", "ret"])
    if x.empty:
        return pd.Series(dtype=float)

    vals, idx = [], []
    for t, g in x.groupby("date", sort=True):
        s = g["score"]; r = g["ret"]
        if s.notna().sum() < 2 * min_group:
            continue
        rnk = s.rank(method="first", pct=True)
        lo = r[rnk <= q]
        hi = r[rnk >= 1 - q]
        if len(lo) >= min_group and len(hi) >= min_group:
            vals.append(float(hi.mean() - lo.mean()))
            idx.append(t)
    if not idx:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=pd.Index(idx, name="date")).sort_index()

def _long_short_fallback(df: pd.DataFrame, factor: pd.Series, date_col: str, ret_col: str) -> pd.Series:
    """最后回退：中位数二分（极端横截面也尽量给出 LS）"""
    x = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "score": pd.to_numeric(factor, errors="coerce"),
        "ret":   pd.to_numeric(df[ret_col], errors="coerce"),
    }).dropna(subset=["date", "score", "ret"])
    if x.empty:
        return pd.Series(dtype=float)

    vals, idx = [], []
    for t, g in x.groupby("date", sort=True):
        if g.shape[0] < 2:
            continue
        med = g["score"].median()
        lo = g.loc[g["score"] <= med, "ret"]
        hi = g.loc[g["score"] >  med, "ret"]
        if len(lo) >= 1 and len(hi) >= 1:
            vals.append(float(hi.mean() - lo.mean()))
            idx.append(t)
    if not idx:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=pd.Index(idx, name="date")).sort_index()

def _long_short_returns(df: pd.DataFrame, factor: pd.Series, date_col: str, ret_col: str) -> pd.Series:
    """多级分层回退：10/90 → 20/80 → 25/75 → 30/70 → 中位数二分"""
    for q in QUANTILES:
        ls = _long_short_once(df, factor, date_col, ret_col, q=q, min_group=MIN_GROUP)
        if not ls.empty:
            return ls
    return _long_short_fallback(df, factor, date_col, ret_col)

def _compute_maxdd_with_fallback(ls: pd.Series, nav: pd.Series) -> tuple[float, float]:
    """
    计算最大回撤，带异常检测和回退机制
    返回: (max_dd, D)
    
    异常情况（返回中性值0.5）：
    1. 样本量不足（<12个观测点）
    2. 负收益观测太少（<3个），说明信号太弱
    3. 收益序列近似常数或波动极小
    """
    # 基础检查
    if ls.empty or nav.empty:
        return NEUTRAL_MDD, NEUTRAL_D
    
    ls_clean = pd.to_numeric(ls, errors="coerce").dropna()
    
    # 检查1：样本量
    if len(ls_clean) < MIN_DD_OBS:
        return NEUTRAL_MDD, NEUTRAL_D
    
    # 检查2：负收益观测数（信号强度）
    neg_count = int((ls_clean < 0).sum())
    if neg_count < MIN_NEG_OBS:
        return NEUTRAL_MDD, NEUTRAL_D
    
    # 检查3：波动性（避免近似常数）
    ls_std = ls_clean.std()
    ls_mean = abs(ls_clean.mean())
    if ls_std < 1e-6 or (ls_std < 1e-4 and ls_mean < 1e-4):
        return NEUTRAL_MDD, NEUTRAL_D
    
    # 正常计算最大回撤（加保护：防止净值变0）
    try:
        nav_clean = pd.to_numeric(nav, errors="coerce").dropna()
        if nav_clean.empty:
            return NEUTRAL_MDD, NEUTRAL_D
        
        # 确保净值不会变成0或负数（极端情况下的保护）
        nav_clean = nav_clean.clip(lower=1e-6)
        
        peak = nav_clean.cummax()
        dd = nav_clean / peak - 1.0
        mdd = dd.min()  # 负数
        
        if not np.isfinite(mdd):
            return NEUTRAL_MDD, NEUTRAL_D
        
        mdd_val = float(abs(mdd))
        # 限制在合理范围 [0, 1]
        mdd_val = np.clip(mdd_val, 0.0, 1.0)
        D_val = np.clip(1.0 - mdd_val, 0.0, 1.0)
        
        return mdd_val, D_val
        
    except Exception:
        return NEUTRAL_MDD, NEUTRAL_D

def batch_evaluate(
    factors: list[dict],
    splits: dict,
    ret_col: str = "ret",
    date_col: str = "datadate",
    periods_per_year: int = 4,
    id_start: int = 1,
) -> pd.DataFrame:
    """
    打分规则：
      - 训练集只定方向并记录 train_score；
      - 验证集计算 coverage / Sharpe / AnnRet / MaxDD / D，并聚合为 val_score；
      - 除 train_score 外，其余指标一律用验证集；
      - 任意失败/空序列写入完整 NaN 指标并标注 status，保证 CSV 结构稳定。
    """
    rows = []

    for i, f in enumerate(factors, start=id_start):
        code = f["code"]

         # 提取使用的字段（失败也能记录）
        n_fields, fields_used = _extract_fields_used(code)  # ← 新增这行

        # 1) 执行因子代码（若失败，落空行）
        try:
            tr_factor = safe_execute(code, splits["train"])
            va_factor = safe_execute(code, splits["val"])
        except Exception as e:
            rows.append({
                "factor_id": i, 
                "code": code, 
                "n_fields": n_fields,      # ← 新增
                "fields_used": fields_used, # ← 新增
                **_empty_metrics(), 
                "status": f"fail: {e}"
            })
            continue

        # 2) 轻度稳健化（仅影响排序稳定性）：clip + tanh
        tr_factor = _soft_clip(tr_factor, *FACTOR_CLIP_Q)
        va_factor = _soft_clip(va_factor, *FACTOR_CLIP_Q)
        tr_factor = np.tanh(tr_factor / 2.0)
        va_factor = np.tanh(va_factor / 2.0)

        # 3) 训练期：先用原始 LS 定方向（负均值则乘 -1）
        tr_ls_raw = _long_short_returns(splits["train"], tr_factor, date_col, ret_col)
        if tr_ls_raw.empty:
            rows.append({
                "factor_id": i, 
                "code": code, 
                "n_fields": n_fields,      # ← 新增
                "fields_used": fields_used, # ← 新增
                **_empty_metrics(), 
                "status": "fail: empty train LS"
            })
            continue
     
        sign = -1.0 if tr_ls_raw.mean() < 0 else 1.0
        tr_factor = tr_factor * sign
        va_factor = va_factor * sign

        # 4) 训练指标（记录用）
        tr_ls = _sanitize_ls(_long_short_returns(splits["train"], tr_factor, date_col, ret_col))
        tr_nav = (1.0 + tr_ls.fillna(0)).cumprod()
        tr_sh  = sharpe(tr_ls, periods_per_year=periods_per_year)
        tr_ar  = ann_ret(tr_ls,  periods_per_year=periods_per_year)
        tr_mdd, tr_D = _compute_maxdd_with_fallback(tr_ls, tr_nav)
        train_score = aggregate_score(tr_sh, tr_ar, tr_D)

        # ← 添加这4行 ←
        tr_cov = _coverage_val(tr_factor, splits["train"], date_col, ret_col)
        tr_diversity = float(pd.to_numeric(tr_factor, errors="coerce").std(skipna=True))
        tr_autocorr = _lag1_autocorr(tr_ls)
        tr_skew = tr_ls.dropna().skew()

        # 5) 验证指标（排序/早停用）
        va_ls = _sanitize_ls(_long_short_returns(splits["val"], va_factor, date_col, ret_col))
        if va_ls.empty:
            rows.append({
                "factor_id": i, 
                "code": code, 
                "n_fields": n_fields,      # ← 新增
                "fields_used": fields_used, # ← 新增
                **_empty_metrics(), 
                "status": "fail: empty val LS"
            })
            continue
   

        va_cov = _coverage_val(va_factor, splits["val"], date_col, ret_col)  # 与 factor_score 一致的 coverage 口径
        va_nav = (1.0 + va_ls.fillna(0)).cumprod()
        va_sh  = sharpe(va_ls, periods_per_year=periods_per_year)
        va_ar  = ann_ret(va_ls,  periods_per_year=periods_per_year)
        va_mdd, va_D = _compute_maxdd_with_fallback(va_ls, va_nav)
        val_score = aggregate_score(va_sh, va_ar, va_D)

        rows.append({
            "factor_id": i,
            "code": code,
            "n_fields": n_fields,      # ← 新增
            "fields_used": fields_used, # ← 新增
            # Train指标
            "train_score": train_score,
            "train_coverage": tr_cov,
            "train_ann_ret": tr_ar,
            "train_sharpe": tr_sh,
            "train_max_dd": tr_mdd,
            "train_D": tr_D,
            "train_diversity": tr_diversity,
            "train_autocorr": tr_autocorr,
            "train_skew": tr_skew,
            # Val指标
            "val_score": val_score,
            "val_coverage": va_cov,
            "val_ann_ret": va_ar,
            "val_sharpe": va_sh,
            "val_max_dd": va_mdd,
            "val_D": va_D,
            "val_diversity": float(pd.to_numeric(va_factor, errors="coerce").std(skipna=True)),
            "val_autocorr": _lag1_autocorr(va_ls),
            "val_skew": va_ls.dropna().skew(),
            "status": "success",
        })

    # 6) 统一列顺序，失败行也齐全，永不 KeyError
    df = pd.DataFrame(rows).reindex(columns=COLS)
    return df
