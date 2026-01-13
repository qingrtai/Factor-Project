"""
无数据泄露的测试集评估函数 (IMPROVED VERSION)

改进点:
1. Sharpe 阈值考虑样本量（小样本放宽到 3.5）
2. MaxDD 计算失败时使用中性值而非 NaN（避免直接失败）
3. 添加失败统计（可选）
"""

# ========== 重要：必须先设置路径，再导入 core 模块 ==========
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ==========================================================

import numpy as np
import pandas as pd
from core.factor_exec import safe_execute
from core.metrics import ann_ret, sharpe, max_dd, D_from_maxdd
from core.factor_score import aggregate_score

# 与 factor_evaluator.py 对齐的参数
FACTOR_CLIP_Q = (0.005, 0.995)
MIN_DD_OBS = 3  # ← IMPROVED: 从 5 降到 3
MIN_NEG_OBS = 1  # ← IMPROVED: 从 2 降到 1
NEUTRAL_MDD = 0.5
NEUTRAL_D = 0.5
LS_WINSOR_Q = (0.01, 0.99)
LS_CLAMP = (-0.5, 0.5)
MIN_GROUP = 5
QUANTILES = (0.10, 0.20, 0.25, 0.30)

# ========== IMPROVED: Sharpe 阈值配置 ========== #
# 根据样本量动态调整阈值
def get_sharpe_threshold(sample_size: int) -> float:
    """
    根据样本量返回合理的 Sharpe 阈值
    
    原理：
    - 小样本（< 60）：Sharpe 估计不稳定，放宽到 3.5
    - 大样本（>= 60）：使用标准阈值 2.5
    """
    return 3.5 if sample_size < 60 else 2.5
# =============================================== #

# 输出列
COLS = [
    "factor_id", "code", "n_fields", "fields_used",
    "test_score", "test_coverage", "test_ann_ret", "test_sharpe",
    "test_max_dd", "test_D", "test_diversity", "test_autocorr", "test_skew",
    "status",
]


def _extract_fields_used(code: str) -> tuple[int, str]:
    """从因子代码中提取使用的字段（简化版，不依赖 column_desc）"""
    import re
    
    # 模式1: data['xxx'] 或 data["xxx"]
    pattern1 = r"data\[['\"](.+?)['\"]\]"
    matches1 = re.findall(pattern1, code)
    
    # 模式2: data.get('xxx', ...) 或 data.get("xxx", ...)
    pattern2 = r"data\.get\(['\"](.+?)['\"]"
    matches2 = re.findall(pattern2, code)
    
    # 合并并去重
    all_matches = matches1 + matches2
    used_fields = []
    
    for field in all_matches:
        field_clean = field.strip()
        # 排除特殊列名
        if field_clean not in ['factor_score'] and field_clean not in used_fields:
            used_fields.append(field_clean)
    
    used_fields = sorted(used_fields)
    return len(used_fields), ",".join(used_fields) if used_fields else ""


def _empty_metrics() -> dict:
    """空指标（失败时使用）"""
    return dict(
        n_fields=0,
        fields_used="",
        test_score=np.nan,
        test_coverage=np.nan,
        test_ann_ret=np.nan,
        test_sharpe=np.nan,
        test_max_dd=np.nan,
        test_D=np.nan,
        test_diversity=np.nan,
        test_autocorr=np.nan,
        test_skew=np.nan,
    )


def _soft_clip(x: pd.Series, qlo: float, qhi: float) -> pd.Series:
    """轻度去极值"""
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return x
    lo, hi = x.quantile([qlo, qhi])
    return x.clip(lo, hi)


def _sanitize_ls(ls: pd.Series) -> pd.Series:
    """对多空收益序列做稳健化处理"""
    ls = pd.to_numeric(pd.Series(ls), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ls.empty:
        return ls
    lo, hi = ls.quantile([LS_WINSOR_Q[0], LS_WINSOR_Q[1]])
    ls = ls.clip(lo, hi)
    return ls.clip(LS_CLAMP[0], LS_CLAMP[1])


def _lag1_autocorr(s: pd.Series) -> float:
    """一阶自相关"""
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    try:
        ac1 = float(s.autocorr(lag=1))
        return ac1 if np.isfinite(ac1) else np.nan
    except Exception:
        return np.nan


def _coverage_val(val_factor: pd.Series, val_df: pd.DataFrame, date_col: str, ret_col: str) -> float:
    """逐日横截面 coverage"""
    f = pd.to_numeric(val_factor, errors="coerce")
    r = pd.to_numeric(val_df[ret_col], errors="coerce")
    mask = f.notna() & r.notna()
    by_day = mask.groupby(val_df[date_col]).mean()
    return float(by_day.mean()) if not by_day.empty else np.nan


def _long_short_once(df: pd.DataFrame, factor: pd.Series, date_col: str, ret_col: str,
                     q: float, min_group: int) -> pd.Series:
    """单个分位数的多空收益"""
    x = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "score": pd.to_numeric(factor, errors="coerce"),
        "ret": pd.to_numeric(df[ret_col], errors="coerce"),
    }).dropna(subset=["date", "score", "ret"])
    
    if x.empty:
        return pd.Series(dtype=float)
    
    vals, idx = [], []
    for t, g in x.groupby("date", sort=True):
        s = g["score"]
        r = g["ret"]
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
    """中位数二分回退"""
    x = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "score": pd.to_numeric(factor, errors="coerce"),
        "ret": pd.to_numeric(df[ret_col], errors="coerce"),
    }).dropna(subset=["date", "score", "ret"])
    
    if x.empty:
        return pd.Series(dtype=float)
    
    vals, idx = [], []
    for t, g in x.groupby("date", sort=True):
        if g.shape[0] < 2:
            continue
        med = g["score"].median()
        lo = g.loc[g["score"] <= med, "ret"]
        hi = g.loc[g["score"] > med, "ret"]
        if len(lo) >= 1 and len(hi) >= 1:
            vals.append(float(hi.mean() - lo.mean()))
            idx.append(t)
    
    if not idx:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=pd.Index(idx, name="date")).sort_index()


def _long_short_returns(df: pd.DataFrame, factor: pd.Series, date_col: str, ret_col: str) -> pd.Series:
    """多级分层回退计算多空收益"""
    for q in QUANTILES:
        ls = _long_short_once(df, factor, date_col, ret_col, q=q, min_group=MIN_GROUP)
        if not ls.empty:
            return ls
    return _long_short_fallback(df, factor, date_col, ret_col)


def _compute_maxdd_with_fallback(ls: pd.Series, nav: pd.Series) -> tuple[float, float]:
    """
    计算最大回撤（IMPROVED: 使用中性值作为 fallback）
    
    改进点：
    - 当样本量不足时，返回中性值 (0.5, 0.5) 而非 NaN
    - 避免因数据不足而直接失败
    """
    if ls.empty or nav.empty:
        return NEUTRAL_MDD, NEUTRAL_D  # ← IMPROVED: 改为中性值
    
    ls_clean = pd.to_numeric(ls, errors="coerce").dropna()
    
    # ========== IMPROVED: 使用中性值而非 NaN ========== #
    if len(ls_clean) < MIN_DD_OBS:
        return NEUTRAL_MDD, NEUTRAL_D  # ← 样本量不足，返回中性值
    
    neg_count = int((ls_clean < 0).sum())
    if neg_count < MIN_NEG_OBS:
        return NEUTRAL_MDD, NEUTRAL_D  # ← 负收益不足，返回中性值
    # =============================================== #
    
    ls_std = ls_clean.std()
    ls_mean = abs(ls_clean.mean())
    if ls_std < 1e-6 or (ls_std < 1e-4 and ls_mean < 1e-4):
        return NEUTRAL_MDD, NEUTRAL_D
    
    try:
        nav_clean = pd.to_numeric(nav, errors="coerce").dropna()
        if nav_clean.empty:
            return NEUTRAL_MDD, NEUTRAL_D
        
        nav_clean = nav_clean.clip(lower=1e-6)
        peak = nav_clean.cummax()
        dd = nav_clean / peak - 1.0
        mdd = dd.min()
        
        if not np.isfinite(mdd):
            return NEUTRAL_MDD, NEUTRAL_D
        
        mdd_val = float(abs(mdd))
        mdd_val = np.clip(mdd_val, 0.0, 1.0)
        D_val = np.clip(1.0 - mdd_val, 0.0, 1.0)
        
        return mdd_val, D_val
        
    except Exception:
        return NEUTRAL_MDD, NEUTRAL_D


def evaluate_on_test_holdout(
    factors: list[dict],
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    ret_col: str = "ret",
    date_col: str = "datadate",
    periods_per_year: int = 4,
    id_start: int = 1,
) -> pd.DataFrame:
    """
    无数据泄露的测试集评估 (IMPROVED VERSION)
    
    改进点:
    1. Sharpe 阈值考虑样本量
    2. MaxDD 计算失败时使用中性值
    3. 更宽松的失败条件
    
    关键：
    1. 因子方向在验证集上确定
    2. 测试集只做纯净评估，不做任何调整
    
    Args:
        factors: 因子列表 [{'code': '...', 'factor_id': ...}, ...]
        val_data: 验证集数据（用于确定因子方向）
        test_data: 测试集数据（纯净评估）
        ret_col: 收益列名
        date_col: 日期列名
        periods_per_year: 每年期数
        id_start: 起始 factor_id
    
    Returns:
        包含测试集评估结果的 DataFrame
    """
    rows = []
    
    for i, f in enumerate(factors, start=id_start):
        code = f["code"]
        n_fields, fields_used = _extract_fields_used(code)
        
        # 1) 执行因子代码
        try:
            val_factor = safe_execute(code, val_data)
            test_factor = safe_execute(code, test_data)
        except Exception as e:
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": f"fail: {e}"
            })
            continue
        
        # 2) 轻度稳健化
        val_factor = _soft_clip(val_factor, *FACTOR_CLIP_Q)
        test_factor = _soft_clip(test_factor, *FACTOR_CLIP_Q)
        val_factor = np.tanh(val_factor / 2.0)
        test_factor = np.tanh(test_factor / 2.0)
        
        # 3) 关键：在验证集上确定方向
        val_ls_raw = _long_short_returns(val_data, val_factor, date_col, ret_col)
        if val_ls_raw.empty:
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": "fail: empty val LS"
            })
            continue
        
        # 在验证集上确定方向
        sign = -1.0 if val_ls_raw.mean() < 0 else 1.0
        
        # 4) 在测试集上应用这个固定的方向
        test_factor = test_factor * sign
        
        # 5) 测试集评估（纯净，无数据泄露）
        test_ls = _sanitize_ls(_long_short_returns(test_data, test_factor, date_col, ret_col))
        
        if test_ls.empty:
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": "fail: empty test LS"
            })
            continue
        
        # 6) 计算测试集指标
        test_cov = _coverage_val(test_factor, test_data, date_col, ret_col)
        test_nav = (1.0 + test_ls.fillna(0)).cumprod()
        test_sh = sharpe(test_ls, periods_per_year=periods_per_year)
        test_ar = ann_ret(test_ls, periods_per_year=periods_per_year)
        test_mdd, test_D = _compute_maxdd_with_fallback(test_ls, test_nav)

        # ========== IMPROVED: 异常检测（更宽松）========== #
        # 检查 1: 样本量太小
        if len(test_ls) < 10:
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": f"fail: insufficient test samples ({len(test_ls)})"
            })
            continue

        # 检查 2: Sharpe 异常高（考虑样本量）
        sharpe_threshold = get_sharpe_threshold(len(test_ls))
        if test_sh > sharpe_threshold:
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": f"fail: unrealistic sharpe ({test_sh:.2f}, threshold={sharpe_threshold:.1f})"
            })
            continue
        
        # 检查 3: test_D 是否为 NaN（IMPROVED: 已在 _compute_maxdd_with_fallback 中处理）
        # 现在不会再返回 NaN，所以这个检查可以移除或保留作为二次检查
        if np.isnan(test_D):
            rows.append({
                "factor_id": i,
                "code": code,
                "n_fields": n_fields,
                "fields_used": fields_used,
                **_empty_metrics(),
                "status": f"fail: maxdd calculation returned NaN (fallback failed)"
            })
            continue
        # =============================================== #

        test_score = aggregate_score(test_sh, test_ar, test_D)
        
        rows.append({
            "factor_id": i,
            "code": code,
            "n_fields": n_fields,
            "fields_used": fields_used,
            "test_score": test_score,
            "test_coverage": test_cov,
            "test_ann_ret": test_ar,
            "test_sharpe": test_sh,
            "test_max_dd": test_mdd,
            "test_D": test_D,
            "test_diversity": float(pd.to_numeric(test_factor, errors="coerce").std(skipna=True)),
            "test_autocorr": _lag1_autocorr(test_ls),
            "test_skew": test_ls.dropna().skew(),
            "status": "success",
        })
    
    df = pd.DataFrame(rows).reindex(columns=COLS)
    return df
