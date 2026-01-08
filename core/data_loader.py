# core/data_loader.py
from __future__ import annotations
import numpy as np
import pandas as pd
from core.utils import DataValidationError


def ensure_ret_column(
    df: pd.DataFrame,
    ret_col: str = "ret",
    id_col: str | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    统一构造/补齐收益列 ret，优先级：
      1) 若已有 ret，清洗为数值后优先使用；
      2) 若有 retq，则仅在 ret 缺失处用 retq 填补；
      3) 若具备 prccq + id + date，则在“按 id+date 排序”的副本中用 shift(-1) 计算下一期简单收益，
         再按索引对齐回写到原 df，仅在 ret 仍缺失处填补。
    所有步骤均移除 ±inf，保留 NaN。
    """
    df = df.copy()

    # 1) 标准化已有 ret；若无则先建空列
    if ret_col in df.columns:
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        df[ret_col] = np.nan

    # 2) 用 retq 填补缺失
    if "retq" in df.columns:
        retq_numeric = pd.to_numeric(df["retq"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = df[ret_col].isna() & retq_numeric.notna()
        if mask.any():
            df.loc[mask, ret_col] = retq_numeric.loc[mask]

    # 3) 用 prccq 构造（要求 id + date 存在）
    has_px_inputs = id_col and date_col and all(c in df.columns for c in ("prccq", id_col, date_col))
    if has_px_inputs:
        # 在“排序后的副本 out”里计算下一期简单收益
        out = df.sort_values([id_col, date_col]).copy()
        px = pd.to_numeric(out["prccq"], errors="coerce")
        px.loc[px <= 0] = np.nan
        nxt = px.groupby(out[id_col]).shift(-1)
        ret_from_px = (nxt - px) / px
        ret_from_px = ret_from_px.replace([np.inf, -np.inf], np.nan)

        # 关键：按索引对齐回写，只填补 ret 缺失
        ret_from_px.index = out.index                   # 先与排序副本对齐
        ret_aligned = ret_from_px.reindex(df.index)     # 再回到原 df 的行序
        mask = df[ret_col].isna() & ret_aligned.notna()
        if mask.any():
            df.loc[mask, ret_col] = ret_aligned.loc[mask]

    # 最终校验
    if df[ret_col].notna().sum() == 0:
        raise DataValidationError(
            "找不到有效收益：既无可用 ret/retq，也无法基于 prccq（或缺 id/date）计算。"
        )

    return df


def load_splits(
    raw_csv: str,
    date_col: str,
    years: dict,
    id_col: str | None = None,
    ret_col: str = "ret",
) -> dict:
    """
    读取原始 CSV → 标准化日期 → 补齐收益 ret → 按年份切分 train/val/test。
    注意：这里只做数据准备；评分口径固定在 core/factor_evaluator.py。
    """
    df = pd.read_csv(raw_csv, low_memory=False)

    # 日期列校验与标准化
    if date_col not in df.columns:
        raise DataValidationError(f"缺少日期列：{date_col}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise DataValidationError(f"{date_col} 无有效日期，请检查原始数据格式。")

    # 统一构造/补齐收益
    df = ensure_ret_column(df, ret_col=ret_col, id_col=id_col, date_col=date_col)

    # 按年份切分
    tr0, tr1 = years["train"]
    v0, v1   = years["val"]
    te0, te1 = years["test"]

    dtrain = df[(df[date_col].dt.year >= tr0) & (df[date_col].dt.year <= tr1)].copy()
    dval   = df[(df[date_col].dt.year >= v0 ) & (df[date_col].dt.year <= v1 )].copy()
    dtest  = df[(df[date_col].dt.year >= te0) & (df[date_col].dt.year <= te1)].copy()

    # 轻量诊断（可注释）：看下各段的有效 ret 行数 & 日期数
    try:
        def _brief(name: str, x: pd.DataFrame):
            n_valid = pd.to_numeric(x[ret_col], errors="coerce").notna().sum()
            n_dates = x[date_col].dt.normalize().nunique()
            print(f"[split] {name:5s}: {len(x):8d} rows | valid_ret={n_valid:8d} | dates={n_dates:4d}")
        _brief("train", dtrain)
        _brief("val",   dval)
        _brief("test",  dtest)
    except Exception:
        pass

    return {"train": dtrain, "val": dval, "test": dtest}
