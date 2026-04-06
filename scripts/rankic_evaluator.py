"""
scripts/rankic_evaluator.py
────────────────────────────
对已有的 factor_metrics CSV 补算 RankIC / RankICIR 指标。

用法：
    # 单个文件
    python scripts/rankic_evaluator.py \
        --input  experiments/.../round_2_factor_metrics.csv \
        --output results/round_2_rankic.csv

    # 多个文件（一次加载数据，批量处理）
    python scripts/rankic_evaluator.py \
        --input  round_1_factor_metrics.csv round_2_factor_metrics.csv round_3_factor_metrics.csv \
        --output results/all_rankic.csv

    # 目录模式（自动匹配 *factor_metrics*.csv）
    python scripts/rankic_evaluator.py \
        --input  experiments/iterative_baseline_with_reports/results/ \
        --output experiments/iterative_baseline_with_reports/results/all_rankic.csv

原理：
    RankIC  = mean of per-period Spearman correlation (factor_score vs next-period ret)
    RankICIR = mean(RankIC) / std(RankIC)

方向判定（v2 修正）：
    使用训练集 RankIC 符号判定因子方向。若 train_RankIC < 0 则翻转，
    确保所有因子方向一致，避免正负抵消导致均值被压低。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── 项目根目录加入 sys.path（从 scripts/ 运行时需要）──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_loader import load_splits
from core.factor_exec import safe_execute

# ══════════════════════════════════════════════════════════
# 配置（与 global.yaml 保持一致）
# ══════════════════════════════════════════════════════════
RAW_FILE    = "data/raw/1961-2025.csv"
DATE_COL    = "datadate"
ID_COL      = "gvkey"
RET_COL     = "ret"
YEARS       = {"train": (1981, 2008), "val": (2009, 2014), "test": (2015, 2021)}

# 因子稳健化参数（与 factor_evaluator.py 保持一致）
FACTOR_CLIP_Q = (0.005, 0.995)


# ══════════════════════════════════════════════════════════
# 核心计算
# ══════════════════════════════════════════════════════════
def compute_rankic_series(
    data: pd.DataFrame,
    factor: pd.Series,
    date_col: str = DATE_COL,
    ret_col: str = RET_COL,
) -> list[float]:
    """
    逐截面计算 Spearman 秩相关 (factor_score vs ret)。
    返回：每个截面期的 RankIC 值列表。
    """
    df = pd.DataFrame({
        "date":   pd.to_datetime(data[date_col]),
        "factor": pd.to_numeric(factor, errors="coerce"),
        "ret":    pd.to_numeric(data[ret_col], errors="coerce"),
    }).dropna(subset=["date", "factor", "ret"])

    if df.empty:
        return []

    rankic_list = []
    for _, grp in df.groupby("date", sort=True):
        if len(grp) < 10:          # 截面样本太少则跳过
            continue
        corr, _ = spearmanr(grp["factor"], grp["ret"])
        if np.isfinite(corr):
            rankic_list.append(corr)

    return rankic_list


def summarize_rankic(rankic_list: list[float]) -> dict:
    """
    汇总 RankIC 序列 → RankIC (mean) + RankICIR (mean/std)。
    """
    if not rankic_list:
        return {"RankIC": np.nan, "RankICIR": np.nan, "RankIC_std": np.nan, "n_periods": 0}

    arr = np.array(rankic_list)
    mean_ic = float(np.mean(arr))
    std_ic  = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
    icir    = mean_ic / std_ic if (std_ic and std_ic > 1e-9) else np.nan

    return {
        "RankIC":     round(mean_ic, 6),
        "RankICIR":   round(icir, 6),
        "RankIC_std": round(std_ic, 6) if np.isfinite(std_ic) else np.nan,
        "n_periods":  len(arr),
    }


# ══════════════════════════════════════════════════════════
# 因子预处理（与 factor_evaluator 对齐）
# ══════════════════════════════════════════════════════════
def _soft_clip(x: pd.Series, qlo: float, qhi: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return x
    lo, hi = x.quantile([qlo, qhi])
    return x.clip(lo, hi)


def preprocess_factor(factor_raw: pd.Series) -> pd.Series:
    """与 factor_evaluator.batch_evaluate 中的预处理保持一致。"""
    f = _soft_clip(factor_raw, *FACTOR_CLIP_Q)
    f = np.tanh(f / 2.0)
    return f


# ══════════════════════════════════════════════════════════
# 方向判定（v2：基于训练集 RankIC 符号）
# ══════════════════════════════════════════════════════════
def _determine_sign(train_data: pd.DataFrame, factor: pd.Series) -> float:
    """
    用训练集 RankIC 的符号判定因子方向。

    v2 修正：直接计算训练集上的 RankIC 均值，
    若为负则返回 -1（需翻转），否则返回 +1。

    相比 v1（基于 top10%/bottom10% 多空收益差）：
    - 与 RankIC 指标本身完全一致，不会出现判定方向与实际 RankIC 符号矛盾的情况
    - 保证翻转后所有因子的 train_RankIC > 0，避免正负抵消
    """
    rankic_list = compute_rankic_series(train_data, factor)

    if not rankic_list:
        return 1.0

    mean_rankic = float(np.mean(rankic_list))
    return -1.0 if mean_rankic < 0 else 1.0


# ══════════════════════════════════════════════════════════
# 均值汇总行
# ══════════════════════════════════════════════════════════
def _append_mean_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 DataFrame 末尾追加一行均值汇总。
    - 仅对 success 的因子计算均值
    - 数值列取 mean，非数值列留空
    - factor_id 标记为 "MEAN"
    """
    success = df[df["status"] == "success"] if "status" in df.columns else df

    mean_row = {}
    for col in df.columns:
        if col == "factor_id":
            mean_row[col] = "MEAN"
        elif col == "source_file":
            mean_row[col] = "--- MEAN ---"
        elif col == "status":
            mean_row[col] = f"mean of {len(success)} factors"
        elif col in ("code", "fields_used", "factor_report"):
            mean_row[col] = ""
        else:
            # 尝试数值均值
            if col in success.columns:
                vals = pd.to_numeric(success[col], errors="coerce")
                if vals.notna().any():
                    mean_row[col] = round(float(vals.mean()), 6)
                else:
                    mean_row[col] = np.nan
            else:
                mean_row[col] = np.nan

    mean_df = pd.DataFrame([mean_row])
    return pd.concat([df, mean_df], ignore_index=True)


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def evaluate_factors(input_csv: str, splits: dict) -> pd.DataFrame:
    """
    读取一个 factor_metrics CSV，对每个因子补算 RankIC/RankICIR。
    返回 DataFrame，列：factor_id, code, train_RankIC, train_RankICIR, val_RankIC, val_RankICIR, test_RankIC, test_RankICIR, ...
    """
    df_factors = pd.read_csv(input_csv)

    if "code" not in df_factors.columns:
        raise ValueError(f"输入 CSV 缺少 'code' 列: {input_csv}")

    results = []

    for _, row in df_factors.iterrows():
        factor_id = row.get("factor_id", _)
        code = row["code"]
        result = {"factor_id": factor_id, "code": code}

        try:
            # 1) 在各 split 上执行因子代码
            factors_by_split = {}
            for split_name in ["train", "val", "test"]:
                raw_factor = safe_execute(code, splits[split_name])
                factors_by_split[split_name] = preprocess_factor(raw_factor)

            # 2) 用训练集 RankIC 符号判定方向（v2 修正）
            sign = _determine_sign(splits["train"], factors_by_split["train"])
            for k in factors_by_split:
                factors_by_split[k] = factors_by_split[k] * sign

            # 3) 逐 split 计算 RankIC / RankICIR
            for split_name in ["train", "val", "test"]:
                ic_list = compute_rankic_series(
                    splits[split_name], factors_by_split[split_name]
                )
                stats = summarize_rankic(ic_list)
                result[f"{split_name}_RankIC"]     = stats["RankIC"]
                result[f"{split_name}_RankICIR"]   = stats["RankICIR"]
                result[f"{split_name}_RankIC_std"]  = stats["RankIC_std"]
                result[f"{split_name}_n_periods"]   = stats["n_periods"]

            result["sign"] = int(sign)
            result["status"] = "success"

        except Exception as e:
            for split_name in ["train", "val", "test"]:
                result[f"{split_name}_RankIC"]     = np.nan
                result[f"{split_name}_RankICIR"]   = np.nan
                result[f"{split_name}_RankIC_std"]  = np.nan
                result[f"{split_name}_n_periods"]   = 0
            result["sign"] = 0
            result["status"] = f"fail: {e}"

        results.append(result)
        print(f"  Factor {factor_id}: "
              f"train_RankIC={result.get('train_RankIC', 'N/A'):.4f}  "
              f"val_RankIC={result.get('val_RankIC', 'N/A'):.4f}  "
              f"test_RankIC={result.get('test_RankIC', 'N/A'):.4f}  "
              f"sign={result.get('sign', '?')}  "
              f"[{result['status']}]")

    return pd.DataFrame(results)


def _resolve_csv_files(input_paths: list[str]) -> list[str]:
    """
    将 --input 参数解析为 CSV 文件列表。
    支持三种模式：
      1. 单个目录 → glob *factor_metrics*.csv
      2. 多个文件 → 直接使用
      3. 单个文件 → 直接使用
    """
    # 如果只传了一个参数且是目录 → 目录模式
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        csv_files = sorted(glob.glob(os.path.join(input_paths[0], "*factor_metrics*.csv")))
        print(f"\n[2/3] 目录模式：发现 {len(csv_files)} 个 factor_metrics 文件")
        return csv_files

    # 否则视为显式文件列表
    csv_files = []
    for p in input_paths:
        if not os.path.isfile(p):
            print(f"  ⚠ 跳过不存在的文件: {p}")
            continue
        csv_files.append(p)

    print(f"\n[2/3] 文件模式：共 {len(csv_files)} 个 CSV")
    return csv_files


def main():
    parser = argparse.ArgumentParser(description="补算 RankIC / RankICIR 指标（v2：基于 RankIC 符号判定方向）")
    parser.add_argument("--input", "-i", required=True, nargs="+",
                        help="一个或多个 factor_metrics CSV 文件，或一个目录")
    parser.add_argument("--output", "-o", required=True,
                        help="输出 CSV 路径")
    parser.add_argument("--data", "-d", default=RAW_FILE,
                        help=f"原始数据文件路径 (默认: {RAW_FILE})")
    args = parser.parse_args()

    # 1) 加载数据 —— 只加载一次
    print(f"[1/3] 加载数据: {args.data}")
    splits = load_splits(
        raw_csv=args.data,
        date_col=DATE_COL,
        years={"train": YEARS["train"], "val": YEARS["val"], "test": YEARS["test"]},
        id_col=ID_COL,
        ret_col=RET_COL,
    )
    print(f"  Train: {len(splits['train']):,} rows | "
          f"Val: {len(splits['val']):,} rows | "
          f"Test: {len(splits['test']):,} rows")

    # 2) 解析输入文件
    csv_files = _resolve_csv_files(args.input)

    if not csv_files:
        print("没有找到任何 CSV 文件，退出。")
        sys.exit(1)

    # 3) 逐文件计算（数据已加载，不再重复读取）
    all_results = []
    for csv_path in csv_files:
        round_name = os.path.basename(csv_path).replace("_factor_metrics.csv", "")
        print(f"\n── {round_name} ──")
        df_result = evaluate_factors(csv_path, splits)
        df_result.insert(0, "source_file", os.path.basename(csv_path))
        all_results.append(df_result)

    # 4) 合并 & 追加均值汇总行
    df_all = pd.concat(all_results, ignore_index=True)
    df_all = _append_mean_row(df_all)

    df_all.to_csv(args.output, index=False)
    print(f"\n[3/3] 结果已保存: {args.output}")
    print(f"  共 {len(df_all) - 1} 个因子 + 1 行均值汇总 | "
          f"成功 {(df_all['status'].str.startswith('success') if 'status' in df_all.columns else pd.Series()).sum()} | "
          f"失败 {(df_all['status'].str.startswith('fail') if 'status' in df_all.columns else pd.Series()).sum()}")

    # 打印汇总（从 MEAN 行直接读取）
    mean_row = df_all[df_all["factor_id"] == "MEAN"]
    if not mean_row.empty:
        print("\n══ 均值汇总（最后一行） ══")
        for split in ["train", "val", "test"]:
            ic_col = f"{split}_RankIC"
            ir_col = f"{split}_RankICIR"
            if ic_col in mean_row.columns:
                ic_val = mean_row[ic_col].values[0]
                ir_val = mean_row[ir_col].values[0] if ir_col in mean_row.columns else np.nan
                print(f"  {split:5s} RankIC={ic_val:.4f}  RankICIR={ir_val:.4f}")


if __name__ == "__main__":
    main()
