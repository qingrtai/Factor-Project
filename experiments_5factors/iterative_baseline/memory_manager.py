# iterative_baseline/memory_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any
import re
import pandas as pd
import numpy as np
import os

# ===== 尝试导入 config；失败则使用默认 =====
from shared.paths import results_dir as get_results_dir  # ← 改个名字

try:
    from experiments.iterative_baseline import config as CFG
except Exception:
    class CFG:
        RESULTS_DIR = get_results_dir("iterative_baseline")  # ← 使用新名字
        BASELINE_FILE = get_results_dir("factor_baseline") / "baseline_factor_metrics.csv"
        BASELINE_MEMORY_FILE = get_results_dir("factor_baseline") / "baseline_memory_train_only.csv"
        MEMORY_SCORE_FIELD = "train_score"

# ===== 内部工具 =====
# === 改动点2：本文件内部的默认列清单也加入 D（保留 max_dd） ===
_DEFAULT_ROUND_COLUMNS = [
    "factor_id", "code", "n_fields", "fields_used",
    # Train 指标（9个）
    "train_score", "train_coverage", "train_ann_ret", "train_sharpe", 
    "train_max_dd", "train_D", "train_diversity", "train_autocorr", "train_skew",
    # Val 指标（9个）
    "val_score", "val_coverage", "val_ann_ret", "val_sharpe", 
    "val_max_dd", "val_D", "val_diversity", "val_autocorr", "val_skew",
    # 可选
    "status"
]

def _required_columns() -> List[str]:
    cols = getattr(CFG, "ROUND_COLUMNS", None)
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        return list(cols)
    cols = getattr(CFG, "REQUIRED_COLUMNS", None)
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        return list(cols)
    return list(_DEFAULT_ROUND_COLUMNS)

def _canon_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # ❶ 不要把所有列名无脑转小写；仅做必要别名映射
    # df = df.rename(columns={c: c.lower() for c in df.columns})   # ← 删掉这一行

    # 针对个别常用别名做宽松规范（大小写不敏感地查找）
    colmap = {c.lower(): c for c in df.columns}

    # code
    for cand in ["code"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "code"})
            break

    # train_score
    for cand in ["train_score","trainscore","train","trainperf","train_performance"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "train_score"})
            break

    # val_score（仅为列齐）
    for cand in ["val_score","valscore","val","validation","validationscore"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "val_score"})
            break

    # ❷ 关键：如果被下游改成了小写 'd'，把它映射回大写 'D'
    if "d" in colmap and "D" not in df.columns:
        df = df.rename(columns={colmap["d"]: "D"})

    return df


def _read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return _normalize_columns(df)

def _ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _get_cfg(key: str, default=None):
    return getattr(CFG, key, default)

# ===== 对外接口 =====
@dataclass
class MemoryManager:
    project_root: Path = Path(_get_cfg("PROJECT_ROOT", Path(__file__).resolve().parent))
    results_dir: Path  = Path(_get_cfg("RESULTS_DIR",  get_results_dir("iterative_baseline")))  # ← 使用新名字
    baseline_csv: Path = Path(_get_cfg("BASELINE_FILE", get_results_dir("factor_baseline") / "baseline_factor_metrics.csv"))
    baseline_train_only: Path = Path(_get_cfg("BASELINE_MEMORY_FILE", get_results_dir("factor_baseline") / "baseline_memory_train_only.csv"))
    memory_score_field: str = _get_cfg("MEMORY_SCORE_FIELD", "train_score")
    strict_memory: bool = bool(_get_cfg("STRICT_MEMORY", True))
    memory_source_preference: tuple = _get_cfg("MEMORY_SOURCE_PREFERENCE", ("round_csv","baseline_train_only","baseline_csv"))

    def __post_init__(self):
        _ensure_dir(self.results_dir)
        # 强制只允许 train_score 作为记忆字段
        if str(self.memory_score_field).lower() != "train_score":
            # 兜底强制
            self.memory_score_field = "train_score"

    # ---------- 公开：读取上一轮记忆（只返回 code + train_score） ----------
    def load_memory(self, previous_round: Optional[int]) -> Tuple[List[str], List[float]]:
        """
        返回 (codes, train_scores)
        - previous_round = None 时（第1轮），采用配置的优先级从 baseline 读取；
        - previous_round = k 时，从 results/round_k_factor_metrics.csv 读取。
        """
        df = None
        tried = []

        # 1) 上一轮 round CSV
        if previous_round is not None and "round_csv" in self.memory_source_preference:
            p = self._round_csv_path(previous_round)
            tried.append(("round_csv", p))
            try:
                df = _read_csv(p)
            except Exception:
                df = None

        # 2) baseline train-only memory
        if df is None and "baseline_train_only" in self.memory_source_preference:
            p = Path(self.baseline_train_only)
            tried.append(("baseline_train_only", p))
            try:
                df = _read_csv(p)
            except Exception:
                df = None

        # 3) baseline full csv（仅取 train_score）
        if df is None and "baseline_csv" in self.memory_source_preference:
            p = Path(self.baseline_csv)
            tried.append(("baseline_csv", p))
            try:
                df = _read_csv(p)
            except Exception:
                df = None

        if df is None or df.empty:
            raise FileNotFoundError(f"无法加载上轮记忆，尝试路径：{tried}")

        # 只抽取 code + train_score
        df = _normalize_columns(df)
        if "code" not in df.columns:
            raise KeyError("记忆文件缺少 'code' 列。")
        if "train_score" not in df.columns:
            # 某些旧的 baseline 可能没有 train_score（极少数情况）
            raise KeyError("记忆文件缺少 'train_score' 列。请先运行基线以生成 train-only memory。")

        codes = [str(c).strip() for c in df["code"].astype(str).tolist()]
        trains = pd.to_numeric(df["train_score"], errors="coerce").fillna(0.0).astype(float).tolist()

        # 严格模式：确保不会把 val_score 暴露到上层
        if self.strict_memory and "val_score" in df.columns:
            # 这里不报错，只是提醒；真正喂给上层的只有 train_score
            pass

        return codes, trains

    # ---------- 公开：保存本轮完整 22 列结果 ----------
    def save_round_results(self, round_num: int, df_results: pd.DataFrame) -> Path:
        """
        将评估结果保存为 results/round_{round_num}_factor_metrics.csv
        - 列名按 baseline 的 11 列（factor_id 放首）对齐；
        - 自动补齐缺失列并做数值化；
        - 不做任何“只记忆 train”裁剪（裁剪发生在 load_memory 阶段）。
        """
        out_path = self._round_csv_path(round_num)
        _ensure_dir(out_path.parent)

        cols = _required_columns()  # 与 baseline 对齐的 11 列（已包含 D）
        df = _normalize_columns(df_results.copy() if isinstance(df_results, pd.DataFrame) else pd.DataFrame())
        # 自动补齐必要列
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        # factor_id 自动编号（若缺失）
        if "factor_id" not in df.columns or df["factor_id"].isna().any():  # ← 先检查列是否存在
            n = len(df)
            df["factor_id"] = list(range(1, n + 1))  # ← 直接赋值，不用 .loc

       # 数值列稳健化（排除 factor_id, code, fields_used）
        num_cols = [c for c in cols if c not in ("factor_id", "code", "n_fields", "fields_used", "status")]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 列序与导出
        df = df[cols]
        df.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    # ---------- 辅助 ----------
    def _round_csv_path(self, round_num: int) -> Path:
        return Path(self.results_dir) / f"round_{int(round_num)}_factor_metrics.csv"
