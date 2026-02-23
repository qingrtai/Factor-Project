# experiments/iterative_baseline_with_reports/memory_manager_reports.py
"""
记忆管理器（带报告版本）
核心改动：
1. load_memory() 返回 (codes, reports) 而非 (codes, train_scores)
2. 标准列包含 factor_report
3. 保存时确保 factor_report 列存在
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import re
import pandas as pd
import numpy as np

# ===== 导入配置 =====
from shared.paths import results_dir as get_results_dir

try:
    from .config import CONFIG as _CFG
except Exception:
    _CFG = None
try:
    from . import config as _cfg_module
except Exception:
    _cfg_module = None


# ===== 配置获取辅助 =====
def _get_cfg(key: str, default=None):
    """从 CONFIG 字典或 config 模块获取配置"""
    if isinstance(_CFG, dict) and key in _CFG:
        return _CFG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default


# ===== 标准列定义（新增 factor_report）=====
_DEFAULT_ROUND_COLUMNS = [
    "factor_id", "code", "n_fields", "fields_used",
    # Train 指标（9个）
    "train_score", "train_coverage", "train_ann_ret", "train_sharpe", 
    "train_max_dd", "train_D", "train_diversity", "train_autocorr", "train_skew",
    # Val 指标（9个）
    "val_score", "val_coverage", "val_ann_ret", "val_sharpe", 
    "val_max_dd", "val_D", "val_diversity", "val_autocorr", "val_skew",
    "factor_report",  # ← 新增：因子报告列
    "status"
]


def _required_columns() -> List[str]:
    """获取必需的列名列表"""
    cols = _get_cfg("ROUND_COLUMNS", None)
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        return list(cols)
    return list(_DEFAULT_ROUND_COLUMNS)


# ===== 列名标准化 =====
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 DataFrame 列名
    - 不改变大小写（保留原始列名）
    - 只做必要的别名映射
    """
    if df is None or df.empty:
        return df
    
    # 创建小写映射（用于查找）
    colmap = {c.lower(): c for c in df.columns}
    
    # code 列映射
    for cand in ["code", "factor_code"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "code"})
            break
    
    # train_score 列映射
    for cand in ["train_score", "trainscore", "train"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "train_score"})
            break
    
    # factor_report 列映射
    for cand in ["factor_report", "report", "desc_report"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "factor_report"})
            break
    
    # D 列映射（大小写修复）
    if "d" in colmap and "D" not in df.columns:
        df = df.rename(columns={colmap["d"]: "D"})
    
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    """读取 CSV 并标准化列名"""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return _normalize_columns(df)


def _ensure_dir(p: Path) -> None:
    """确保目录存在"""
    Path(p).mkdir(parents=True, exist_ok=True)


# ===== 对外接口 =====
@dataclass
class MemoryManager:
    """
    记忆管理器（带报告版本）
    
    核心功能：
    1. load_memory: 读取上一轮的 (codes, reports)
    2. save_round_results: 保存本轮的评估结果（含 factor_report 列）
    """
    
    # 路径配置
    results_dir: Path = Path(_get_cfg("RESULTS_DIR", get_results_dir("iterative_baseline_with_reports")))
    baseline_csv: Path = Path(_get_cfg("BASELINE_FILE", get_results_dir("factor_baseline") / "baseline_factor_metrics.csv"))
    
    # 记忆字段配置
    memory_score_field: str = _get_cfg("MEMORY_SCORE_FIELD", "train_score")
    gpt_memory_field: str = _get_cfg("GPT_MEMORY_FIELD", "factor_report")
    
    def __post_init__(self):
        """初始化时确保目录存在"""
        _ensure_dir(self.results_dir)
    
    # ========== 核心方法：加载记忆 ========== #
    def load_memory(self, previous_round: Optional[int]) -> Tuple[List[str], List[str]]:
        """
        读取上一轮的记忆：(codes, reports)
        
        Args:
            previous_round: 上一轮编号
                - None 或 0: 从 baseline 读取（可能没有 report 列）
                - k (k>0): 从 round_k_factor_metrics.csv 读取
        
        Returns:
            (codes, reports): 代码列表和报告列表
                - codes: List[str] - 因子代码
                - reports: List[str] - 因子报告（baseline 时可能为空字符串）
        
        Raises:
            FileNotFoundError: 如果上一轮文件不存在
            KeyError: 如果缺少 code 列
        """
        # 1. 确定读取路径
        if previous_round is None or previous_round == 0:
            # 第一轮：从 baseline 读取
            csv_path = self.baseline_csv
            source = "baseline"
        else:
            # 后续轮次：从 round CSV 读取
            csv_path = self._round_csv_path(previous_round)
            source = f"round_{previous_round}"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"上一轮文件不存在: {csv_path}")
        
        # 2. 读取并标准化
        df = _read_csv(csv_path)
        
        # 3. 检查必需列
        if 'code' not in df.columns:
            raise KeyError(f"记忆文件缺少 'code' 列: {csv_path}")
        
        # 4. 提取 codes
        codes = [str(c).strip() for c in df['code'].astype(str).tolist()]
        
        # 5. 提取 reports（可能为空）
        if 'factor_report' not in df.columns:
            # baseline 通常没有报告列，返回空字符串
            reports = ["" for _ in range(len(codes))]
        else:
            # 后续轮次有报告列
            reports = df['factor_report'].fillna("").astype(str).tolist()
        
        # 6. 验证长度一致
        if len(codes) != len(reports):
            raise ValueError(f"codes 和 reports 长度不一致: {len(codes)} vs {len(reports)}")
        
        return codes, reports
    
    # ========== 核心方法：保存结果 ========== #
    def save_round_results(self, round_num: int, df_results: pd.DataFrame) -> Path:
        """
        保存本轮评估结果（必须包含 factor_report 列）
        
        Args:
            round_num: 轮次编号
            df_results: 包含评估结果的 DataFrame
                必需列: code, train_score, val_score, factor_report, ...
        
        Returns:
            保存的 CSV 文件路径
        
        Raises:
            ValueError: 如果 df_results 为空或缺少关键列
        """
        # 1. 准备输出路径
        out_path = self._round_csv_path(round_num)
        _ensure_dir(out_path.parent)
        
        # 2. 标准化列名
        cols = _required_columns()
        df = _normalize_columns(df_results.copy() if isinstance(df_results, pd.DataFrame) else pd.DataFrame())
        
        # 3. 自动补齐缺失列
        for c in cols:
            if c not in df.columns:
                if c == 'factor_report':
                    # 报告列默认空字符串（理论上不应该缺失）
                    df[c] = ""
                else:
                    # 其他数值列默认 NaN
                    df[c] = np.nan
        
        # 4. factor_id 自动编号（若缺失）
        if "factor_id" not in df.columns or df["factor_id"].isna().any():
            n = len(df)
            df["factor_id"] = list(range(1, n + 1))
        
        # 5. 数值列稳健化（排除 factor_id, code, fields_used, factor_report, status）
        num_cols = [c for c in cols if c not in ("factor_id", "code", "n_fields", "fields_used", "factor_report", "status")]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # 6. 列序与导出
        df = df[cols]
        df.to_csv(out_path, index=False, encoding="utf-8")
        
        return out_path
    
    # ========== 辅助方法 ========== #
    def _round_csv_path(self, round_num: int) -> Path:
        """获取指定轮次的 CSV 文件路径"""
        return Path(self.results_dir) / f"round_{int(round_num)}_factor_metrics.csv"