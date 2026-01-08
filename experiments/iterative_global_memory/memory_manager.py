# experiments/iterative_global_memory/memory_manager.py
"""
全局记忆管理器 - 综合优化版
结合了iterative_baseline的稳健性和旧版本的强大功能

核心特点：
1. 自动标准化baseline文件（补齐缺失列、生成D等）
2. 累积记忆加载（所有历史轮次）
3. flatten_for_gpt方法（专为GPT优化的记忆展平）
4. 完整的早停逻辑
5. 详细的统计摘要
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import pandas as pd
import numpy as np
import logging
import shutil

# ===== 导入共享模块 =====
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
    """从CONFIG字典或config模块获取配置"""
    if isinstance(_CFG, dict) and key in _CFG:
        return _CFG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default

# ===== 内部工具 =====
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
    """获取必需的列名列表"""
    cols = _get_cfg("ROUND_COLUMNS", None)
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        return list(cols)
    return list(_DEFAULT_ROUND_COLUMNS)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化DataFrame列名"""
    if df is None or df.empty:
        return df
    
    # 针对常用别名做映射（大小写不敏感）
    colmap = {c.lower(): c for c in df.columns}
    
    # code
    for cand in ["code"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "code"})
            break
    
    # train_score
    for cand in ["train_score", "trainscore", "train"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "train_score"})
            break
    
    # val_score
    for cand in ["val_score", "valscore", "val"]:
        if cand in colmap:
            df = df.rename(columns={colmap[cand]: "val_score"})
            break
    
    # 修复小写'd'到大写'D'
    if "d" in colmap and "D" not in df.columns:
        df = df.rename(columns={colmap["d"]: "D"})
    
    return df

def _read_csv(path: Path) -> pd.DataFrame:
    """读取CSV并标准化列名"""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return _normalize_columns(df)

def _ensure_dir(p: Path) -> None:
    """确保目录存在"""
    Path(p).mkdir(parents=True, exist_ok=True)

# ===== 对外接口 =====
@dataclass
class GlobalMemoryManager:
    """
    全局记忆管理器
    结合了iterative_baseline的稳健性和旧版本的强大功能
    """
    project_root: Path = Path(_get_cfg("PROJECT_ROOT", Path(__file__).resolve().parent))
    results_dir: Path = Path(_get_cfg("RESULTS_DIR", get_results_dir("iterative_global_memory")))
    baseline_csv: Path = Path(_get_cfg("BASELINE_FILE", get_results_dir("factor_baseline") / "baseline_factor_metrics.csv"))
    memory_score_field: str = _get_cfg("MEMORY_SCORE_FIELD", "train_score")
    strict_memory: bool = bool(_get_cfg("STRICT_MEMORY", True))
    max_memory_items: int = int(_get_cfg("MAX_MEMORY_ITEMS", 2000))
    gpt_reference_score: str = _get_cfg("GPT_REFERENCE_SCORE", "val")  # GPT学习用的分数类型

    def __post_init__(self):
        # 设置日志
        self.logger = logging.getLogger("GlobalMemoryManager")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)
        
        _ensure_dir(self.results_dir)
        
        # 强制memory_score_field为train_score（用于early stopping等）
        if str(self.memory_score_field).lower() != "train_score":
            self.memory_score_field = "train_score"
        
        # 但GPT参考分数可以是val（用于学习）
        # 这是一个重要区别：早停用train，GPT学习用val

    # ========== 基础：初始化/标准化 ========== #
    def _standardize_baseline_file(self):
        """
        自动标准化baseline文件
        - 列名映射：val_sharpe→sharpe, val_ann_ret→ann_ret等
        - 生成D字段：D = 1 - max_dd（如果缺失）
        - 补齐缺失列：train_score, test_score等
        - 回填val_score：(sharpe+ann_ret+D)/3（如果缺失）
        """
        if not self.baseline_csv.exists():
            self.logger.warning(f"Baseline文件不存在: {self.baseline_csv}")
            return
        
        df = pd.read_csv(self.baseline_csv)
        
        # 1) 列名映射到新规范
        if 'sharpe' not in df.columns and 'val_sharpe' in df.columns:
            df['sharpe'] = df['val_sharpe']
        if 'ann_ret' not in df.columns and 'val_ann_ret' in df.columns:
            df['ann_ret'] = df['val_ann_ret']
        if 'max_dd' not in df.columns:
            if 'val_max_dd' in df.columns:
                df['max_dd'] = df['val_max_dd']
            elif 'max_drawdown' in df.columns:
                df['max_dd'] = df['max_drawdown']
        
        # 2) 生成/补齐 D
        if 'D' not in df.columns:
            if 'val_D' in df.columns:
                df['D'] = df['val_D']
            elif 'val_max_dd' in df.columns:
                df['D'] = 1.0 - pd.to_numeric(df['val_max_dd'], errors='coerce').fillna(0.0)
            elif 'max_dd' in df.columns:
                df['D'] = 1.0 - pd.to_numeric(df['max_dd'], errors='coerce').fillna(0.0)
            else:
                df['D'] = 0.5  # 默认中性值
        
        # 3) 补齐分数字段
        if 'train_score' not in df.columns:
            df['train_score'] = 0.0
        if 'val_score' not in df.columns:
            # 尝试用 (sharpe+ann_ret+D)/3 回填
            if all(c in df.columns for c in ['sharpe', 'ann_ret', 'D']):
                s = pd.to_numeric(df['sharpe'], errors='coerce')
                a = pd.to_numeric(df['ann_ret'], errors='coerce')
                d = pd.to_numeric(df['D'], errors='coerce')
                df['val_score'] = ((s.fillna(0.0) + a.fillna(0.0) + d.fillna(0.0)) / 3.0).astype(float)
            else:
                df['val_score'] = 0.0
        
        # 4) 其它常用列兜底
        if 'diversity' not in df.columns:
            df['diversity'] = 0.0
        if 'coverage' not in df.columns:
            df['coverage'] = 0.0
        
        # 5) factor_id 补齐
        if 'factor_id' not in df.columns or df['factor_id'].isnull().any():
            df['factor_id'] = [f"base_{i+1:03d}" for i in range(len(df))]
        
        # 6) 数值化关键列
        num_cols = [c for c in _required_columns() if c not in ('factor_id', 'code', 'n_fields', 'fields_used', 'status')]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
        # 7) 保存回文件
        df.to_csv(self.baseline_csv, index=False)
        self.logger.info("Baseline已标准化（补齐缺失列并完成映射）")

    # ========== 核心方法：加载累积记忆 ========== #
    def _ensure_factor_id(self, df: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """确保factor_id存在"""
        if 'factor_id' not in df.columns:
            df = df.copy()
            df['factor_id'] = [f"r{round_num}_{i+1:03d}" for i in range(len(df))]
            return df
        if df['factor_id'].isnull().any():
            df = df.copy()
            for i, is_na in enumerate(df['factor_id'].isnull()):
                if is_na:
                    df.at[df.index[i], 'factor_id'] = f"r{round_num}_{i+1:03d}"
        return df

    def _coerce_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """强制转换分数列为数值"""
        df = df.copy()
        if 'val_score' in df.columns:
            df['val_score'] = pd.to_numeric(df['val_score'], errors='coerce')
        if 'train_score' in df.columns:
            df['train_score'] = pd.to_numeric(df['train_score'], errors='coerce')
        else:
            df['train_score'] = np.nan
        return df

    def _pack_factors_for_memory(self, df: pd.DataFrame) -> Tuple[List[Dict], pd.Series, pd.Series]:
        """
        将DataFrame打包为记忆格式
        返回: (factors, val_scores, train_scores)
        """
        df = self._coerce_scores(df)
        
        # 根据GPT参考分数类型决定使用哪个分数
        score_col = 'val_score' if self.gpt_reference_score == 'val' else 'train_score'
        
        # 至少需要有效的score
        df_ok = df.dropna(subset=[score_col]) if score_col in df.columns else df
        
        factors = []
        for _, r in df_ok.iterrows():
            factor = {
                'factor_id': r.get('factor_id', ''),
                'code': r.get('code', ''),
            }
            
            # 添加train_score（如果有）
            if 'train_score' in df.columns and pd.notna(r.get('train_score')):
                factor['train_score'] = float(r['train_score'])
            
            # 添加val_score（如果有）
            if 'val_score' in df.columns and pd.notna(r.get('val_score')):
                factor['val_score'] = float(r['val_score'])
            
            factors.append(factor)
        
        val_scores = df_ok['val_score'] if 'val_score' in df_ok.columns else pd.Series()
        train_scores = df_ok['train_score'] if 'train_score' in df_ok.columns else pd.Series()
        
        return factors, val_scores, train_scores

    def _load_baseline_memory(self) -> Dict:
        """加载baseline记忆（作为round 0）"""
        df = _read_csv(self.baseline_csv)
        df = self._ensure_factor_id(df, round_num=0)
        factors, val_scores, train_scores = self._pack_factors_for_memory(df)
        
        return {
            'round': 0,
            'type': 'baseline',
            'count': len(factors),
            'best_val_score': float(val_scores.max()) if len(val_scores) > 0 else 0.0,
            'avg_val_score': float(val_scores.mean()) if len(val_scores) > 0 else 0.0,
            'best_train_score': float(train_scores.max()) if len(train_scores.dropna()) > 0 else 0.0,
            'avg_train_score': float(train_scores.mean()) if len(train_scores.dropna()) > 0 else 0.0,
            'factors': factors
        }

    def _load_round_memory(self, round_num: int) -> Optional[Dict]:
        """加载指定轮次的记忆"""
        round_file = self._round_csv_path(round_num)
        if not round_file.exists():
            self.logger.warning(f"轮次文件不存在: {round_file}")
            return None
        
        df = _read_csv(round_file)
        df = self._ensure_factor_id(df, round_num=round_num)
        factors, val_scores, train_scores = self._pack_factors_for_memory(df)
        
        return {
            'round': round_num,
            'type': 'iteration',
            'count': len(factors),
            'best_val_score': float(val_scores.max()) if len(val_scores) > 0 else 0.0,
            'avg_val_score': float(val_scores.mean()) if len(val_scores) > 0 else 0.0,
            'best_train_score': float(train_scores.max()) if len(train_scores.dropna()) > 0 else 0.0,
            'avg_train_score': float(train_scores.mean()) if len(train_scores.dropna()) > 0 else 0.0,
            'factors': factors
        }

    def get_cumulative_memory(self, current_round: int) -> List[Dict]:
        """
        获取累积记忆（所有历史轮次）
        
        Args:
            current_round: 当前轮次（加载0到current_round-1的记忆）
        
        Returns:
            累积记忆列表，每个元素包含：
            {
                'round': 轮次编号,
                'type': 'baseline' or 'iteration',
                'count': 因子数量,
                'best_val_score': 最佳val分数,
                'avg_val_score': 平均val分数,
                'best_train_score': 最佳train分数,
                'avg_train_score': 平均train分数,
                'factors': [{'factor_id': ..., 'code': ..., 'train_score': ..., 'val_score': ...}]
            }
        """
        cumulative = [self._load_baseline_memory()]
        for r in range(1, current_round):
            mem = self._load_round_memory(r)
            if mem:
                cumulative.append(mem)
        return cumulative

    # ========== GPT专用：扁平化记忆 ========== #
    def flatten_for_gpt(
        self,
        current_round: int,
        top_k_per_round: Optional[int] = None,
        deduplicate: bool = True
    ) -> List[Dict]:
        """
        扁平化记忆，专门给GPT使用
        
        Args:
            current_round: 当前轮次
            top_k_per_round: 每轮取top-k个因子（按score排序）
            deduplicate: 是否全局去重（同code只保留高分的）
        
        Returns:
            扁平化的因子列表：
            [{'round': 0, 'factor_id': ..., 'code': ..., 'train_score': ..., 'val_score': ...}, ...]
        """
        score_key = 'val_score' if self.gpt_reference_score == 'val' else 'train_score'
        
        packs = []
        for mem in self.get_cumulative_memory(current_round):
            rows = mem['factors']
            
            # 每轮取top-k
            if top_k_per_round is not None and top_k_per_round > 0:
                rows = sorted(
                    rows, 
                    key=lambda x: x.get(score_key, 0.0), 
                    reverse=True
                )[:top_k_per_round]
            
            for f in rows:
                packs.append({'round': mem['round'], **f})
        
        # 全局去重
        if deduplicate:
            best_by_code: Dict[str, Dict] = {}
            for x in packs:
                key = x['code'].strip()
                curr_score = x.get(score_key, 0.0)
                if key not in best_by_code:
                    best_by_code[key] = x
                elif curr_score > best_by_code[key].get(score_key, 0.0):
                    best_by_code[key] = x
            packs = list(best_by_code.values())
        
        # 排序
        packs.sort(key=lambda x: (x['round'], -x.get(score_key, 0.0)))
        return packs

    # ========== 保存结果 ========== #
    def save_round_results(self, round_num: int, df_results: pd.DataFrame) -> Path:
        """
        保存本轮评估结果
        
        Args:
            round_num: 轮次编号
            df_results: 包含评估结果的DataFrame
        
        Returns:
            保存的CSV文件路径
        """
        out_path = self._round_csv_path(round_num)
        _ensure_dir(out_path.parent)
        
        cols = _required_columns()
        df = _normalize_columns(df_results.copy() if isinstance(df_results, pd.DataFrame) else pd.DataFrame())
        
        # 自动补齐必要列
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        
        # factor_id 自动编号（若缺失）
        if "factor_id" not in df.columns or df["factor_id"].isna().any():
            n = len(df)
            df["factor_id"] = list(range(1, n + 1))
        
        # 数值列稳健化
        num_cols = [c for c in cols if c not in ("factor_id", "code", "n_fields", "fields_used", "status")]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # 列序与导出
        df = df[cols]
        df.to_csv(out_path, index=False, encoding="utf-8")
        
        # 记录统计
        val = pd.to_numeric(df['val_score'], errors='coerce').dropna()
        train = pd.to_numeric(df['train_score'], errors='coerce').dropna()
        self.logger.info(f"Round {round_num} 已保存: {out_path} | n={len(df)}")
        self.logger.info(f"  Val分数  - best={float(val.max()) if len(val) > 0 else 0.0:.4f}, avg={float(val.mean()) if len(val) > 0 else 0.0:.4f}")
        self.logger.info(f"  Train分数 - best={float(train.max()) if len(train) > 0 else 0.0:.4f}, avg={float(train.mean()) if len(train) > 0 else 0.0:.4f}")
        
        return out_path

    # ========== 统计与早停 ========== #
    def get_memory_summary(self, current_round: int) -> Dict:
        """
        获取记忆摘要统计（旧版本的优秀功能）
        
        Returns:
            详细的摘要字典
        """
        cumulative = self.get_cumulative_memory(current_round)
        
        vals = []
        trains = []
        for mem in cumulative:
            for f in mem['factors']:
                if 'val_score' in f:
                    vals.append(f['val_score'])
                if 'train_score' in f:
                    trains.append(f['train_score'])
        
        return {
            'total_rounds': len(cumulative),
            'total_factors': sum(mem['count'] for mem in cumulative),
            'overall_best_val_score': float(np.max(vals)) if len(vals) > 0 else 0.0,
            'overall_avg_val_score': float(np.mean(vals)) if len(vals) > 0 else 0.0,
            'overall_best_train_score': float(np.max(trains)) if len(trains) > 0 else 0.0,
            'overall_avg_train_score': float(np.mean(trains)) if len(trains) > 0 else 0.0,
            'round_details': [
                {
                    'round': mem['round'], 
                    'type': mem['type'], 
                    'count': mem['count'],
                    'best_val_score': mem['best_val_score'], 
                    'avg_val_score': mem['avg_val_score'],
                    'best_train_score': mem['best_train_score'], 
                    'avg_train_score': mem['avg_train_score']
                }
                for mem in cumulative
            ]
        }

    def round_avg_val(self, round_num: int) -> Optional[float]:
        """获取指定轮次的平均val_score"""
        mem = self._load_baseline_memory() if round_num == 0 else self._load_round_memory(round_num)
        return None if (mem is None) else float(mem['avg_val_score'])

    def round_avg_train(self, round_num: int) -> Optional[float]:
        """获取指定轮次的平均train_score"""
        mem = self._load_baseline_memory() if round_num == 0 else self._load_round_memory(round_num)
        return None if (mem is None) else float(mem['avg_train_score'])

    def should_early_stop(
        self,
        current_round: int,
        min_rounds: int = 3,
        compare_to: str = "best",
        min_improve: float = 0.0,
        patience: int = 0,
        score_type: str = "val"
    ) -> Tuple[bool, Dict]:
        """
        判断是否应该早停（旧版本的完整功能）
        
        Args:
            current_round: 当前轮次
            min_rounds: 最少运行轮数
            compare_to: 比较基准 ("best" or "last")
            min_improve: 最小提升阈值
            patience: 容忍轮数
            score_type: 分数类型 ("val" or "train")
        
        Returns:
            (should_stop, detail_dict)
        """
        if current_round <= min_rounds:
            return False, {'reason': 'warmup', 'min_rounds': min_rounds}
        
        avg_func = self.round_avg_train if score_type == "train" else self.round_avg_val
        
        last_round = current_round - 1
        last_avg = avg_func(last_round)
        if last_avg is None:
            return False, {'reason': 'no_last_round_data'}
        
        if compare_to == "last":
            ref_avg = avg_func(last_round - 1)
        else:
            avgs = [avg_func(r) for r in range(0, last_round)]
            avgs = [a for a in avgs if a is not None]
            ref_avg = max(avgs) if avgs else None
        
        if ref_avg is None:
            return False, {'reason': 'no_reference'}
        
        improve = last_avg - ref_avg
        improve_ok = (improve >= min_improve)
        
        series = [avg_func(r) for r in range(0, last_round + 1)]
        series = [x for x in series if x is not None]
        streak = 0
        if len(series) >= 2:
            for i in range(len(series) - 1, 0, -1):
                if series[i] >= series[i - 1] + min_improve:
                    break
                streak += 1
        
        stop = (not improve_ok) and (streak >= patience)
        detail = {
            'score_type': score_type,
            'compare_to': compare_to,
            'ref_avg': ref_avg,
            'last_avg': last_avg,
            'improve': improve,
            'min_improve': min_improve,
            'streak_no_improve': streak,
            'patience': patience
        }
        return stop, detail

    # ========== 辅助方法 ========== #
    def _round_csv_path(self, round_num: int) -> Path:
        """获取指定轮次的CSV文件路径"""
        return Path(self.results_dir) / f"round_{int(round_num)}_factor_metrics.csv"

    def get_all_round_files(self) -> List[Path]:
        """获取所有已存在的轮次文件"""
        files = []
        for p in Path(self.results_dir).glob("round_*_factor_metrics.csv"):
            files.append(p)
        return sorted(files)

    def count_total_factors(self, current_round: int) -> int:
        """统计累积因子总数"""
        cumulative_memory = self.get_cumulative_memory(current_round)
        total = sum(mem['count'] for mem in cumulative_memory)
        return total

    def cleanup_round(self, round_num: int) -> bool:
        """清理指定轮次的文件"""
        f = self._round_csv_path(round_num)
        if f.exists():
            f.unlink()
            self.logger.info(f"已清理Round {round_num}文件: {f}")
            return True
        self.logger.warning(f"Round {round_num}文件不存在，无需清理")
        return False