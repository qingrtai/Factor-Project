# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory_with_reports/aggregator_reports.py
"""
aggregator_reports.py — Generate round_summary_mean.csv

FIXED VERSION:
- 修复列名：round -> round_num
- 添加 factors_count, success_count, success_rate
- 指标后缀统一为 _mean, _max, _min, _std
"""

import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .config import RESULTS_DIR, BASELINE_FILE

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """结果聚合器（FIXED 格式）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = RESULTS_DIR
        self.baseline_file = BASELINE_FILE
    
    def generate_round_summary(self) -> str:
        """生成 round_summary_mean.csv (FIXED 格式)"""
        self.logger.info("[aggregator] 开始生成 round_summary_mean.csv...")
        
        all_rounds_data = []
        
        # ========== Round 0 (Baseline) ========== #
        if os.path.exists(self.baseline_file):
            try:
                baseline_stats = self._compute_round_stats(
                    csv_path=self.baseline_file,
                    round_num=0
                )
                if baseline_stats:
                    all_rounds_data.append(baseline_stats)
                    self.logger.info(f"[aggregator] ✓ Round 0 (baseline)")
            except Exception as e:
                self.logger.warning(f"[aggregator] Round 0 统计失败: {e}")
        
        # ========== Round 1+ ========== #
        round_files = sorted([
            f for f in os.listdir(self.results_dir) 
            if f.startswith("round_") and f.endswith("_factor_metrics.csv")
        ])
        
        for fn in round_files:
            try:
                round_str = fn.replace("round_", "").replace("_factor_metrics.csv", "")
                round_num = int(round_str)
                
                csv_path = os.path.join(self.results_dir, fn)
                stats = self._compute_round_stats(csv_path, round_num)
                
                if stats:
                    all_rounds_data.append(stats)
                    self.logger.info(f"[aggregator] ✓ Round {round_num}")
                
            except Exception as e:
                self.logger.warning(f"[aggregator] {fn} 统计失败: {e}")
                continue
        
        if not all_rounds_data:
            self.logger.warning("[aggregator] 没有可聚合的轮次数据")
            return ""
        
        # ========== 生成 DataFrame ========== #
        summary_df = pd.DataFrame(all_rounds_data)
        
        # FIXED: 按 round_num 排序（不是 round）
        summary_df = summary_df.sort_values("round_num")
        
        # FIXED: 确保列顺序
        key_cols = [
            "round_num",           # ← FIXED (was "round")
            "factors_count",       # ← FIXED 添加
            "success_count",       # ← FIXED 添加
            "success_rate",        # ← FIXED 添加
            "train_score_mean", "train_score_max", "train_score_min", "train_score_std",
            "val_score_mean", "val_score_max", "val_score_min", "val_score_std",
            "val_sharpe_mean", "val_sharpe_max", "val_sharpe_min", "val_sharpe_std",
            "val_ann_ret_mean", "val_ann_ret_max", "val_ann_ret_min", "val_ann_ret_std",
            "val_D_mean", "val_D_max", "val_D_min", "val_D_std",
            "val_max_dd_mean", "val_max_dd_max", "val_max_dd_min", "val_max_dd_std",
            "val_coverage_mean", "val_coverage_max", "val_coverage_min", "val_coverage_std",
        ]
        
        # 保留存在的列
        summary_df = summary_df[[c for c in key_cols if c in summary_df.columns]]
        
        # ========== 保存 ========== #
        output_path = os.path.join(self.results_dir, "round_summary_mean.csv")
        summary_df.to_csv(output_path, index=False, float_format="%.6f")
        
        self.logger.info(f"[aggregator] ✓ 已生成: {output_path}")
        self.logger.info(f"[aggregator]   - 总轮数: {len(summary_df)}")
        
        # 打印摘要
        self._print_summary(summary_df)
        
        return output_path
    
    def _compute_round_stats(
        self, 
        csv_path: str, 
        round_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        计算单轮统计指标（FIXED 格式）
        """
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # ========== FIXED: 基础统计 ========== #
        stats = {
            "round_num": round_num,  # ← FIXED (was "round")
            "factors_count": len(df),  # ← FIXED 添加
        }
        
        # FIXED: 成功率统计
        if "status" in df.columns:
            success = (df["status"] == "success").sum()
            stats["success_count"] = int(success)  # ← FIXED 添加
            stats["success_rate"] = float(success / len(df)) if len(df) > 0 else 0.0
        else:
            stats["success_count"] = len(df)  # ← FIXED 添加
            stats["success_rate"] = 1.0
        
        # ========== FIXED: 指标统计 ========== #
        metrics_to_aggregate = [
            "train_score",   # ← FIXED 添加
            "val_score", 
            "val_sharpe", 
            "val_ann_ret", 
            "val_D",
            "val_max_dd",
            "val_coverage",
        ]
        
        for metric in metrics_to_aggregate:
            if metric not in df.columns:
                continue
            
            # 转为数值
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            
            if len(values) == 0:
                continue
            
            # FIXED: 计算统计量（添加 _mean, _max, _min, _std 后缀）
            stats[f"{metric}_mean"] = float(values.mean())
            stats[f"{metric}_max"] = float(values.max())
            stats[f"{metric}_min"] = float(values.min())
            stats[f"{metric}_std"] = float(values.std()) if len(values) > 1 else 0.0
        
        return stats
    
    def _print_summary(self, summary_df: pd.DataFrame) -> None:
        """打印汇总表格"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Round Summary (FIXED 格式)")
        self.logger.info("=" * 80)
        
        key_cols = [
            "round_num",  # ← FIXED
            "factors_count",  # ← FIXED
            "success_rate",
            "val_score_mean", 
            "val_score_max",
            "val_sharpe_mean",
            "val_sharpe_max"
        ]
        
        available_cols = [c for c in key_cols if c in summary_df.columns]
        
        if available_cols:
            display_df = summary_df[available_cols].copy()
            
            # 格式化
            if "success_rate" in display_df.columns:
                display_df["success_rate"] = display_df["success_rate"].apply(lambda x: f"{x:.1%}")
            
            for col in display_df.columns:
                if col.startswith("val_") or col.startswith("train_"):
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            self.logger.info("\n" + display_df.to_string(index=False))
        
        self.logger.info("=" * 80 + "\n")
