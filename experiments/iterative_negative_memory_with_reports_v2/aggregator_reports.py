# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory_with_reports_v2/aggregator_reports.py
"""
aggregator_reports.py — Generate round_summary_mean.csv

FIXED VERSION - 对齐简化格式:
- 列名：round (不是 round_num)
- 只保留平均值（不要 _mean, _max, _min, _std 后缀）
- 去掉 factors_count, success_count, success_rate
"""

import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .config import RESULTS_DIR, BASELINE_FILE

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """结果聚合器（简化格式版）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = RESULTS_DIR
        self.baseline_file = BASELINE_FILE
    
    def generate_round_summary(self) -> str:
        """生成 round_summary_mean.csv (简化格式)"""
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
        
        # 按 round 排序
        summary_df = summary_df.sort_values("round")
        
        # ========== FIXED: 确保列顺序 ========== #
        key_cols = [
            "round",              # ← 改为 round（不是 round_num）
            "train_score",        # ← 只保留平均值（不要 _mean 后缀）
            "val_score",
            "val_coverage",
            "val_ann_ret",
            "val_sharpe",
            "val_max_dd",
            "val_D",
            # 可选指标（如果有的话）
            "val_diversity",
            "val_autocorr",
            "val_skew",
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
        计算单轮统计指标（简化格式）
        
        返回格式：
        {
            "round": 0,
            "train_score": 0.28,  # 平均值
            "val_score": 1.94,    # 平均值
            ...
        }
        """
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # ========== FIXED: 基础统计（只保留 round）========== #
        stats = {
            "round": round_num,  # ← 改为 round
        }
        
        # ========== FIXED: 指标统计（只计算平均值）========== #
        metrics_to_aggregate = [
            "train_score",
            "val_score", 
            "val_sharpe", 
            "val_ann_ret", 
            "val_D",
            "val_max_dd",
            "val_coverage",
            # 可选指标
            "val_diversity",
            "val_autocorr",
            "val_skew",
        ]
        
        for metric in metrics_to_aggregate:
            if metric not in df.columns:
                continue
            
            # 转为数值
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            
            if len(values) == 0:
                continue
            
            # FIXED: 只保留平均值（不要 _mean 后缀）
            stats[metric] = float(values.mean())
        
        return stats
    
    def _print_summary(self, summary_df: pd.DataFrame) -> None:
        """打印汇总表格"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Round Summary (简化格式)")
        self.logger.info("=" * 80)
        
        # 显示所有列
        display_cols = list(summary_df.columns)
        
        if display_cols:
            display_df = summary_df.copy()
            
            # 格式化数值列
            for col in display_df.columns:
                if col != "round":
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            self.logger.info("\n" + display_df.to_string(index=False))
        
        self.logger.info("=" * 80 + "\n")


# ========== 独立运行入口 ========== #

def main():
    """独立运行 aggregator（手动聚合）"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到 sys.path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 执行聚合
    aggregator = ResultsAggregator()
    
    # 生成 round_summary_mean.csv
    summary_path = aggregator.generate_round_summary()
    
    if summary_path:
        print(f"\n✓ 成功生成: {summary_path}")
    else:
        print("✗ 聚合失败（无数据）")


if __name__ == "__main__":
    main()
