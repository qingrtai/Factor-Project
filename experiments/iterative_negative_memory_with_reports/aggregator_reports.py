# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory_with_reports/aggregator_reports.py
"""
aggregator_reports.py — Generate round_summary_mean.csv

职责：
1. 读取所有轮次的 CSV（baseline + round_1, round_2, ...）
2. 对每轮计算统计指标
3. 生成 round_summary_mean.csv（每轮一行）

输出格式：
round | factors | success_rate | val_score_mean | val_score_max | val_sharpe_mean | ...
"""

import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 配置导入
from .config import RESULTS_DIR, BASELINE_FILE

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    结果聚合器（生成 round_summary_mean.csv）
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = RESULTS_DIR
        self.baseline_file = BASELINE_FILE
    
    def generate_round_summary(self) -> str:
        """
        生成 round_summary_mean.csv
        
        Returns:
            输出文件路径
        """
        self.logger.info("[aggregator] 开始生成 round_summary_mean.csv...")
        
        # 收集所有轮次数据
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
                # 提取轮次号
                round_str = fn.replace("round_", "").replace("_factor_metrics.csv", "")
                round_num = int(round_str)
                
                # 计算统计
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
        
        # 按 round_num 排序
        summary_df = summary_df.sort_values("round_num")
        
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
        计算单轮统计指标
        
        Args:
            csv_path: 轮次 CSV 文件路径
            round_num: 轮次号（0=baseline）
        
        Returns:
            统计字典
        """
        if not os.path.exists(csv_path):
            return None
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # ========== 基础统计 ========== #
        stats = {
            "round_num": round_num,
            "factors_count": len(df),
        }
        
        # 成功率
        if "status" in df.columns:
            success = (df["status"] == "success").sum()
            stats["success_count"] = int(success)
            stats["success_rate"] = float(success / len(df)) if len(df) > 0 else 0.0
        else:
            stats["success_count"] = len(df)
            stats["success_rate"] = 1.0
        
        # ========== 指标统计 ========== #
        # 定义要统计的指标
        metrics_to_aggregate = [
            "val_score", 
            "train_score", 
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
            
            # 计算统计量
            stats[f"{metric}_mean"] = float(values.mean())
            stats[f"{metric}_max"] = float(values.max())
            stats[f"{metric}_min"] = float(values.min())
            stats[f"{metric}_std"] = float(values.std()) if len(values) > 1 else 0.0
        
        # ========== 特殊统计（Top 3 平均）========== #
        if "val_score" in df.columns:
            df_sorted = df.sort_values("val_score", ascending=False, na_position="last")
            top3 = pd.to_numeric(df_sorted["val_score"], errors="coerce").dropna().head(3)
            if len(top3) > 0:
                stats["val_score_top3_mean"] = float(top3.mean())
        
        return stats
    
    def _print_summary(self, summary_df: pd.DataFrame) -> None:
        """打印汇总表格（便于查看）"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Round Summary (val_score & val_sharpe)")
        self.logger.info("=" * 80)
        
        # 选择关键列
        key_cols = [
            "round_num", 
            "factors_count", 
            "success_rate",
            "val_score_mean", 
            "val_score_max",
            "val_sharpe_mean",
            "val_sharpe_max"
        ]
        
        # 过滤存在的列
        available_cols = [c for c in key_cols if c in summary_df.columns]
        
        if available_cols:
            display_df = summary_df[available_cols].copy()
            
            # 格式化
            if "success_rate" in display_df.columns:
                display_df["success_rate"] = display_df["success_rate"].apply(lambda x: f"{x:.1%}")
            
            for col in display_df.columns:
                if col.startswith("val_"):
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            self.logger.info("\n" + display_df.to_string(index=False))
        
        self.logger.info("=" * 80 + "\n")
    
    def get_best_factors_per_round(self, top_k: int = 3) -> pd.DataFrame:
        """
        获取每轮最佳的 top_k 个因子
        
        Args:
            top_k: 每轮取前 k 个
        
        Returns:
            DataFrame（包含 round_num, factor_id, val_score, code 等）
        """
        all_best = []
        
        # Round 0
        if os.path.exists(self.baseline_file):
            df = pd.read_csv(self.baseline_file)
            df["round_num"] = 0
            df_sorted = df.sort_values("val_score", ascending=False, na_position="last")
            all_best.append(df_sorted.head(top_k))
        
        # Round 1+
        round_files = sorted([
            f for f in os.listdir(self.results_dir) 
            if f.startswith("round_") and f.endswith("_factor_metrics.csv")
        ])
        
        for fn in round_files:
            try:
                round_str = fn.replace("round_", "").replace("_factor_metrics.csv", "")
                round_num = int(round_str)
                
                csv_path = os.path.join(self.results_dir, fn)
                df = pd.read_csv(csv_path)
                df["round_num"] = round_num
                
                df_sorted = df.sort_values("val_score", ascending=False, na_position="last")
                all_best.append(df_sorted.head(top_k))
                
            except Exception as e:
                self.logger.warning(f"[aggregator] {fn} 读取失败: {e}")
                continue
        
        if all_best:
            result = pd.concat(all_best, ignore_index=True)
            result = result.sort_values(["round_num", "val_score"], ascending=[True, False])
            return result
        
        return pd.DataFrame()


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
        
        # 额外：生成 top 3 因子列表
        best_df = aggregator.get_best_factors_per_round(top_k=3)
        if not best_df.empty:
            best_path = os.path.join(RESULTS_DIR, "best_factors_per_round.csv")
            best_df.to_csv(best_path, index=False)
            print(f"✓ 额外生成: {best_path}")
    else:
        print("✗ 聚合失败（无数据）")


if __name__ == "__main__":
    main()