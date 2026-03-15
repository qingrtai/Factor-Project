# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/memory_manager.py
"""
FIXED VERSION V2 - 更保守的混合策略

核心改动：
1. Round 1: 100% baseline （不混合，专注学习）
2. Round 2: 100% round 1
3. Round 3+: 100% round 2
"""

import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


from .config import (
    RESULTS_DIR,
    BASELINE_FILE,
    TOP_RATIO,        # 新增
    MIDDLE_RATIO,     # 新增
)

from shared.config_loader import load_global_config
from core.data_loader import load_splits
from core.factor_evaluator import batch_evaluate
from .negative_agents import NegativeAgent

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器（更保守的混合策略）
    
    关键改进：
    - Round 1: 100% baseline （避免过早混合）
    - Round 2: 80% baseline + 20% round 1
    - Round 3+: 70% baseline + 30% previous
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        g = load_global_config()
      # ========== 新增：保存 periods_per_year ========== #
        self.periods_per_year = int(g.get("freq_per_year", 4))
        # ============================================== #
        
        self.logger.info("加载数据切分...")
        self.splits = load_splits(
            raw_csv=g["data"]["raw_file"],
            date_col=g["schema"]["date_col"],
            years=g["years"],
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        self.logger.info("数据切分加载完成")
        self.logger.info(f"Periods per year: {self.periods_per_year}")  # ← 新增日志
        
        self.neg_agent = NegativeAgent(logger=self.logger)

    def _allocate_factors(self, df: pd.DataFrame, source_label: str) -> List[Dict]:
        """
        按比例分层：Top 35% + Middle 30%，Bottom 35% 丢弃
        每条记录带 tier 标签
        """
        df = df.copy()
        df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")
        df = df.sort_values("val_score", ascending=False, na_position="last").reset_index(drop=True)
        
        n = len(df)
        n_top = max(1, int(n * TOP_RATIO))
        n_middle = max(1, int(n * MIDDLE_RATIO))
        
        top_df = df.iloc[:n_top]
        middle_df = df.iloc[n_top:n_top + n_middle]
        # bottom = df.iloc[n_top + n_middle:]  ← 丢弃
        
        records = []
        for _, row in top_df.iterrows():
            records.append({**row.to_dict(), "tier": "top"})
        for _, row in middle_df.iterrows():
            records.append({**row.to_dict(), "tier": "middle"})
        
        self.logger.info(
            f"[memory] {source_label}: {n} 总因子 → "
            f"top={len(top_df)}, middle={len(middle_df)}, "
            f"bottom={n - n_top - n_middle}(丢弃)"
        )
        
        return records

    def get_memory_for_round(self, round_num: int) -> List[Dict]:
        # Round 1: baseline
        if round_num == 1:
            if not os.path.exists(BASELINE_FILE):
                raise FileNotFoundError(f"Baseline not found: {BASELINE_FILE}")
            df = pd.read_csv(BASELINE_FILE)
            return self._allocate_factors(df, source_label=f"Round {round_num} (baseline)")
        
        # Round 2+: previous round
        prev_path = os.path.join(RESULTS_DIR, f"round_{round_num-1}_factor_metrics.csv")
        if not os.path.exists(prev_path):
            raise FileNotFoundError(f"Round {round_num}: 上一轮结果不存在: {prev_path}")
        df = pd.read_csv(prev_path)
        return self._allocate_factors(df, source_label=f"Round {round_num} (round {round_num-1})")

    def get_worst_factors_for_negative_generation(
        self,
        round_num: int,
        n: int = 5,
        positives_for_context: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        生成负样本
        
        改进：Round 1 使用更少的负样本（避免干扰）
        """
        
        positives_for_context = positives_for_context or []
        
        pos_ctx = []
        for r in positives_for_context[:10]:
            pos_ctx.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
            })
        
        self.logger.info(f"[memory] 生成 {n} 个最差因子（round {round_num}）...")
        
        raw_negatives = self.neg_agent.generate_negative_factors(
            current_round=round_num,
            target_n=n,
            context_positives=pos_ctx
        )
        
        if not raw_negatives:
            self.logger.warning("[memory] NegativeAgent 未生成任何负样本")
            return []
        
        factors = [{"code": item.get("code", "")} for item in raw_negatives]
        
        try:
            # ========== 修改：使用实例变量 ========== #
            results_df = batch_evaluate(
                factors=factors,
                splits=self.splits,
                ret_col="ret",
                date_col="datadate",
                periods_per_year=self.periods_per_year,  # ← 改为使用实例变量
                id_start=1
            )
            # ======================================== #
        except Exception as e:
            self.logger.error(f"[memory] 负样本评估失败: {e}")
            return []
        
        negatives = []
        for _, row in results_df.iterrows():
            negatives.append({
                "code": row["code"],
                "train_score": float(row.get("train_score", 0.0) or 0.0),
                "memory_type": "negative",
            })
        
        self.logger.info(f"[memory] 最终获得 {len(negatives)} 个负样本（目标 {n}）")
        return negatives[:n]

    def combine_memory(self, positives: List[Dict], negatives: List[Dict]) -> List[Dict]:
        memory = []
        
        for r in positives:
            memory.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
                "val_score": float(r.get("val_score", 0.0) or 0.0),
                "memory_type": "positive",
                "tier": r.get("tier", "top"),       # 新增：保留 tier
            })
        
        for r in negatives:
            memory.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
                "memory_type": "negative",
                "tier": "negative",                  # 新增：负样本 tier
            })
        
        # 更新日志
        n_top = sum(1 for m in memory if m["tier"] == "top")
        n_mid = sum(1 for m in memory if m["tier"] == "middle")
        n_neg = sum(1 for m in memory if m["tier"] == "negative")
        
        self.logger.info(
            f"[memory] 合并记忆: top={n_top}, middle={n_mid}, negative={n_neg}, "
            f"总计={len(memory)}"
        )
        
        return memory
    
    def save_round_results(self, evaluated_factors: List[Dict], round_num: int) -> str:
        """保存本轮评估结果"""
        path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
        
        df = pd.DataFrame(evaluated_factors)
        df.to_csv(path, index=False)
        
        val_scores = pd.to_numeric(df.get("val_score", pd.Series(dtype=float)), errors="coerce").dropna()
        train_scores = pd.to_numeric(df.get("train_score", pd.Series(dtype=float)), errors="coerce").dropna()
        
        self.logger.info(f"[memory] Round {round_num} 已保存: {path}")
        if len(val_scores) > 0:
            self.logger.info(
                f"  - Val scores: mean={val_scores.mean():.4f}, max={val_scores.max():.4f}"
            )
        if len(train_scores) > 0:
            self.logger.info(
                f"  - Train scores: mean={train_scores.mean():.4f}, max={train_scores.max():.4f}"
            )
        
        return path
    
    def update_aggregated_results(self, round_num: int) -> str:
        """聚合所有轮次结果"""
        rows = []
        
        if os.path.exists(BASELINE_FILE):
            try:
                df = pd.read_csv(BASELINE_FILE)
                df["round_num"] = 0
                rows.append(df)
            except Exception as e:
                self.logger.warning(f"[memory] 读取 baseline 失败: {e}")
        
        for fn in sorted(os.listdir(RESULTS_DIR)):
            if fn.startswith("round_") and fn.endswith("_factor_metrics.csv"):
                fp = os.path.join(RESULTS_DIR, fn)
                try:
                    df = pd.read_csv(fp)
                    round_str = fn.replace("round_", "").replace("_factor_metrics.csv", "")
                    df["round_num"] = int(round_str)
                    rows.append(df)
                except Exception as e:
                    self.logger.warning(f"[memory] 读取 {fn} 失败: {e}")
                    continue
        
        if not rows:
            self.logger.warning("[memory] 没有轮次结果可聚合")
            return ""
        
        all_df = pd.concat(rows, ignore_index=True, sort=False)
        all_df = all_df.sort_values(["round_num", "val_score"], ascending=[True, False])
        
        out_path = os.path.join(RESULTS_DIR, "all_rounds_factors.csv")
        all_df.to_csv(out_path, index=False)
        
        self.logger.info(f"[memory] 聚合结果已更新: {out_path}")
        self.logger.info(f"  - 总轮数: {all_df['round_num'].nunique()}")
        self.logger.info(f"  - 总因子数: {len(all_df)}")
        
        return out_path

    def get_round_statistics(self, round_num: int) -> Optional[Dict[str, Any]]:
        """读取指定轮次统计"""
        if round_num == 0:
            csv_path = os.path.join(RESULTS_DIR, "round_0_factor_metrics.csv")
        else:
            csv_path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
        
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")
        df["train_score"] = pd.to_numeric(df["train_score"], errors="coerce")
        
        return {
            "round": round_num,
            "total_factors": int(len(df)),
            "successful_factors": int((df["status"] == "success").sum()) if "status" in df.columns else len(df),
            "val_score_mean": float(df["val_score"].mean()) if "val_score" in df.columns else 0.0,
            "val_score_max": float(df["val_score"].max()) if "val_score" in df.columns else 0.0,
            "val_score_min": float(df["val_score"].min()) if "val_score" in df.columns else 0.0,
            "train_score_mean": float(df["train_score"].mean()) if "train_score" in df.columns else 0.0,
            "train_score_max": float(df["train_score"].max()) if "train_score" in df.columns else 0.0,
            "train_score_min": float(df["train_score"].min()) if "train_score" in df.columns else 0.0,
        }
