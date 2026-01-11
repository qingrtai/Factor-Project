# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/memory_manager.py
"""
FIXED VERSION V2 - 更保守的混合策略

核心改动：
1. Round 1: 100% baseline （不混合，专注学习）
2. Round 2: 80% baseline + 20% round 1
3. Round 3+: 70% baseline + 30% previous
4. 始终保持 baseline 的主导地位
"""

import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from .config import (
    RESULTS_DIR,
    BASELINE_FILE,
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

    def get_memory_for_round(self, round_num: int) -> List[Dict]:
        """
        返回本轮的正样本记忆（更保守的混合策略）
        
        策略：
        - Round 1: 100% baseline (Top 10) - 专注学习 baseline
        - Round 2: 80% baseline (Top 8) + 20% round 1 (Top 2)
        - Round 3+: 70% baseline (Top 7) + 30% previous (Top 3)
        """
        # ========== Round 1: 100% baseline ========== #
        if round_num <= 1:
            csv_path = BASELINE_FILE
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Baseline not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")
            top10 = df.sort_values("val_score", ascending=False, na_position="last").head(10)
            
            positives = top10.to_dict(orient="records")
            
            self.logger.info(
                f"[memory] Round {round_num} 使用 100% baseline: {len(positives)} 个正样本"
            )
            if len(top10) > 0:
                val_range = f"[{top10['val_score'].min():.4f}, {top10['val_score'].max():.4f}]"
                self.logger.info(f"  - Val score 范围: {val_range}")
            
            return positives
        
        # ========== Round 2: 80% baseline + 20% round 1 ========== #
        if round_num == 2:
            try:
                baseline_df = pd.read_csv(BASELINE_FILE)
                baseline_df["val_score"] = pd.to_numeric(baseline_df["val_score"], errors="coerce")
                baseline_top8 = baseline_df.sort_values("val_score", ascending=False).head(8)
            except Exception as e:
                self.logger.warning(f"读取 baseline 失败: {e}")
                baseline_top8 = pd.DataFrame()
            
            prev_path = os.path.join(RESULTS_DIR, f"round_1_factor_metrics.csv")
            
            if os.path.exists(prev_path):
                try:
                    prev_df = pd.read_csv(prev_path)
                    prev_df["val_score"] = pd.to_numeric(prev_df["val_score"], errors="coerce")
                    prev_top2 = prev_df.sort_values("val_score", ascending=False).head(2)
                except Exception as e:
                    self.logger.warning(f"读取 round 1 失败: {e}")
                    prev_top2 = pd.DataFrame()
            else:
                prev_top2 = pd.DataFrame()
            
            # 合并
            frames = []
            if not baseline_top8.empty:
                frames.append(baseline_top8)
            if not prev_top2.empty:
                frames.append(prev_top2)
            
            if not frames:
                raise ValueError(f"Round {round_num}: 无法加载任何记忆")
            
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values("val_score", ascending=False).head(10)
            
            positives = combined.to_dict(orient="records")
            
            baseline_count = len(baseline_top8) if not baseline_top8.empty else 0
            prev_count = len(prev_top2) if not prev_top2.empty else 0
            
            self.logger.info(
                f"[memory] Round {round_num} 使用混合策略: "
                f"{baseline_count} from baseline + {prev_count} from round 1 "
                f"(80/20 比例)"
            )
            self.logger.info(f"  - 最终选取 {len(positives)} 个正样本")
            
            return positives
        
        # ========== Round 3+: 70% baseline + 30% previous ========== #
        try:
            baseline_df = pd.read_csv(BASELINE_FILE)
            baseline_df["val_score"] = pd.to_numeric(baseline_df["val_score"], errors="coerce")
            baseline_top7 = baseline_df.sort_values("val_score", ascending=False).head(7)
        except Exception as e:
            self.logger.warning(f"读取 baseline 失败: {e}")
            baseline_top7 = pd.DataFrame()
        
        prev_path = os.path.join(RESULTS_DIR, f"round_{round_num-1}_factor_metrics.csv")
        
        if os.path.exists(prev_path):
            try:
                prev_df = pd.read_csv(prev_path)
                prev_df["val_score"] = pd.to_numeric(prev_df["val_score"], errors="coerce")
                prev_top3 = prev_df.sort_values("val_score", ascending=False).head(3)
            except Exception as e:
                self.logger.warning(f"读取上一轮失败: {e}")
                prev_top3 = pd.DataFrame()
        else:
            prev_top3 = pd.DataFrame()
        
        # 合并
        frames = []
        if not baseline_top7.empty:
            frames.append(baseline_top7)
        if not prev_top3.empty:
            frames.append(prev_top3)
        
        if not frames:
            raise ValueError(f"Round {round_num}: 无法加载任何记忆")
        
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("val_score", ascending=False).head(10)
        
        positives = combined.to_dict(orient="records")
        
        baseline_count = len(baseline_top7) if not baseline_top7.empty else 0
        prev_count = len(prev_top3) if not prev_top3.empty else 0
        
        self.logger.info(
            f"[memory] Round {round_num} 使用混合策略: "
            f"{baseline_count} from baseline + {prev_count} from round {round_num-1} "
            f"(70/30 比例)"
        )
        self.logger.info(f"  - 最终选取 {len(positives)} 个正样本")
        
        if len(positives) > 0:
            val_scores = [float(p.get("val_score", 0)) for p in positives]
            val_range = f"[{min(val_scores):.4f}, {max(val_scores):.4f}]"
            self.logger.info(f"  - Val score 范围: {val_range}")
            self.logger.info(f"  - Val score 均值: {np.mean(val_scores):.4f}")
        
        return positives

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
        # Round 1 使用 0 个负样本（专注学习 baseline）
        if round_num == 1:
            self.logger.info(f"[memory] Round 1: 跳过负样本生成（专注学习 baseline）")
            return []
        
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
        """合并正负样本记忆"""
        memory = []
        
        for r in positives:
            memory.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
                "val_score": float(r.get("val_score", 0.0) or 0.0),  # 添加 val_score
                "memory_type": "positive",
            })
        
        for r in negatives:
            memory.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
                "memory_type": "negative",
            })
        
        self.logger.info(
            f"[memory] 合并记忆: {len(positives)} 正样本 + {len(negatives)} 负样本 "
            f"= {len(memory)} 总记忆"
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
