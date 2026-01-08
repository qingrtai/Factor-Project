# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/memory_manager.py
"""
FIXED VERSION - 混合 baseline 策略防止逐轮退化

核心改动：
1. Round 2+ 始终混合 50% baseline + 50% 上一轮
2. 防止质量逐轮下降的恶性循环
3. 保持高质量参考基准
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
    记忆管理器（改进版）
    
    关键改进：
    - Round 2+ 混合 baseline 和上一轮，防止质量退化
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        g = load_global_config()
        
        self.logger.info("加载数据切分...")
        self.splits = load_splits(
            raw_csv=g["data"]["raw_file"],
            date_col=g["schema"]["date_col"],
            years=g["years"],
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        self.logger.info("数据切分加载完成")
        
        self.neg_agent = NegativeAgent(logger=self.logger)

    def get_memory_for_round(self, round_num: int) -> List[Dict]:
        """
        返回本轮的正样本记忆
        
        ⚠️ 关键改进：Round 2+ 混合 baseline + 上一轮
        
        策略：
        - Round 1: 100% baseline (Top 10)
        - Round 2+: 50% baseline (Top 5) + 50% 上一轮 (Top 5)
        
        这样可以：
        1. 始终保持高质量参考（来自 baseline）
        2. 避免质量逐轮下降的恶性循环
        3. 同时学习上一轮的改进
        """
        # ========== Round 1: 使用 baseline ========== #
        if round_num <= 1:
            csv_path = BASELINE_FILE
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Baseline not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")
            top10 = df.sort_values("val_score", ascending=False, na_position="last").head(10)
            
            positives = top10.to_dict(orient="records")
            
            self.logger.info(
                f"[memory] Round {round_num} 加载了 {len(positives)} 个正样本 (来自 baseline)"
            )
            if len(top10) > 0:
                val_range = f"[{top10['val_score'].min():.4f}, {top10['val_score'].max():.4f}]"
                self.logger.info(f"  - Val score 范围: {val_range}")
            
            return positives
        
        # ========== Round 2+: 混合策略 ========== #
        # 读取 baseline Top 5
        try:
            baseline_df = pd.read_csv(BASELINE_FILE)
            baseline_df["val_score"] = pd.to_numeric(baseline_df["val_score"], errors="coerce")
            baseline_top5 = baseline_df.sort_values("val_score", ascending=False).head(5)
        except Exception as e:
            self.logger.warning(f"读取 baseline 失败: {e}，退化为只用上一轮")
            baseline_top5 = pd.DataFrame()
        
        # 读取上一轮 Top 5
        prev_path = os.path.join(RESULTS_DIR, f"round_{round_num-1}_factor_metrics.csv")
        
        if not os.path.exists(prev_path):
            self.logger.warning(f"上一轮文件不存在: {prev_path}")
            # 如果上一轮不存在，只用 baseline
            if not baseline_top5.empty:
                return baseline_top5.head(10).to_dict(orient="records")
            else:
                raise FileNotFoundError(f"无法加载任何记忆")
        
        try:
            prev_df = pd.read_csv(prev_path)
            prev_df["val_score"] = pd.to_numeric(prev_df["val_score"], errors="coerce")
            prev_top5 = prev_df.sort_values("val_score", ascending=False).head(5)
        except Exception as e:
            self.logger.warning(f"读取上一轮失败: {e}")
            prev_top5 = pd.DataFrame()
        
        # 合并
        frames = []
        if not baseline_top5.empty:
            frames.append(baseline_top5)
        if not prev_top5.empty:
            frames.append(prev_top5)
        
        if not frames:
            raise ValueError(f"Round {round_num}: 无法加载任何记忆")
        
        combined = pd.concat(frames, ignore_index=True)
        
        # 按 val_score 排序，取 Top 10
        combined = combined.sort_values("val_score", ascending=False).head(10)

        # 确保包含 val_score 用于 prompt
        if 'val_score' not in combined.columns and os.path.exists(BASELINE_FILE):
            # 从baseline读取val_score
            baseline_df = pd.read_csv(BASELINE_FILE)
            if 'val_score' in baseline_df.columns:
                combined = combined.merge(
                    baseline_df[['code', 'val_score']], 
                    on='code', 
                    how='left'
                )
        
        positives = combined.to_dict(orient="records")
        
        positives = combined.to_dict(orient="records")
        
        # ========== 日志输出改进的策略信息 ========== #
        baseline_count = len(baseline_top5) if not baseline_top5.empty else 0
        prev_count = len(prev_top5) if not prev_top5.empty else 0
        
        self.logger.info(
            f"[memory] Round {round_num} 使用混合策略: "
            f"{baseline_count} from baseline + {prev_count} from round {round_num-1}"
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
        """生成负样本"""
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
            results_df = batch_evaluate(
                factors=factors,
                splits=self.splits,
                ret_col="ret",
                date_col="datadate",
                periods_per_year=4,
                id_start=1
            )
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
