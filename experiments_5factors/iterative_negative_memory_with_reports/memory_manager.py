# experiments/iterative_negative_memory_with_reports/memory_manager.py
"""
Memory Manager WITH REPORTS

FIXED VERSION:
1. Added periods_per_year configuration reading
2. Use self.periods_per_year in batch_evaluate() call
3. Simplified memory strategy (100% baseline -> 100% previous round)
"""

import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

# 配置导入
from .config import (
    RESULTS_DIR,
    BASELINE_FILE,
    MEMORY_FIELDS,  # ← ["code", "factor_report"]
)

# 全局配置
from shared.config_loader import load_global_config

# 核心模块
from core.data_loader import load_splits
from core.factor_evaluator import batch_evaluate

# 报告生成
from reports.report_builder import generate_factor_report, FactorMetrics

# 负向代理
from .negative_agents_reports import NegativeAgent

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器（with reports + FIXED）
    
    职责：
    1. 加载上一轮 Top10（包含 factor_report）
    2. 生成负样本（代码 + 评估 + 报告）
    3. 合并记忆：只保留 code + factor_report + memory_type
    4. 保存结果
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 加载全局配置
        g = load_global_config()
        
        # ========== FIXED: 保存 periods_per_year ========== #
        self.periods_per_year = int(g.get("freq_per_year", 4))
        # ================================================== #
        
        # 一次性加载数据切分
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
        
        # 初始化负向代理
        self.neg_agent = NegativeAgent(logger=self.logger)
    
    # ========== 加载记忆（Top10 + factor_report）========== #
    
    def get_memory_for_round(self, round_num: int) -> List[Dict]:
        """
        返回上一轮的 Top10 因子（FIXED: 简化策略）
        
        策略：
        - Round 1: 100% baseline (Top 10)
        - Round 2+: 100% previous round (Top 10)
        """
        # ========== Round 1: 100% baseline ========== #
        if round_num == 1:
            csv_path = BASELINE_FILE
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Baseline not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # 确保有 factor_report 列
            if "factor_report" not in df.columns:
                self.logger.warning(f"[memory] Baseline 缺少 factor_report 列，使用空字符串")
                df["factor_report"] = ""
            
            # 按 train_score 排序
            df["train_score"] = pd.to_numeric(df["train_score"], errors="coerce")
            top10 = df.sort_values("train_score", ascending=False, na_position="last").head(5)
            
            positives = top10.to_dict(orient="records")
            
            self.logger.info(
                f"[memory] Round {round_num} 使用 100% baseline: {len(positives)} 个正样本"
            )
            if len(top10) > 0:
                train_range = f"[{top10['train_score'].min():.4f}, {top10['train_score'].max():.4f}]"
                self.logger.info(f"  - Train score 范围: {train_range}")
            
            return positives
        
        # ========== Round 2+: 100% previous round ========== #
        prev_path = os.path.join(RESULTS_DIR, f"round_{round_num-1}_factor_metrics.csv")
        
        if not os.path.exists(prev_path):
            raise FileNotFoundError(f"Round {round_num}: 上一轮结果不存在: {prev_path}")
        
        try:
            prev_df = pd.read_csv(prev_path)
            
            # 确保有 factor_report 列
            if "factor_report" not in prev_df.columns:
                self.logger.warning(f"[memory] Round {round_num-1} 缺少 factor_report 列，使用空字符串")
                prev_df["factor_report"] = ""
            
            # 按 train_score 排序
            prev_df["train_score"] = pd.to_numeric(prev_df["train_score"], errors="coerce")
            top10 = prev_df.sort_values("train_score", ascending=False, na_position="last").head(5)
            
        except Exception as e:
            raise ValueError(f"Round {round_num}: 读取上一轮失败: {e}")
        
        positives = top10.to_dict(orient="records")
        
        self.logger.info(
            f"[memory] Round {round_num} 使用 100% round {round_num-1}: {len(positives)} 个正样本"
        )
        
        if len(positives) > 0:
            train_scores = [float(p.get("train_score", 0)) for p in positives]
            train_range = f"[{min(train_scores):.4f}, {max(train_scores):.4f}]"
            self.logger.info(f"  - Train score 范围: {train_range}")
            self.logger.info(f"  - Train score 均值: {np.mean(train_scores):.4f}")
        
        return positives
    
    # ========== 生成负样本（代码 + 评估 + 报告）========== #
    
    def get_worst_factors_with_reports(
        self,
        round_num: int,
        n: int = 3,  # ← FIXED: default 改为 3
        positives_for_context: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        生成最差因子并附带报告
        
        流程：
        1. 调用 NegativeAgent 生成坏因子代码 + 简短报告
        2. 调用 batch_evaluate 评估（获取 train_score）
        3. 返回 [{code, factor_report, train_score, memory_type="negative"}, ...]
        """
        positives_for_context = positives_for_context or []
        
        # Step 1: 准备上下文（仅 code + factor_report）
        pos_ctx = []
        for r in positives_for_context[:10]:
            pos_ctx.append({
                "code": r.get("code", ""),
                "factor_report": r.get("factor_report", ""),
            })
        
        self.logger.info(f"[memory] 生成 {n} 个最差因子（round {round_num}）...")
        
        # Step 2: 调用 NegativeAgent 生成（两步法：代码 + 报告）
        raw_negatives = self.neg_agent.generate_negative_factors_with_reports(
            current_round=round_num,
            target_n=n,
            context_positives=pos_ctx
        )
        
        if not raw_negatives:
            self.logger.warning("[memory] NegativeAgent 未生成任何负样本")
            return []
        
        # Step 3: 调用 FactorEvaluator 评估 train_score
        factors = [{"code": item.get("code", "")} for item in raw_negatives]
        
        try:
            # ========== FIXED: 使用 self.periods_per_year ========== #
            results_df = batch_evaluate(
                factors=factors,
                splits=self.splits,
                ret_col="ret",
                date_col="datadate",
                periods_per_year=self.periods_per_year,  # ← FIXED
                id_start=1
            )
            # ==================================================== #
        except Exception as e:
            self.logger.error(f"[memory] 负样本评估失败: {e}")
            return []
        
        # Step 4: 合并：code + factor_report + train_score + memory_type
        negatives = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            negatives.append({
                "code": row["code"],
                "factor_report": raw_negatives[i].get("factor_report", ""),  # ← 来自 NegativeAgent
                "train_score": float(row.get("train_score", 0.0) or 0.0),
                "memory_type": "negative",
            })
        
        self.logger.info(f"[memory] 最终获得 {len(negatives)} 个负样本（目标 {n}）")
        return negatives[:n]
    
    # ========== 合并记忆（仅保留 code + factor_report）========== #
    
    def combine_memory(self, positives: List[Dict], negatives: List[Dict]) -> List[Dict]:
        """
        合并正负样本记忆
        
        **关键改动**：只保留 code + factor_report + memory_type
        （不再包含 train_score，避免泄露）
        """
        memory = []
        
        # 正样本
        for r in positives:
            memory.append({
                "code": r.get("code", ""),
                "factor_report": r.get("factor_report", ""),  # ← 使用报告
                "memory_type": "positive",
            })
        
        # 负样本
        for r in negatives:
            memory.append({
                "code": r.get("code", ""),
                "factor_report": r.get("factor_report", ""),  # ← 使用报告
                "memory_type": "negative",
            })
        
        self.logger.info(
            f"[memory] 合并记忆: {len(positives)} 正样本 + {len(negatives)} 负样本 "
            f"= {len(memory)} 总记忆 (字段: {MEMORY_FIELDS})"
        )
        
        return memory
    
    # ========== 保存结果（与 negative_memory 相同）========== #
    
    def save_round_results(self, evaluated_factors: List[Dict], round_num: int) -> str:
        """保存本轮评估结果（包含 factor_report）"""
        path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
        
        # 转为 DataFrame
        df = pd.DataFrame(evaluated_factors)
        
        # 确保有 factor_report 列
        if "factor_report" not in df.columns:
            self.logger.warning(f"[memory] Round {round_num} 缺少 factor_report 列")
            df["factor_report"] = ""
        
        # 保存
        df.to_csv(path, index=False)
        
        # 统计
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
        """聚合所有轮次结果（与 negative_memory 相同）"""
        rows = []
        
        # Baseline (Round 0)
        if os.path.exists(BASELINE_FILE):
            try:
                df = pd.read_csv(BASELINE_FILE)
                df["round_num"] = 0
                
                # 确保有 factor_report（baseline 可能没有）
                if "factor_report" not in df.columns:
                    df["factor_report"] = ""
                
                rows.append(df)
            except Exception as e:
                self.logger.warning(f"[memory] 读取 baseline 失败: {e}")
        
        # Round 1+
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
        
        # 合并
        all_df = pd.concat(rows, ignore_index=True, sort=False)
        all_df = all_df.sort_values(["round_num", "val_score"], ascending=[True, False])
        
        # 保存
        out_path = os.path.join(RESULTS_DIR, "all_rounds_factors.csv")
        all_df.to_csv(out_path, index=False)
        
        self.logger.info(f"[memory] 聚合结果已更新: {out_path}")
        self.logger.info(f"  - 总轮数: {all_df['round_num'].nunique()}")
        self.logger.info(f"  - 总因子数: {len(all_df)}")
        
        return out_path
