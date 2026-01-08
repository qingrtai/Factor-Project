# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/memory_manager.py
"""
memory_manager.py — Memory orchestration for negative memory experiment

核心职责（精简版）：
1. 加载记忆：读取上一轮 Top10 正样本（按 val_score 排序）
2. 生成负样本：调用 NegativeAgent → 调用 FactorEvaluator 评估 train_score
3. 合并记忆：正样本 + 负样本（只保留 code + train_score + memory_type）
4. 保存结果：写入 CSV

设计原则：
- 复用 core/factor_evaluator.py 的 batch_evaluate()，避免重复代码
- 复用 core/data_loader.py 的 load_splits()，一次加载数据
- 不包含代码清洗、指标计算等逻辑（这些都在 core 模块里）
- 聚焦于记忆管理和流程编排
"""

import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

# ========== 配置导入（同目录）========== #
#  正确：只导入实验特定配置
from .config import (
    RESULTS_DIR,
    BASELINE_FILE,
)

#  全局配置从 shared/config_loader 读取
from shared.config_loader import load_global_config

# ========== 核心模块导入 ========== #
from core.data_loader import load_splits
from core.factor_evaluator import batch_evaluate

# ========== 负向代理导入（同目录）========== #
from .negative_agents import NegativeAgent

logger = logging.getLogger(__name__)


# ===================== 核心类 =====================

class MemoryManager:
    """
    记忆管理器（精简版）
    
    职责：
    1. Round 0：读 baseline code → 调用 FactorEvaluator 重新评估
    2. 每轮：加载上一轮 Top10 正样本 + 生成当轮负样本
    3. 合并记忆：10 正 + n 负（只保留 code + train_score + memory_type）
    4. 保存结果：写 CSV，聚合所有轮次
    """


    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 加载全局配置
        g = load_global_config()
    
        # 一次性加载数据切分
        self.logger.info("加载数据切分...")
        self.splits = load_splits(
            raw_csv=g["data"]["raw_file"],      # ← 从全局配置读取
            date_col=g["schema"]["date_col"],   # ← 从全局配置读取
            years=g["years"],                   # ← 从全局配置读取
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        self.logger.info("数据切分加载完成")
        
        # 初始化负向代理
        self.neg_agent = NegativeAgent(logger=self.logger)


    # ========== 加载记忆（上一轮 Top10）========== #
    def get_memory_for_round(self, round_num: int) -> List[Dict]:
        """
        返回上一轮的 Top10 因子（按 val_score 降序）
        
        - Round 1: 使用 round_0（baseline 重评估结果）
        - Round 2+: 使用 round_{n-1}
        
        返回格式：[{完整记录}, ...] （包含所有列）
        注意：进入 prompt 前会过滤为 code + train_score + memory_type
        """
        # 确定源文件
        if round_num <= 1:
            csv_path = BASELINE_FILE 
        else:
            csv_path = os.path.join(RESULTS_DIR, f"round_{round_num-1}_factor_metrics.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Memory source not found: {csv_path}")
        
        # 读取 CSV
        df = pd.read_csv(csv_path)
        
        # 确保 val_score 是数值
        df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")
        
        # 按 val_score 降序排序，取 Top10
        top10 = df.sort_values("val_score", ascending=False, na_position="last").head(10)
        
        positives = top10.to_dict(orient="records")
        
        self.logger.info(
            f"[memory] Round {round_num} 加载了 {len(positives)} 个正样本 "
            f"(来自 {'baseline' if round_num <= 1 else f'round {round_num-1}'})"
        )
        if len(top10) > 0:
            val_range = f"[{top10['val_score'].min():.4f}, {top10['val_score'].max():.4f}]"
            self.logger.info(f"  - Val score 范围: {val_range}")
        
        return positives

    # ========== 生成负样本（NegativeAgent + FactorEvaluator）========== #
    def get_worst_factors_for_negative_generation(
        self,
        round_num: int,
        n: int = 5,
        positives_for_context: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        通过 NegativeAgent 生成最差因子，然后用 FactorEvaluator 评估 train_score
        
        流程：
        1. 调用 NegativeAgent.generate_negative_factors() 生成坏因子代码
        2. 调用 batch_evaluate() 评估（获取 train_score）
        3. 返回 [{code, train_score, memory_type="negative"}, ...]
        
        注意：只评估 train_score（不向 prompt 泄露 val_score）
        """
        positives_for_context = positives_for_context or []
        
        # ========== Step 1: 准备上下文（仅 code + train_score）========== #
        pos_ctx = []
        for r in positives_for_context[:10]:
            pos_ctx.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
            })
        
        self.logger.info(f"[memory] 生成 {n} 个最差因子（round {round_num}）...")
        
        # ========== Step 2: 调用 NegativeAgent 生成 ========== #
        raw_negatives = self.neg_agent.generate_negative_factors(
            current_round=round_num,
            target_n=n,
            context_positives=pos_ctx
        )
        
        if not raw_negatives:
            self.logger.warning("[memory] NegativeAgent 未生成任何负样本")
            return []
        
        # ========== Step 3: 调用 FactorEvaluator 评估 train_score ========== #
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
        
        # ========== Step 4: 提取 code + train_score + memory_type ========== #
        negatives = []
        for _, row in results_df.iterrows():
            negatives.append({
                "code": row["code"],
                "train_score": float(row.get("train_score", 0.0) or 0.0),
                "memory_type": "negative",
            })
        
        self.logger.info(f"[memory] 最终获得 {len(negatives)} 个负样本（目标 {n}）")
        return negatives[:n]

    # ========== 合并记忆（仅供 prompt 使用）========== #
    def combine_memory(self, positives: List[Dict], negatives: List[Dict]) -> List[Dict]:
        """
        合并正负样本记忆（10 正 + n 负）
        
        为避免泄露验证分，**仅保留**以下字段：
        - code: 因子代码
        - train_score: 训练集分数
        - memory_type: "positive" 或 "negative"
        
        这个列表会传给 PositiveAgent 构建 prompt
        """
        memory = []
        
        # 正样本（来自上一轮结果）
        for r in positives:
            memory.append({
                "code": r.get("code", ""),
                "train_score": float(r.get("train_score", 0.0) or 0.0),
                "memory_type": "positive",
            })
        
        # 负样本（当轮生成）
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

    # ========== 保存结果 ========== #
    def save_round_results(self, evaluated_factors: List[Dict], round_num: int) -> str:
        """
        保存本轮评估结果
        
        输入：batch_evaluate() 返回的 DataFrame 转为 dict list
        输出：round_{n}_factor_metrics.csv
        """
        path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
        
        # 转为 DataFrame
        df = pd.DataFrame(evaluated_factors)
        
        # 保存
        df.to_csv(path, index=False)
        
        # 简要统计
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
        """
        聚合所有轮次结果到 results/all_rounds_factors.csv
        
        - Round 0: 直接读 BASELINE_FILE  # ← 改这里
        - Round 1+: 读 round_*_factor_metrics.csv
        """
        rows = []
        
        # ========== Baseline (Round 0) ========== #
        if os.path.exists(BASELINE_FILE):  # ← 改这里
            try:
                df = pd.read_csv(BASELINE_FILE)
                df["round_num"] = 0
                rows.append(df)
            except Exception as e:
                self.logger.warning(f"[memory] 读取 baseline 失败: {e}")
        
        # ========== Round 1+ ========== #
        for fn in sorted(os.listdir(RESULTS_DIR)):
            if fn.startswith("round_") and fn.endswith("_factor_metrics.csv"):
                fp = os.path.join(RESULTS_DIR, fn)
                try:
                    df = pd.read_csv(fp)
                    
                    # 提取轮次号
                    round_str = fn.replace("round_", "").replace("_factor_metrics.csv", "")
                    df["round_num"] = int(round_str)
                    
                    rows.append(df)
                except Exception as e:
                    self.logger.warning(f"[memory] 读取 {fn} 失败: {e}")
                    continue
        
        if not rows:
            self.logger.warning("[memory] 没有轮次结果可聚合")
            return ""
        
        # 合并所有轮次
        all_df = pd.concat(rows, ignore_index=True, sort=False)
        
        # 按轮次排序
        all_df = all_df.sort_values(["round_num", "val_score"], ascending=[True, False])
        
        # 保存
        out_path = os.path.join(RESULTS_DIR, "all_rounds_factors.csv")
        all_df.to_csv(out_path, index=False)
        
        self.logger.info(f"[memory] 聚合结果已更新: {out_path}")
        self.logger.info(f"  - 总轮数: {all_df['round_num'].nunique()}")
        self.logger.info(f"  - 总因子数: {len(all_df)}")
        
        return out_path

    # ========== 辅助统计（可选）========== #
    def get_round_statistics(self, round_num: int) -> Optional[Dict[str, Any]]:
        """
        读取指定轮次 CSV 并给出简要统计
        
        用于日志、面板或调试
        """
        # 确定文件路径
        if round_num == 0:
            csv_path = os.path.join(RESULTS_DIR, "round_0_factor_metrics.csv")
        else:
            csv_path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
        
        if not os.path.exists(csv_path):
            return None
        
        # 读取
        df = pd.read_csv(csv_path)
        
        # 转数值
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