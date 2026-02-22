# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/iterator.py
"""
Iterator for negative memory experiment

核心流程（每轮）：
1. 加载上一轮 Top10 正样本（code + train_score）
2. 生成 n 个负样本（NegativeAgent → FactorEvaluator）
3. 合并记忆（10 正 + n 负）
4. 生成新因子（PositiveAgents，基于对比学习）
5. 评估新因子（batch_evaluate）
6. 保存结果（memory_manager）

设计原则：
- 简洁：只负责流程编排，不重复 memory_manager 的功能
- 复用：所有评估、保存、记忆管理都交给 memory_manager
- 聚焦：处理负样本生成 + 正向生成的衔接
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

# ========== 本地模块导入 ========== #
from .config import (
    FACTORS_PER_ROUND,
    NEGATIVE_SAMPLES_COUNT,
    MAX_GENERATION_ATTEMPTS,
)
from .memory_manager import MemoryManager
from .positive_agents import PositiveAgents

# ========== 核心模块导入 ========== #
from core.factor_evaluator import batch_evaluate


class FactorIterator:
    """
    因子迭代器（负记忆版本）
    
    职责：
    1. 编排每轮流程（加载记忆 → 生成负样本 → 生成正样本 → 评估 → 保存）
    2. 处理生成失败的重试逻辑
    3. 记录日志和统计
    """
    
    def __init__(self):
        """
        初始化迭代器
        
        注意：
        - memory_manager 负责所有数据加载、评估、保存
        - iterator 只负责流程编排
        """
        self.logger = logging.getLogger(__name__)
        
        # 组件初始化
        self.memory_manager = MemoryManager()
        self.positive_agent = PositiveAgents()
        
        # 配置参数
        self.factors_per_round = FACTORS_PER_ROUND
        self.negative_count = NEGATIVE_SAMPLES_COUNT
        self.max_attempts = MAX_GENERATION_ATTEMPTS

        # ========== 新增：从全局配置读取 periods_per_year ========== #
        from shared.config_loader import load_global_config
        g = load_global_config()
        self.periods_per_year = int(g.get("freq_per_year", 4))
        # ========================================================== #
        
        # 统计信息
        self.total_factors_generated = 0
        self.total_factors_evaluated = 0
        
        self.logger.info("[iterator] Initialized")
        self.logger.info(f"  - Factors per round: {self.factors_per_round}")
        self.logger.info(f"  - Negative samples: {self.negative_count}")
        self.logger.info(f"  - Max generation attempts: {self.max_attempts}")
        self.logger.info(f"  - Periods per year: {self.periods_per_year}")  # ← 新增日志
    
    # ========== 主流程 ========== #
    
    def execute_round(self, round_num: int) -> Dict[str, Any]:
        """
        执行单轮迭代
        
        Args:
            round_num: 轮次编号（从 1 开始）
        
        Returns:
            本轮统计信息
        """
        round_start = time.time()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ROUND {round_num} START")
        self.logger.info(f"{'='*60}")
        
        try:
            # ========== Step 1: 加载上一轮记忆（Top10 正样本）========== #
            self.logger.info(f"[Round {round_num}] Step 1: 加载上一轮 Top10 正样本")
            positives = self.memory_manager.get_memory_for_round(round_num)
            
            if not positives:
                raise ValueError(f"Round {round_num}: 无法加载上一轮记忆")
            
            # ========== Step 2: 生成负样本 ========== #
            self.logger.info(f"[Round {round_num}] Step 2: 生成 {self.negative_count} 个负样本")
            negatives = self.memory_manager.get_worst_factors_for_negative_generation(
                round_num=round_num,
                n=self.negative_count,
                positives_for_context=positives
            )
            
            if not negatives:
                self.logger.warning(f"[Round {round_num}] 负样本生成失败，继续使用空负样本")
                negatives = []
            
            # ========== Step 3: 合并记忆（正 + 负）========== #
            self.logger.info(f"[Round {round_num}] Step 3: 合并记忆")
            combined_memory = self.memory_manager.combine_memory(positives, negatives)
            
            self.logger.info(
                f"  - 记忆组成: {len(positives)} 正样本 + {len(negatives)} 负样本 "
                f"= {len(combined_memory)} 总记忆"
            )
            
            # ========== Step 4: 生成新因子（带重试）========== #
            self.logger.info(f"[Round {round_num}] Step 4: 生成新因子（目标 {self.factors_per_round}）")
            
            new_factors = self._generate_with_retry(
                round_num=round_num,
                memory_records=combined_memory,
                target_n=self.factors_per_round
            )
            
            if not new_factors:
                raise ValueError(f"Round {round_num}: 因子生成失败")
            
            self.total_factors_generated += len(new_factors)
            
            # ========== Step 5: 评估新因子 ========== #
            self.logger.info(f"[Round {round_num}] Step 5: 评估 {len(new_factors)} 个新因子")
            
            evaluated_df = self._evaluate_factors(new_factors)
            
            if evaluated_df is None or evaluated_df.empty:
                raise ValueError(f"Round {round_num}: 因子评估失败")
            
            self.total_factors_evaluated += len(evaluated_df)
            
            # ========== Step 6: 保存结果 ========== #
            self.logger.info(f"[Round {round_num}] Step 6: 保存结果")
            
            # 转为 dict list（memory_manager 需要）
            evaluated_records = evaluated_df.to_dict(orient="records")
            
            csv_path = self.memory_manager.save_round_results(
                evaluated_factors=evaluated_records,
                round_num=round_num
            )
            
            # ========== Step 7: 更新聚合结果 ========== #
            self.logger.info(f"[Round {round_num}] Step 7: 更新聚合结果")
            self.memory_manager.update_aggregated_results(round_num)
            
            # ========== 返回本轮统计 ========== #
            duration = time.time() - round_start
            
            stats = self._compute_round_stats(
                round_num=round_num,
                evaluated_df=evaluated_df,
                duration=duration
            )
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ROUND {round_num} COMPLETED in {duration:.2f}s")
            self.logger.info(f"{'='*60}")
            self._log_round_stats(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"[Round {round_num}] 执行失败: {e}")
            return {
                "round_num": round_num,
                "status": "failed",
                "error": str(e),
                "duration": time.time() - round_start,
                "factors_generated": 0,
                "factors_evaluated": 0,
            }
    
    # ========== 辅助方法 ========== #
    def _generate_with_retry(
        self,
        round_num: int,
        memory_records: List[Dict[str, Any]],
        target_n: int
    ) -> List[Dict[str, Any]]:
        """带重试的因子生成"""
        collected = []
        seen_codes = set()
        
        for attempt in range(1, self.max_attempts + 1):
            need = target_n - len(collected)
            
            if need <= 0:
                break
            
            self.logger.info(
                f"  - Attempt {attempt}/{self.max_attempts}: "
                f"需要 {need} 个，已收集 {len(collected)}"
            )
            
            # 调用 PositiveAgents 生成
            try:
                new_batch = self.positive_agent.generate_factors(
                    current_round=round_num,
                    target_n=need,
                    id_prefix=f"r{round_num}_",
                    memory_records=memory_records
                )
            except Exception as e:
                self.logger.error(f"  - Attempt {attempt} 生成失败: {e}")
                continue
            
            if not new_batch:
                self.logger.warning(f"  - Attempt {attempt} 返回空列表")
                continue
            
            # 去重并收集
            accepted = 0
            for factor in new_batch:
                code = factor.get("code", "")
                if not code:
                    continue
                
                # 简单去重（基于 code 文本）
                code_key = code.replace(" ", "").replace("\n", "").lower()
                if code_key in seen_codes:
                    continue
                
                seen_codes.add(code_key)
                collected.append(factor)
                accepted += 1
            
            self.logger.info(f"  - Attempt {attempt}: 接受 {accepted}/{len(new_batch)} 个")
            
            # 如果已经够了，提前退出
            if len(collected) >= target_n:
                break
        
        # 最终裁剪到目标数量
        final = collected[:target_n]
        
        # ========== 新增：数量不足警告 ========== #
        if len(final) < target_n:
            shortage = target_n - len(final)
            self.logger.warning(
                f"  生成不足！目标 {target_n}，实际 {len(final)}，"
                f"缺少 {shortage} 个"
            )
            self.logger.warning(
                f"  建议：1) 增加 MAX_GENERATION_ATTEMPTS (当前 {self.max_attempts})"
            )
            self.logger.warning(
                f"        2) 降低相似度阈值或增加 temperature"
            )
        # ========================================= #
        
        self.logger.info(
            f"  - 生成完成: {len(final)}/{target_n} "
            f"（共 {len(collected)} 个去重后，{self.max_attempts} 次尝试）"
        )
        
        return final
        
    def _evaluate_factors(self, factors: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        if not factors:
            return None
        
        try:
            df = batch_evaluate(
                factors=factors,
                splits=self.memory_manager.splits,
                ret_col="ret",
                date_col="datadate",
                periods_per_year=self.periods_per_year
            )
            
            # 过拟合过滤
            MAX_RATIO = 12.0
            if df is not None and not df.empty:
                if 'val_score' in df.columns and 'train_score' in df.columns:
                    val = pd.to_numeric(df['val_score'], errors='coerce').abs()
                    train = pd.to_numeric(df['train_score'], errors='coerce').abs()
                    ratio = val / train.replace(0, float('nan'))
                    overfit_mask = ratio > MAX_RATIO
                    n_filtered = int(overfit_mask.sum())
                    if n_filtered > 0:
                        self.logger.warning(f"⚠️ 过滤掉 {n_filtered} 个严重过拟合因子 (|val/train| > {MAX_RATIO}x)")
                        df = df[~overfit_mask].reset_index(drop=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return None
    

    
    def _compute_round_stats(
        self,
        round_num: int,
        evaluated_df: pd.DataFrame,
        duration: float
    ) -> Dict[str, Any]:
        """
        计算本轮统计信息
        
        Args:
            round_num: 轮次号
            evaluated_df: 评估结果
            duration: 耗时（秒）
        
        Returns:
            统计字典
        """
        stats = {
            "round_num": round_num,
            "status": "success",
            "duration": duration,
            "factors_generated": len(evaluated_df),
            "factors_evaluated": len(evaluated_df),
        }
        
        # 计算指标统计
        for metric in ["val_score", "train_score", "val_sharpe", "val_ann_ret", "val_D"]:
            if metric in evaluated_df.columns:
                values = pd.to_numeric(evaluated_df[metric], errors="coerce").dropna()
                if len(values) > 0:
                    stats[f"{metric}_mean"] = float(values.mean())
                    stats[f"{metric}_max"] = float(values.max())
                    stats[f"{metric}_min"] = float(values.min())
        
        # 成功率
        if "status" in evaluated_df.columns:
            success_count = (evaluated_df["status"] == "success").sum()
            stats["success_rate"] = float(success_count / len(evaluated_df))
        
        return stats
    
    def _log_round_stats(self, stats: Dict[str, Any]) -> None:
        """
        打印本轮统计信息
        """
        self.logger.info("\n本轮统计:")
        self.logger.info(f"  状态: {stats.get('status', 'unknown')}")
        self.logger.info(f"  耗时: {stats.get('duration', 0):.2f}s")
        self.logger.info(f"  因子数: {stats.get('factors_evaluated', 0)}")
        
        if "success_rate" in stats:
            self.logger.info(f"  成功率: {stats['success_rate']:.1%}")
        
        # Val 指标
        if "val_score_mean" in stats:
            self.logger.info(
                f"  Val Score: mean={stats['val_score_mean']:.4f}, "
                f"max={stats['val_score_max']:.4f}, "
                f"min={stats['val_score_min']:.4f}"
            )
        
        if "val_sharpe_mean" in stats:
            self.logger.info(
                f"  Val Sharpe: mean={stats['val_sharpe_mean']:.4f}, "
                f"max={stats['val_sharpe_max']:.4f}"
            )
        
        if "val_D_mean" in stats:
            self.logger.info(
                f"  Val D: mean={stats['val_D_mean']:.4f}, "
                f"max={stats['val_D_max']:.4f}"
            )
        
        # Train 指标（参考）
        if "train_score_mean" in stats:
            self.logger.info(
                f"  Train Score (参考): mean={stats['train_score_mean']:.4f}"
            )
    
    # ========== 全局统计 ========== #
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        返回全局统计信息
        
        用于最终报告
        """
        return {
            "total_factors_generated": self.total_factors_generated,
            "total_factors_evaluated": self.total_factors_evaluated,
        }
