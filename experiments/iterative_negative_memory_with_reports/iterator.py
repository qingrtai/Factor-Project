# experiments/iterative_negative_memory_with_reports/iterator.py
"""
Iterator for negative memory WITH REPORTS experiment

核心改动 vs iterative_negative_memory:
1. 评估后生成 factor_report（调用 reports/report_builder.py）
2. 记忆传递 ["code", "factor_report"] 而非 ["code", "train_score"]
3. 负样本生成包含报告（两步法）
"""

import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

# 本地模块
from .config import (
    FACTORS_PER_ROUND,
    NEGATIVE_SAMPLES_COUNT,
    MAX_GENERATION_ATTEMPTS,
)
from .memory_manager import MemoryManager
from .positive_agents_reports import PositiveAgents
from .aggregator_reports import ResultsAggregator  # ← 新增

# 核心模块
from core.factor_evaluator import batch_evaluate

# 报告生成模块
from reports.report_builder import generate_factor_report, FactorMetrics




class FactorIterator:
    """
    因子迭代器（负记忆 + 报告版本）
    
    关键流程：
    1. 生成因子 → 评估 → **生成报告** → 保存
    2. 记忆管理：传递 code + factor_report（而非 train_score）
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 组件初始化
        self.memory_manager = MemoryManager()
        self.positive_agent = PositiveAgents()
        self.aggregator = ResultsAggregator()  # ← 新增
        
        # 配置参数
        self.factors_per_round = FACTORS_PER_ROUND
        self.negative_count = NEGATIVE_SAMPLES_COUNT
        self.max_attempts = MAX_GENERATION_ATTEMPTS

        # __init__ 中添加（第 52-54 行）
        from shared.config_loader import load_global_config
        g = load_global_config()
        self.periods_per_year = int(g.get("freq_per_year", 4))
        self.logger.info(f"  - Periods per year: {self.periods_per_year}")
        
        # 统计信息
        self.total_factors_generated = 0
        self.total_factors_evaluated = 0
        
        self.logger.info("[iterator] Initialized (WITH REPORTS)")
        self.logger.info(f"  - Factors per round: {self.factors_per_round}")
        self.logger.info(f"  - Negative samples: {self.negative_count}")
    
    # ========== 主流程 ========== #
    
    def execute_round(self, round_num: int) -> Dict[str, Any]:
        """执行单轮迭代"""
        round_start = time.time()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ROUND {round_num} START")
        self.logger.info(f"{'='*60}")
        
        try:
            # Step 1: 加载上一轮记忆（Top10，包含 factor_report）
            self.logger.info(f"[Round {round_num}] Step 1: 加载记忆")
            positives = self.memory_manager.get_memory_for_round(round_num)
            
            if not positives:
                raise ValueError(f"Round {round_num}: 无法加载上一轮记忆")
            
            # Step 2: 生成负样本（代码 + 报告）
            self.logger.info(f"[Round {round_num}] Step 2: 生成 {self.negative_count} 个负样本")
            negatives = self.memory_manager.get_worst_factors_with_reports(
                round_num=round_num,
                n=self.negative_count,
                positives_for_context=positives
            )
            
            if not negatives:
                self.logger.warning(f"[Round {round_num}] 负样本生成失败，继续使用空负样本")
                negatives = []
            
            # Step 3: 合并记忆（正 + 负，只保留 code + factor_report）
            self.logger.info(f"[Round {round_num}] Step 3: 合并记忆")
            combined_memory = self.memory_manager.combine_memory(positives, negatives)
            
            self.logger.info(
                f"  - 记忆组成: {len(positives)} 正样本 + {len(negatives)} 负样本 "
                f"= {len(combined_memory)} 总记忆"
            )
            
            # Step 4: 生成新因子（基于 factor_report 学习）
            self.logger.info(f"[Round {round_num}] Step 4: 生成新因子（目标 {self.factors_per_round}）")
            
            new_factors = self._generate_with_retry(
                round_num=round_num,
                memory_records=combined_memory,
                target_n=self.factors_per_round
            )
            
            if not new_factors:
                raise ValueError(f"Round {round_num}: 因子生成失败")
            
            self.total_factors_generated += len(new_factors)

            # Step 5: 评估新因子
            self.logger.info(f"[Round {round_num}] Step 5: 评估 {len(new_factors)} 个新因子")
            
            evaluated_df = self._evaluate_factors(new_factors)
            
            if evaluated_df is None or evaluated_df.empty:
                raise ValueError(f"Round {round_num}: 因子评估失败")
            
            # ===== 新增：过滤 NaN + 补货 =====
            evaluated_df = evaluated_df.dropna(subset=['val_score']).reset_index(drop=True)
            self.logger.info(f"[Round {round_num}] 过滤后有效因子: {len(evaluated_df)}/{self.factors_per_round}")
            
            nan_refill_attempt = 0
            while len(evaluated_df) < self.factors_per_round and nan_refill_attempt < 3:
                nan_refill_attempt += 1
                deficit = self.factors_per_round - len(evaluated_df)
                self.logger.info(f"[Round {round_num}] NaN补货 attempt {nan_refill_attempt}: 需要 {deficit} 个")
                
                extra_codes = self.positive_agent.generate_factors(
                    current_round=round_num,
                    target_n=deficit,
                    id_prefix=f"r{round_num}_refill{nan_refill_attempt}_",
                    memory_records=combined_memory
                )
                if not extra_codes:
                    break
                
                extra_df = self._evaluate_factors(extra_codes)
                if extra_df is not None and not extra_df.empty:
                    extra_df = extra_df.dropna(subset=['val_score'])
                        # ← 在这里加去重，然后才 concat
                    existing_codes = set(evaluated_df['code'].astype(str).tolist())
                    extra_df = extra_df[~extra_df['code'].astype(str).isin(existing_codes)]
                    evaluated_df = pd.concat([evaluated_df, extra_df], ignore_index=True)
    
            
            self.logger.info(f"[Round {round_num}] 最终有效因子数: {len(evaluated_df)}")
            # ===== 新增结束 =====
            
            # ========== Step 6: 生成因子报告（新增）========== #
            self.logger.info(f"[Round {round_num}] Step 6: 生成因子报告")
            
            evaluated_df = self._generate_reports(evaluated_df, round_num)
            
            # Step 7: 保存结果
            self.logger.info(f"[Round {round_num}] Step 7: 保存结果")
            
            evaluated_records = evaluated_df.to_dict(orient="records")
            
            csv_path = self.memory_manager.save_round_results(
                evaluated_factors=evaluated_records,
                round_num=round_num
            )
            
            # Step 8: 更新聚合结果
            self.logger.info(f"[Round {round_num}] Step 8: 更新聚合结果")
            self.memory_manager.update_aggregated_results(round_num)
            
            # Step 9: 生成 round_summary_mean.csv（新增）
            self.logger.info(f"[Round {round_num}] Step 9: 生成轮次汇总")
            try:
                self.aggregator.generate_round_summary()
            except Exception as e:
                self.logger.warning(f"[Round {round_num}] 汇总生成失败: {e}")
            
            # 返回本轮统计
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
        """带重试的因子生成（与 negative_memory 相同）"""
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
                
                code_key = code.replace(" ", "").replace("\n", "").lower()
                if code_key in seen_codes:
                    continue
                
                seen_codes.add(code_key)
                collected.append(factor)
                accepted += 1
            
            self.logger.info(f"  - Attempt {attempt}: 接受 {accepted}/{len(new_batch)} 个")
            
            if len(collected) >= target_n:
                break
        
        final = collected[:target_n]
        
        if len(final) < target_n:
            shortage = target_n - len(final)
            self.logger.warning(
                f"  生成不足！目标 {target_n}，实际 {len(final)}，缺少 {shortage} 个"
            )
        
        self.logger.info(f"  - 生成完成: {len(final)}/{target_n}")
        
        return final
    
    def _evaluate_factors(self, factors: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """评估因子列表（与 negative_memory 相同）"""
        if not factors:
            return None
        
        try:
            df = batch_evaluate(
                factors=factors,
                splits=self.memory_manager.splits,
                ret_col="ret",
                date_col="datadate",
                periods_per_year=self.periods_per_year  # ← FIXED
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return None
    
    def _generate_reports(self, df: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """
        为评估后的因子生成 factor_report（新增方法）
        
        Args:
            df: 评估结果 DataFrame
            round_num: 轮次号
        
        Returns:
            添加了 factor_report 列的 DataFrame
        """
        self.logger.info(f"  - 为 {len(df)} 个因子生成报告...")
        
        reports = []
        
        for idx, row in df.iterrows():
            try:
                # 构建 FactorMetrics
                metrics = FactorMetrics(
                    sharpe=row.get("val_sharpe"),
                    ann_ret=row.get("val_ann_ret"),
                    max_dd=row.get("val_max_dd"),
                    coverage=row.get("val_coverage"),
                    train_score=row.get("train_score"),
                    val_score=row.get("val_score"),
                    rank=idx + 1,                    # 排名（按 val_score 降序）
                    total_factors=len(df),
                )
                
                # 生成报告
                report = generate_factor_report(
                    code=row["code"],
                    metrics=metrics,
                    round_id=round_num,
                    factor_id=row.get("factor_id", f"r{round_num}_{idx+1:02d}"),
                    is_detailed=True  # 正向因子用详细报告
                )
                
                reports.append(report)
                
            except Exception as e:
                self.logger.warning(f"  - 因子 {idx+1} 报告生成失败: {e}")
                reports.append("(fallback) Report generation failed.")
        
        # 添加到 DataFrame
        df["factor_report"] = reports
        
        self.logger.info(f"  - 报告生成完成: {len(reports)}/{len(df)}")
        
        return df
    
    def _compute_round_stats(
        self,
        round_num: int,
        evaluated_df: pd.DataFrame,
        duration: float
    ) -> Dict[str, Any]:
        """计算本轮统计信息（与 negative_memory 相同）"""
        stats = {
            "round_num": round_num,
            "status": "success",
            "duration": duration,
            "factors_generated": len(evaluated_df),
            "factors_evaluated": len(evaluated_df),
        }
        
        for metric in ["val_score", "train_score", "val_sharpe", "val_ann_ret", "val_D"]:
            if metric in evaluated_df.columns:
                values = pd.to_numeric(evaluated_df[metric], errors="coerce").dropna()
                if len(values) > 0:
                    stats[f"{metric}_mean"] = float(values.mean())
                    stats[f"{metric}_max"] = float(values.max())
                    stats[f"{metric}_min"] = float(values.min())
        
        if "status" in evaluated_df.columns:
            success_count = (evaluated_df["status"] == "success").sum()
            stats["success_rate"] = float(success_count / len(evaluated_df))
        
        return stats
    
    def _log_round_stats(self, stats: Dict[str, Any]) -> None:
        """打印本轮统计信息（与 negative_memory 相同）"""
        self.logger.info("\n本轮统计:")
        self.logger.info(f"  状态: {stats.get('status', 'unknown')}")
        self.logger.info(f"  耗时: {stats.get('duration', 0):.2f}s")
        self.logger.info(f"  因子数: {stats.get('factors_evaluated', 0)}")
        
        if "success_rate" in stats:
            self.logger.info(f"  成功率: {stats['success_rate']:.1%}")
        
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
    
    def get_global_stats(self) -> Dict[str, Any]:
        """返回全局统计信息（与 negative_memory 相同）"""
        return {
            "total_factors_generated": self.total_factors_generated,
            "total_factors_evaluated": self.total_factors_evaluated,
        }
