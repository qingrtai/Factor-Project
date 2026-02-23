# experiments/iterative_baseline_with_reports/iterator.py
"""
迭代优化器（带报告版本）
核心改动：
1. 使用 memory_manager_reports（返回 codes + reports）
2. 使用 positive_agents_reports（输入 code + report）
3. 评估后生成因子报告（调用 report_builder）
4. 保存时包含 factor_report 列
"""

import os
import re
import json
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# 内部模块
from .memory_manager_reports import MemoryManager
from .positive_agents_reports import PositiveAgent

# 核心评估器
from core.factor_evaluator import batch_evaluate
from core.data_loader import load_splits
from shared.config_loader import load_global_config

# 报告生成器（新增）
from reports.report_builder import generate_factor_report, FactorMetrics

# 兼容配置导入
try:
    from .config import CONFIG as _CONFIG
except Exception:
    _CONFIG = None
try:
    from . import config as _cfg_module
except Exception:
    _cfg_module = None


def _cfg_get(key: str, default=None):
    """从 CONFIG 字典或 config 模块获取配置"""
    if isinstance(_CONFIG, dict) and key in _CONFIG:
        return _CONFIG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default


class IterativeOptimizer:
    """
    迭代优化器（带报告版本）
    
    核心流程（每轮）：
    1. 读取上一轮记忆（codes + reports）
    2. 生成新因子（基于报告学习）
    3. 评估新因子（得到指标）
    4. 生成报告（新增步骤）← 核心改动
    5. 保存结果（含 factor_report 列）
    """

    def __init__(self,
                 baseline_file: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 logs_dir: Optional[str] = None):
        # 路径配置
        self.project_root = Path(_cfg_get('PROJECT_ROOT', Path(__file__).resolve().parent))
        self.results_dir = Path(results_dir or _cfg_get('RESULTS_DIR', self.project_root / 'results'))
        self.logs_dir = Path(logs_dir or _cfg_get('LOGS_DIR', self.project_root / 'logs'))
        self.baseline_csv = Path(
            baseline_file or
            _cfg_get('BASELINE_CSV', _cfg_get('BASELINE_FILE', self.project_root.parent / 'factor_baseline' / 'results' / 'baseline_factor_metrics.csv'))
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 迭代配置
        self.max_rounds = int(_cfg_get('MAX_ROUNDS', _cfg_get('N_ROUNDS_MAX', 5)))
        self.patience = int(_cfg_get('EARLY_STOPPING_PATIENCE', _cfg_get('PATIENCE', 2)))
        self.min_delta = float(_cfg_get('MIN_DELTA', 1e-3))
        self.min_rounds = int(_cfg_get('MIN_ROUNDS', 1))
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', _cfg_get('N_FACTORS', 10)))
        self.val_score_threshold = _cfg_get('VAL_SCORE_THRESHOLD', None)
        self.require_full_round = bool(_cfg_get('REQUIRE_FULL_ROUND', False))
        self.max_refills = int(_cfg_get('MAX_REFILL_CALLS', _cfg_get('MAX_RETRIES', 3)))

        # 报告配置（新增）
        self.report_template_file = Path(_cfg_get('REPORT_TEMPLATE_FILE', self.project_root / 'report_template.txt'))
        self.report_max_tokens = int(_cfg_get('REPORT_MAX_TOKENS', 200))
        self.report_temperature = float(_cfg_get('REPORT_TEMPERATURE', 0.7))

        # 评分与切分说明
        self.score_desc = _cfg_get('SCORE_DESC', "Score = (Sharpe + AnnRet + D) / 3")
        self.derived_desc = _cfg_get('DERIVED_METRIC_DESC', "D = 1 - MaxDD")
        self.train_years = _cfg_get('TRAIN_YEARS', (1961, 2000))
        self.val_years = _cfg_get('VAL_YEARS', (2001, 2010))
        self.test_years = _cfg_get('TEST_YEARS', (2011, 2025))

        # 组件（使用带报告版本）
        self.memory_manager = MemoryManager(
            results_dir=self.results_dir,
            baseline_csv=self.baseline_csv
        )
        self.positive_agent = PositiveAgent()

        # 加载数据 splits
        g = load_global_config()
        self.splits = load_splits(
            g["data"]["raw_file"],
            g["schema"]["date_col"],
            g["years"],
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        self.ret_col = g["schema"]["ret_col"]
        self.date_col = g["schema"]["date_col"]
        self.periods_per_year = int(g.get("freq_per_year", 4))

        # 状态跟踪
        self.logger = self._setup_logger()
        self.history: List[Dict] = []
        self.best_avg_val = -float('inf')
        self.best_round = 0
        self.patience_counter = 0
        self.best_top_val = -float('inf')
        self.start_time = None
        self.total_factors_generated = 0

    # ============== 基础设施 ==============
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('IterativeOptimizer')
        logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        log_file = self.logs_dir / 'optimization_log.txt'
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False
        return logger

    # ============== 主流程 ==============
    def run_optimization(self) -> Dict:
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("开始迭代因子优化（带报告版本）")
        self.logger.info(f"最大轮次: {self.max_rounds} | 耐心: {self.patience} | MIN_DELTA: {self.min_delta} | MIN_ROUNDS: {self.min_rounds}")
        self.logger.info(f"每轮因子数: {self.factors_per_round}")
        self.logger.info(f"baseline: {self.baseline_csv}")
        self.logger.info(f"报告模板: {self.report_template_file}")
        self.logger.info(f"评分体系: {self.score_desc}；{self.derived_desc}")
        self.logger.info(f"时间切分: Train={self.train_years}, Val={self.val_years}, Test={self.test_years}")
        self.logger.info("=" * 60)

        try:
            baseline_summary = self._compute_baseline_summary()
            self.logger.info(f"Baseline统计: {baseline_summary}")

            for round_num in range(1, self.max_rounds + 1):
                round_result = self._run_single_round(round_num)
                if round_result is None:
                    self.logger.error(f"第{round_num}轮失败，停止优化")
                    break

                self.history.append(round_result)

                if self._check_early_stopping(round_result):
                    self.logger.info(f"早停触发，在第{round_num}轮结束优化")
                    break

                if self._check_threshold_stopping(round_result):
                    self.logger.info(f"达到平均 val_score 阈值，在第{round_num}轮结束优化")
                    break

            report = self._generate_final_report(baseline_summary)
            self._save_optimization_summary(report)

            # baseline + 各轮汇总
            try:
                self._write_all_rounds_aggregate()
                self.logger.info("已生成 all_rounds_factors.csv（包含 baseline 与各轮）")
            except Exception as e:
                self.logger.warning(f"汇总 all_rounds_factors.csv 失败：{e}")

            return report

        except Exception as e:
            self.logger.error(f"优化过程异常: {e}")
            raise

    def _run_single_round(self, round_num: int) -> Optional[Dict]:
        t0 = time.time()
        self.logger.info(f"\n{'='*20} 第 {round_num} 轮开始 {'='*20}")
        try:
             # Step 1: 读取上一轮记忆（codes + reports + train_score + val_score）
            self.logger.info("Step 1: 读取上一轮记忆...")
            codes, reports = self.memory_manager.load_memory(previous_round=round_num - 1)
            
            if not codes:
                self.logger.error("上一轮记忆为空")
                return None
            
            self.logger.info(f"读取了 {len(codes)} 个因子的记忆")
            has_reports = any(r.strip() for r in reports)
            self.logger.info(f"报告可用性: {'有报告' if has_reports else '无报告（可能是 baseline）'}")
            
            # 确定上一轮 CSV 路径
            if round_num == 1:
                prev_csv = self.memory_manager.baseline_csv
            else:
                prev_csv = self.memory_manager._round_csv_path(round_num - 1)
            
            # 读取 train_score 和 val_score（合并为一次读取）
            train_scores = [-999] * len(codes)
            val_scores = [-999] * len(codes)
            
            if prev_csv.exists():
                try:
                    df_prev = pd.read_csv(prev_csv)
                    if 'train_score' in df_prev.columns:
                        train_scores = df_prev['train_score'].fillna(-999).tolist()
                    else:
                        self.logger.warning("上一轮 CSV 缺少 train_score 列，使用默认值 -999")
                    if 'val_score' in df_prev.columns:
                        val_scores = df_prev['val_score'].fillna(-999).tolist()
                    else:
                        self.logger.warning("上一轮 CSV 缺少 val_score 列，使用默认值 -999")
                except Exception as e:
                    self.logger.warning(f"读取上一轮 CSV 失败: {e}，使用默认分数")
            else:
                self.logger.warning("上一轮 CSV 不存在，使用默认 train_score / val_score")
            
            # 确保长度一致
            if len(train_scores) != len(codes):
                self.logger.warning(f"train_scores 长度 {len(train_scores)} 与 codes 长度 {len(codes)} 不一致，补齐")
                train_scores = train_scores[:len(codes)] + [-999] * max(0, len(codes) - len(train_scores))
            if len(val_scores) != len(codes):
                self.logger.warning(f"val_scores 长度 {len(val_scores)} 与 codes 长度 {len(codes)} 不一致，补齐")
                val_scores = val_scores[:len(codes)] + [-999] * max(0, len(codes) - len(val_scores))
            
            self.logger.info(f"读取了 {len(train_scores)} 个 train_score，{len(val_scores)} 个 val_score")
            
            # 构建 previous_factors（供后续所有 generate_optimized_factors 调用使用）
            previous_factors = [
                {'code': c, 'report': r, 'train_score': ts, 'val_score': vs}
                for c, r, ts, vs in zip(codes, reports, train_scores, val_scores)
            ]
            
            # Step 2: 生成新因子
            self.logger.info("Step 2: 生成新因子（基于报告学习）")
            existing_codes = self._collect_existing_codes(round_num - 1)
            new_codes = self.positive_agent.generate_optimized_factors(
                previous_factors=previous_factors,
                round_num=round_num,
                save_response=True,
                n_override=self.factors_per_round,
                existing_codes=existing_codes
            )
            self.total_factors_generated += len(new_codes)  # ← 这行需要补上

            # Step 3: 评估新因子（含补货）
            self.logger.info("Step 3: 评估新因子（含自动补货）")
            attempt = 0
            df_eval_all = pd.DataFrame()
            pending_codes = list(new_codes)
            seen_keys = set()

            while True:
                if not pending_codes:
                    if attempt >= self.max_refills:
                        break
                    deficit = self.factors_per_round - len(df_eval_all)
                    if deficit <= 0:
                        break
                    self.logger.info(f"无待评估代码，补货 {deficit} 个（attempt {attempt+1}/{self.max_refills}）")
                    more_codes = self.positive_agent.generate_optimized_factors(
                        previous_factors=previous_factors,
                        round_num=round_num,
                        save_response=True,
                        n_override=deficit,
                        existing_codes=existing_codes + df_eval_all.get('code', pd.Series(dtype=str)).astype(str).tolist()
                    )
                    self.total_factors_generated += len(more_codes)
                    pending_codes = more_codes
                    attempt += 1
                    continue

                df_eval = self._evaluate_codes(pending_codes)
                if df_eval is not None and not df_eval.empty:
                    # 去重
                    df_eval['__key__'] = (
                        df_eval['code'].astype(str)
                        .str.replace('"', "'", regex=False)
                        .str.replace(r"\s+", "", regex=True)
                        .str.lower()
                    )
                    df_eval = df_eval[~df_eval['__key__'].isin(seen_keys)]
                    seen_keys.update(df_eval['__key__'])
                    df_eval_all = pd.concat([df_eval_all, df_eval.drop(columns='__key__')], ignore_index=True)

                # 达标？
                if len(df_eval_all) >= self.factors_per_round:
                    break

                # 补货
                if attempt >= self.max_refills:
                    self.logger.warning(f"已达最大补货次数 {self.max_refills}，本轮有效 {len(df_eval_all)}/{self.factors_per_round}")
                    break
                deficit = self.factors_per_round - len(df_eval_all)
                self.logger.info(f"有效结果不足，补货 {deficit} 个（attempt {attempt+1}/{self.max_refills}）")
                more_codes = self.positive_agent.generate_optimized_factors(
                    previous_factors=previous_factors,
                    round_num=round_num,
                    save_response=True,
                    n_override=deficit,
                    existing_codes=existing_codes + df_eval_all.get('code', pd.Series(dtype=str)).astype(str).tolist()
                )
                self.total_factors_generated += len(more_codes)
                pending_codes = more_codes
                attempt += 1

            # 最终检查
            if self.require_full_round and len(df_eval_all) < self.factors_per_round:
                self.logger.error(f"本轮评估后仍不足 {len(df_eval_all)}/{self.factors_per_round}，且 REQUIRE_FULL_ROUND=True")
                return None
            if df_eval_all is None or df_eval_all.empty:
                self.logger.error("评估得到空结果")
                return None

            # Step 4: 生成报告（新增步骤）← 核心改动
            self.logger.info("Step 4: 生成因子报告")
            df_eval_all = self._generate_reports(df_eval_all, round_num)

            # Step 5: 保存本轮结果（含 factor_report 列）
            out_csv = self.memory_manager.save_round_results(round_num, df_eval_all)
            self.logger.info(f"本轮结果已保存：{out_csv}")

            # Step 6: 汇总并返回本轮指标
            result = self._summarize_round(round_num, df_eval_all, time.time() - t0)
            self.logger.info(f"第{round_num}轮摘要：{result}")
            return result

        except Exception as e:
            self.logger.error(f"单轮失败（round={round_num}）: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    # ============== 新增：报告生成 ==============
    def _generate_reports(self, df_eval: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """
        为每个因子生成报告
        
        核心原则：只使用 train 指标（防止 val 泄露）
        """
        self.logger.info(f"为 {len(df_eval)} 个因子生成报告...")
        
        reports = []
        for idx, row in df_eval.iterrows():
            try:
                # 只使用 train 指标（防止泄露）
                metrics = FactorMetrics(
                    sharpe=row.get('train_sharpe'),
                    ann_ret=row.get('train_ann_ret'),
                    max_dd=row.get('train_max_dd'),
                    coverage=row.get('train_coverage'),
                    train_score=row.get('train_score'),
                    val_score=None  # ← 不传递 val（防止泄露）
                )
                
                report = generate_factor_report(
                    code=row['code'],
                    metrics=metrics,
                    round_id=round_num,
                    factor_id=row.get('factor_id', ''),
                    template_path=self.report_template_file,
                    temperature=self.report_temperature,
                    max_tokens=self.report_max_tokens
                )
                reports.append(report)
                
            except Exception as e:
                self.logger.warning(f"生成报告失败（factor_id={row.get('factor_id', '?')}）: {e}")
                reports.append("(report generation failed)")
        
        df_eval['factor_report'] = reports
        self.logger.info("报告生成完成")
        return df_eval

    # ============== 工具/评估/统计 ==============
    def _collect_existing_codes(self, last_round: Optional[int]) -> List[str]:
        """收集已尝试过的 code（用于去重）"""
        codes = set()

        # baseline
        try:
            if self.baseline_csv.exists():
                dfb = pd.read_csv(self.baseline_csv, low_memory=False)
                if 'code' in dfb.columns:
                    codes.update([str(c) for c in dfb['code'].astype(str).tolist()])
        except Exception:
            pass

        # 历史轮次
        rounds_to_scan = range(1, (last_round or 0) + 1)
        for r in rounds_to_scan:
            p = self.results_dir / f"round_{r}_factor_metrics.csv"
            if not p.exists():
                continue
            try:
                dfr = pd.read_csv(p, low_memory=False)
                if 'code' in dfr.columns:
                    codes.update([str(c) for c in dfr['code'].astype(str).tolist()])
            except Exception:
                continue

        return list(codes)

    def _evaluate_codes(self, codes: List[str]) -> Optional[pd.DataFrame]:
        """评估因子代码"""
        if not codes:
            return None

        try:
            df = batch_evaluate(
                factors=[{"code": c} for c in codes],
                splits=self.splits,
                ret_col=self.ret_col,
                date_col=self.date_col,
                periods_per_year=self.periods_per_year
            )
            return df
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return None

    def _summarize_round(self, round_num: int, df_eval: pd.DataFrame, elapsed: float) -> Dict:
        """计算本轮关键信息"""
        d = {}
        try:
            d['round'] = round_num
            d['round_num'] = round_num
            d['n'] = int(len(df_eval))
            d['avg_val_score'] = float(pd.to_numeric(df_eval['val_score'], errors='coerce').mean())
            d['max_val_score'] = float(pd.to_numeric(df_eval['val_score'], errors='coerce').max())
            d['avg_train_score'] = float(pd.to_numeric(df_eval['train_score'], errors='coerce').mean())
            d['max_train_score'] = float(pd.to_numeric(df_eval['train_score'], errors='coerce').max())
            d['elapsed_sec'] = float(elapsed)
        except Exception:
            pass

        # 更新最佳追踪
        try:
            if d.get('avg_val_score', -1e9) > self.best_avg_val + 1e-12:
                self.best_avg_val = d['avg_val_score']
                self.best_round = round_num
            self.best_top_val = max(self.best_top_val, d.get('max_val_score', -1e9))
        except Exception:
            pass
        return d

    def _compute_baseline_summary(self) -> Dict:
        """读取 baseline CSV，给出简单统计"""
        out = {'exists': False}
        try:
            if self.baseline_csv.exists():
                df = pd.read_csv(self.baseline_csv, low_memory=False)
                out['exists'] = True
                if not df.empty:
                    out['n'] = int(len(df))
                    out['avg_train'] = float(pd.to_numeric(df.get('train_score', pd.Series()), errors='coerce').mean())
                    out['avg_val'] = float(pd.to_numeric(df.get('val_score', pd.Series()), errors='coerce').mean())
                    out['best_val'] = float(pd.to_numeric(df.get('val_score', pd.Series()), errors='coerce').max())
        except Exception:
            pass
        return out

    # ============== 终止条件 ==============
    def _check_early_stopping(self, round_result: Dict) -> bool:
        """早停逻辑"""
        current_round = round_result.get('round_num', round_result.get('round', 0))
        
        # 最少轮数保护
        if len(self.history) + 1 < self.min_rounds:
            return False
        if self.patience <= 0:
            return False
        
        # 简单 patience 逻辑
        cur = round_result.get('avg_val_score', -1e9)
        if cur > self.best_avg_val + self.min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.logger.info(f"Patience早停：连续 {self.patience_counter} 轮无提升")
                return True

        return False

    def _check_threshold_stopping(self, round_result: Dict) -> bool:
        """达到平均 val_score 阈值即停止"""
        if self.val_score_threshold is None:
            return False
        cur = round_result.get('avg_val_score', -1e9)
        return cur >= float(self.val_score_threshold)

    # ============== 汇总导出 ==============
    def _load_round_csv(self, path: Path, round_num: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        df = pd.read_csv(path, low_memory=False)
        df['round'] = round_num
        return df

    def _write_all_rounds_aggregate(self) -> None:
        """baseline + round_*.csv 全量拼接"""
        frames = []

        # 读取 baseline（作为 round 0）
        baseline_path = self.memory_manager.baseline_csv
        
        if baseline_path.exists():
            try:
                df0 = pd.read_csv(baseline_path, low_memory=False)
                df0['round'] = 0
                frames.append(df0)
                self.logger.info(f"汇总：使用 baseline from {baseline_path}")
            except Exception as e:
                self.logger.warning(f"读取 baseline 失败: {e}")
        else:
            self.logger.warning(f"Baseline文件不存在: {baseline_path}")

        # 历史轮次
        for p in sorted(self.results_dir.glob("round_*_factor_metrics.csv")):
            try:
                m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
                r = int(m.group(1)) if m else -1
                if r == 0:
                    continue
                dfr = self._load_round_csv(p, r)
                if dfr is not None and not dfr.empty:
                    frames.append(dfr)
            except Exception as e:
                self.logger.warning(f"读取 {p} 失败: {e}")
                continue

        if not frames:
            self.logger.warning("汇总时未找到任何有效数据帧")
            return

        all_df = pd.concat(frames, ignore_index=True)
        out = self.results_dir / "all_rounds_factors.csv"
        all_df.to_csv(out, index=False, encoding="utf-8")
        self.logger.info(f"已生成汇总文件: {out}，共 {len(all_df)} 行，{all_df['round'].nunique()} 个轮次")
    
    def _generate_final_report(self, baseline_summary: Dict) -> Dict:
        """迭代结束后的汇总报告"""
        frames = []

        # 读取 baseline（作为 round 0）
        baseline_path = self.memory_manager.baseline_csv
        
        if baseline_path.exists():
            try:
                df0 = pd.read_csv(baseline_path, low_memory=False)
                df0['round'] = 0
                frames.append(df0)
                self.logger.info(f"报告：使用 baseline from {baseline_path}")
            except Exception as e:
                self.logger.warning(f"读取 baseline 失败: {e}")
        else:
            self.logger.warning(f"Baseline文件不存在: {baseline_path}")

        # 各轮
        for p in sorted(self.results_dir.glob("round_*_factor_metrics.csv")):
            try:
                m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
                r = int(m.group(1)) if m else -1
                if r == 0:
                    continue
                dfr = self._load_round_csv(p, r)
                if dfr is not None and not dfr.empty:
                    frames.append(dfr)
            except Exception as e:
                self.logger.warning(f"读取 {p} 失败：{e}")

        if not frames:
            self.logger.warning("报告生成时未找到任何有效数据")
            return {
                "rounds": len(self.history),
                "best_round": self.best_round,
                "best_avg_val": self.best_avg_val,
                "best_top_val": self.best_top_val,
                "summary_file": None,
                "baseline": baseline_summary,
                "total_factors_generated": self.total_factors_generated
            }

        all_df = pd.concat(frames, ignore_index=True)

        # 计算每轮均值
        mean_cols = [
            'train_score',
            'val_score', 'val_coverage', 'val_ann_ret', 'val_sharpe',
            'val_max_dd', 'val_D', 'val_diversity', 'val_autocorr', 'val_skew'
        ]
        mean_cols = [c for c in mean_cols if c in all_df.columns]

        round_summary = (all_df
                         .groupby('round', as_index=False)[mean_cols]
                         .mean(numeric_only=True))
        out_summary = self.results_dir / "round_summary_mean.csv"
        round_summary.to_csv(out_summary, index=False)
        self.logger.info(f"已生成每轮均值报告: {out_summary}")

        report = {
            "rounds": len(self.history),
            "best_round": self.best_round,
            "best_avg_val": self.best_avg_val,
            "best_top_val": self.best_top_val,
            "summary_file": str(out_summary),
            "baseline": baseline_summary,
            "total_factors_generated": self.total_factors_generated
        }
        return report

    def _save_optimization_summary(self, report: Dict) -> None:
        out = self.results_dir / "optimization_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"优化报告已保存: {out}")