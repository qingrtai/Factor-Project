# iterative_baseline/iterator.py
import os
import re
import json
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .memory_manager import MemoryManager
from .positive_agents import PositiveAgent
from core.factor_evaluator import batch_evaluate  # ← 改为这个
from core.data_loader import load_splits          # ← 新增
from shared.config_loader import load_global_config  # ← 新增

# 兼容两种 config 入口：dict CONFIG 与模块常量
try:
    from .config import CONFIG as _CONFIG  # dict 风格
except Exception:
    _CONFIG = None
try:
    from . import config as _cfg_module    # 常量风格
except Exception:
    _cfg_module = None


def _cfg_get(key: str, default=None):
    if isinstance(_CONFIG, dict) and key in _CONFIG:
        return _CONFIG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default


class IterativeOptimizer:
    def __init__(self,
                 baseline_file: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 logs_dir: Optional[str] = None):
        # 路径
        self.project_root = Path(_cfg_get('PROJECT_ROOT', Path(__file__).resolve().parent))
        self.results_dir = Path(results_dir or _cfg_get('RESULTS_DIR', self.project_root / 'results'))
        self.logs_dir = Path(logs_dir or _cfg_get('LOGS_DIR', self.project_root / 'logs'))
        self.baseline_csv = Path(
            baseline_file or
            _cfg_get('BASELINE_CSV',
                     _cfg_get('BASELINE_FILE', self.project_root.parent / 'factor_baseline' / 'results' / 'baseline_factor_metrics.csv'))
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        self.max_rounds = int(_cfg_get('MAX_ROUNDS', _cfg_get('N_ROUNDS_MAX', 5)))
        self.patience = int(_cfg_get('EARLY_STOPPING_PATIENCE', _cfg_get('PATIENCE', 2)))
        self.min_delta = float(_cfg_get('MIN_DELTA', 1e-3))
        self.min_rounds = int(_cfg_get('MIN_ROUNDS', 1))  # 最少轮数保护
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', _cfg_get('N_FACTORS', 10)))
        self.val_score_threshold = _cfg_get('VAL_SCORE_THRESHOLD', None)
        self.require_full_round = bool(_cfg_get('REQUIRE_FULL_ROUND', False))
        self.max_refills = int(_cfg_get('MAX_REFILL_CALLS', _cfg_get('MAX_RETRIES', 3)))

        # 评分与切分说明（用于日志）
        # === 改动点1：读取评分与切分说明，便于运行时留痕 ===
        self.score_desc = _cfg_get('SCORE_DESC', "Score = (Sharpe + AnnRet + D) / 3")
        self.derived_desc = _cfg_get('DERIVED_METRIC_DESC', "D = 1 - MaxDD")
        self.train_years = _cfg_get('TRAIN_YEARS', (1981, 2000))
        self.val_years   = _cfg_get('VAL_YEARS',   (2001, 2014))
        self.test_years  = _cfg_get('TEST_YEARS',  (2015, 2021))

        # 组件
        self.memory_manager = MemoryManager()
        self.positive_agent = PositiveAgent()

        # ← 删除这行：self.factor_evaluator = FactorEvaluator()
        # ← 新增：加载数据 splits（在初始化时加载一次，后续复用）
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

        # 其他状态
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
        self.logger.info("开始迭代因子优化（只记忆上一轮 code+train_score）")
        self.logger.info(f"最大轮次: {self.max_rounds} | 耐心: {self.patience} | MIN_DELTA: {self.min_delta} | MIN_ROUNDS: {self.min_rounds}")
        self.logger.info(f"每轮因子数: {self.factors_per_round}")
        self.logger.info(f"baseline: {self.baseline_csv}")
        # === 改动点2：打印评分公式与时间切分，保证日志清晰可追溯 ===
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

            # baseline + 各轮 11 列全量汇总
            try:
                self._write_all_rounds_aggregate()
                self.logger.info("已生成 all_rounds_factors.csv（包含 baseline 与各轮的标准列）")
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
            # Step 1: 读取上一轮记忆（code + train_score）
            last_round = None if round_num == 1 else (round_num - 1)
            self.logger.info("Step 1: 读取上一轮记忆（code+train_score）...")
            prev_codes, prev_trains = self.memory_manager.load_memory(last_round)  # (codes, train_scores)
            previous_factors = [{'code': c, 'train_score': t} for c, t in zip(prev_codes, prev_trains)]
            if not previous_factors:
                self.logger.error("上一轮记忆为空")
                return None
            prev_avg = float(np.mean(prev_trains)) if prev_trains else float('nan')
            self.logger.info(f"上一轮平均 train_score: {prev_avg:.4f}")

            # Step 2: 生成新因子（依赖 code+train_score 记忆）
            self.logger.info("Step 2: 生成新因子（依赖 code+train_score 记忆）")
            existing_codes = self._collect_existing_codes(round_num - 1)
            new_codes = self.positive_agent.generate_optimized_factors(
                previous_factors=previous_factors,
                round_num=round_num,
                save_response=True,
                n_override=self.factors_per_round,
                existing_codes=existing_codes
            )
            self.total_factors_generated += len(new_codes)

            # Step 3: 评估 + 不足则补货（最多 MAX_REFILL_CALLS 次）
            self.logger.info("Step 3: 评估新因子（含自动补货）")
            attempt = 0
            df_eval_all = pd.DataFrame()
            pending_codes = list(new_codes)
            seen_keys = set()

            while True:
                if not pending_codes:
                    # 初次就空，或上一次补货没拿到新代码
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
                    # 合并去重（按 code 规范化键）
                    df_eval['__key__'] = (
                        df_eval['code'].astype(str)
                        .str.replace('"', "'", regex=False)
                        .str.replace(r"\s+", "", regex=True)
                        .str.lower()
                    )
                    df_eval = df_eval[~df_eval['__key__'].isin(seen_keys)]
                    seen_keys.update(df_eval['__key__'])
                    df_eval_all = pd.concat([df_eval_all, df_eval.drop(columns='__key__')], ignore_index=True)

                # 达标？退出
                if len(df_eval_all) >= self.factors_per_round:
                    break

                # 不足则补货
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

            # Step 4: 保存本轮结果（标准列）
            out_csv = self.memory_manager.save_round_results(round_num, df_eval_all)
            self.logger.info(f"本轮结果已保存：{out_csv}")

            # Step 5: 汇总并返回本轮指标（按 val_score 做对比/早停）
            result = self._summarize_round(round_num, df_eval_all, time.time() - t0)
            self.logger.info(f"第{round_num}轮摘要：{result}")
            return result

        except Exception as e:
            self.logger.error(f"单轮失败（round={round_num}）: {e}")
            return None

    # ============== 工具/评估/统计 ==============
    def _collect_existing_codes(self, last_round: Optional[int]) -> List[str]:
        """
        收集已尝试过的 code（用于去重）。包含：
        - baseline 的 code
        - 历史各轮的 code
        """
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
        """
        将若干 code 交给评估器，返回与 baseline 一致的标准列。
        """
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
            
            # ========== 新增：过滤严重过拟合的因子 ========== #
            if df is not None and not df.empty:
                # 计算 val/train 比值
                df['_val_train_ratio'] = df['val_score'] / df['train_score'].replace(0, 0.001)
                
                # 设定阈值（保留 6x 以下的，过滤 8x 以上的）
                MAX_RATIO = 8.0
                
                before = len(df)
                # 过滤掉比值过高的（绝对值）
                df = df[df['_val_train_ratio'].abs() <= MAX_RATIO]
                after = len(df)
                
                if after < before:
                    self.logger.warning(
                        f"⚠️ 过滤掉 {before - after} 个严重过拟合因子 "
                        f"(|val/train| > {MAX_RATIO}x)"
                    )
                
                # 删除临时列
                if '_val_train_ratio' in df.columns:
                    df = df.drop(columns=['_val_train_ratio'])
            # ============================================== #
            
            return df
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return None

    def _summarize_round(self, round_num: int, df_eval: pd.DataFrame, elapsed: float) -> Dict:
        """
        计算并返回本轮关键信息；平均/最大 val_score 作为主判据
        """
        d = {}
        try:
            d['round'] = round_num
            d['n'] = int(len(df_eval))
            d['avg_val_score'] = float(pd.to_numeric(df_eval['val_score'], errors='coerce').mean())
            d['max_val_score'] = float(pd.to_numeric(df_eval['val_score'], errors='coerce').max())
            d['avg_train_score'] = float(pd.to_numeric(df_eval['train_score'], errors='coerce').mean())
            d['max_train_score'] = float(pd.to_numeric(df_eval['train_score'], errors='coerce').max())
            d['elapsed_sec'] = float(elapsed)
        except Exception:
            pass

        # 更新“最佳”追踪
        try:
            if d.get('avg_val_score', -1e9) > self.best_avg_val + 1e-12:
                self.best_avg_val = d['avg_val_score']
                self.best_round = round_num
            self.best_top_val = max(self.best_top_val, d.get('max_val_score', -1e9))
        except Exception:
            pass
        return d

    def _compute_baseline_summary(self) -> Dict:
        """
        读取 baseline CSV，给出简单统计（不影响流程）
        """
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
        """
        简单的“耐心 min_delta”逻辑：和历史最佳平均 val 比较
        """
        if len(self.history) + 1 < self.min_rounds:
            return False
        if self.patience <= 0:
            return False
        cur = round_result.get('avg_val_score', -1e9)
        if cur > self.best_avg_val + self.min_delta:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

    def _check_threshold_stopping(self, round_result: Dict) -> bool:
        """
        达到平均 val_score 阈值即停止（如果配置了阈值）
        """
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
        """baseline + round_*.csv 全量拼接写到 results/all_rounds_factors.csv"""
        frames = []
        
        # ★★★ 优先使用 round_0，不存在才用 baseline ★★★
        round_0_path = self.project_root.parent / "factor_baseline" / "results" / "round_0_factor_metrics.csv"
        
        if round_0_path.exists():
            # 使用已有的 round_0
            try:
                df0 = pd.read_csv(round_0_path, low_memory=False)
                df0['round'] = 0
                frames.append(df0)
                self.logger.info(f"汇总：使用已有 round_0 from {round_0_path}")
            except Exception as e:
                self.logger.warning(f"读取 round_0 失败: {e}，回退到 baseline")
                # 回退到 baseline
                if self.baseline_csv.exists():
                    try:
                        dfb = pd.read_csv(self.baseline_csv, low_memory=False)
                        dfb['round'] = 0
                        frames.append(dfb)
                        self.logger.info(f"汇总：使用 baseline from {self.baseline_csv}")
                    except Exception as e2:
                        self.logger.warning(f"读取 baseline 也失败: {e2}")
        else:
            # round_0 不存在，使用 baseline
            self.logger.info(f"未找到 round_0，使用 baseline from {self.baseline_csv}")
            if self.baseline_csv.exists():
                try:
                    dfb = pd.read_csv(self.baseline_csv, low_memory=False)
                    dfb['round'] = 0
                    frames.append(dfb)
                except Exception as e:
                    self.logger.warning(f"读取 baseline 失败: {e}")
        
        # 历史轮次（round_1, round_2, ...）
        for p in sorted(self.results_dir.glob("round_*_factor_metrics.csv")):
            try:
                m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
                r = int(m.group(1)) if m else -1
                # 跳过可能在 iterative_baseline/results 下的 round_0（如果有的话）
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
        """
        迭代结束后的汇总报告：
        - 写出 round_summary_mean.csv（每轮均值）
        - 返回 dict 供 main 打印/持久化
        """
        frames = []
        
        # ★★★ 优先使用 round_0，不存在才用 baseline ★★★
        round_0_path = self.project_root.parent / "factor_baseline" / "results" / "round_0_factor_metrics.csv"
        
        if round_0_path.exists():
            try:
                df0 = pd.read_csv(round_0_path, low_memory=False)
                df0['round'] = 0
                frames.append(df0)
                self.logger.info(f"报告：使用已有 round_0 from {round_0_path}")
            except Exception as e:
                self.logger.warning(f"读取 round_0 失败: {e}，回退到 baseline")
                # 回退到 baseline
                if self.baseline_csv.exists():
                    try:
                        dfb = pd.read_csv(self.baseline_csv, low_memory=False)
                        dfb['round'] = 0
                        frames.append(dfb)
                        self.logger.info(f"报告：使用 baseline from {self.baseline_csv}")
                    except Exception as e2:
                        self.logger.warning(f"读取 baseline 也失败：{e2}")
        else:
            # round_0 不存在，使用 baseline
            self.logger.info(f"未找到 round_0，使用 baseline from {self.baseline_csv}")
            if self.baseline_csv.exists():
                try:
                    dfb = pd.read_csv(self.baseline_csv, low_memory=False)
                    dfb['round'] = 0
                    frames.append(dfb)
                except Exception as e:
                    self.logger.warning(f"读取 baseline 失败：{e}")

        # 各轮（跳过 round_0 避免重复）
        for p in sorted(self.results_dir.glob("round_*_factor_metrics.csv")):
            try:
                m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
                r = int(m.group(1)) if m else -1
                # 跳过可能在 iterative_baseline/results 下的 round_0
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
        # === 改动点3：把 D 加入均值汇总列（保留 max_dd） ===
        # 计算每轮均值
        # 保留 train_score + 所有 val 指标
        mean_cols = [
            'train_score',  # 训练集参考
            # 所有验证集指标
            'val_score', 'val_coverage', 'val_ann_ret', 'val_sharpe', 
            'val_max_dd', 'val_D', 'val_diversity', 'val_autocorr', 'val_skew'
        ]
        # 过滤掉不存在的列
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
