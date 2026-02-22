#!/usr/bin/env python3
# iterative_baseline/run_iterative.py
"""
迭代因子优化主程序
- 完整优化：IterativeOptimizer.run_optimization()
- 单轮调试：只读取上一轮记忆（code + train_score）
- 单因子测试：用 FactorEvaluator.evaluate_factors([code])
- 结果分析：report 模式读取 iterator 写出的汇总文件
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
import os
import json
import pandas as pd

# 组件
from .config import CONFIG, validate_config, print_config_summary
from .iterator import IterativeOptimizer
from .memory_manager import MemoryManager
from .positive_agents import PositiveAgent
from core.factor_evaluator import batch_evaluate
from core.data_loader import load_splits
from shared.config_loader import load_global_config

# 可选：加载 .env（便于读取 SCHOOL_API_KEY / OPENAI_API_KEY / AZURE_OPENAI_API_KEY）
try:
    from dotenv import load_dotenv
    # 明确定位到当前文件所在目录或项目根
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('main')


def show_api_key_visibility():
    """打印当前环境中可见的 API Key 状态"""
    logger = logging.getLogger('main')
    keys = ["SCHOOL_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GPT_API_KEY"]
    logger.info("=== API KEY VISIBILITY ===")
    for k in keys:
        v = os.getenv(k)
        status = "SET" if v and v.strip() else "NOT SET"
        logger.info(f"{k}: {status}")


def check_environment() -> bool:
    """
    环境与依赖检查（不强制阻断 GPT 密钥，避免本地离线评估被拦）
    """
    logger = logging.getLogger('main')
    try:
        # Python 版本
        if sys.version_info < (3, 8):
            logger.error("Python 版本需要 >= 3.8")
            return False

        # 必要包
        required = ['pandas', 'numpy']
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            logger.error(f"缺少必要的包: {missing}")
            logger.error("请运行: pip install " + " ".join(missing))
            return False

        # 配置校验
        errors, warnings = validate_config()
        if errors:
            logger.error("配置错误:")
            for e in errors:
                logger.error(f"  - {e}")
            return False
        if warnings:
            logger.warning("配置警告:")
            for w in warnings:
                logger.warning(f"  - {w}")

        # GPT 密钥检查（含 SCHOOL_API_KEY）
        ok_keys = any(os.getenv(k) for k in [
            "SCHOOL_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GPT_API_KEY"
        ])
        if not ok_keys:
            logger.warning(
                "未检测到 SCHOOL_API_KEY / OPENAI_API_KEY / AZURE_OPENAI_API_KEY / GPT_API_KEY，"
                "如需调用 GPT，请在 .env 中配置。"
            )
        else:
            show_api_key_visibility()

        logger.info("环境检查通过")
        return True

    except Exception as e:
        logger.error(f"环境检查失败: {e}")
        logger.error(traceback.format_exc())
        return False


def _print_report_summary(report: dict) -> None:
    """友好地打印 iterator.run_optimization() 返回的报告摘要"""
    logger = logging.getLogger('main')
    if not isinstance(report, dict):
        logger.info(report)
        return
    logger.info("=== 优化摘要 ===")
    for k in ["rounds", "best_round", "best_avg_val", "best_top_val", "summary_file", "total_factors_generated"]:
        if k in report:
            logger.info(f"{k}: {report[k]}")
    baseline = report.get("baseline", {})
    if isinstance(baseline, dict) and baseline.get("exists"):
        logger.info(f"baseline: n={baseline.get('n')} avg_train={baseline.get('avg_train')} avg_val={baseline.get('avg_val')} best_val={baseline.get('best_val')}")


def run_full_optimization() -> dict:
    """执行完整多轮优化"""
    logger = logging.getLogger('main')
    try:
        logger.info("开始完整迭代优化...")
        logger.info("Scoring = (Sharpe + AnnRet + D) / 3; D = 1 - MaxDD; Split: Train=1961–2000, Val=2001–2010, Test=2011–2025")

        optimizer = IterativeOptimizer(
            baseline_file=CONFIG['BASELINE_FILE'],
            results_dir=CONFIG['RESULTS_DIR'],
            logs_dir=CONFIG['LOGS_DIR']
        )
        result = optimizer.run_optimization()
        logger.info("迭代优化完成")
        _print_report_summary(result)
        return result
    except Exception as e:
        logger.error(f"迭代优化失败: {e}")
        logger.error(traceback.format_exc())
        raise


def run_single_round(round_num: int, previous_round: int = None) -> pd.DataFrame:
    """运行单轮优化（只记忆上一轮）"""
    logger = logging.getLogger('main')
    try:
        logger.info(f"运行单轮优化 (轮次 {round_num})...")
        
        # 加载配置和数据
        from experiments.iterative_baseline.config import (
            FACTORS_PER_ROUND, RESULTS_DIR
        )
        g = load_global_config()
        splits = load_splits(
            g["data"]["raw_file"],
            g["schema"]["date_col"],
            g["years"],
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        
        # 读取记忆
        memory_manager = MemoryManager()
        positive_agent = PositiveAgent()
        
        codes, trains = memory_manager.load_memory(previous_round)
        previous_factors = [{'code': c, 'train_score': t} for c, t in zip(codes, trains)]
        
        if not previous_factors:
            logger.error("上一轮记忆为空")
            return pd.DataFrame()
        
        avg_train = float(pd.to_numeric(pd.Series(trains), errors='coerce').mean())
        logger.info(f"上一轮平均 train_score: {avg_train:.4f}")
        
        # 生成新因子
        new_codes = positive_agent.generate_optimized_factors(
            previous_factors=previous_factors,
            round_num=round_num,
            n=FACTORS_PER_ROUND
        )
        
        if not new_codes:
            logger.error("未生成到新的因子代码")
            return pd.DataFrame()
        
        # 评估（使用 batch_evaluate）
        df_eval = batch_evaluate(
            factors=[{"code": c} for c in new_codes],
            splits=splits,
            ret_col=g["schema"]["ret_col"],
            date_col=g["schema"]["date_col"],
            periods_per_year=int(g.get("freq_per_year", 4))
        )
        
        if df_eval is None or df_eval.empty:
            logger.error("评估得到空结果")
            return pd.DataFrame()
        
        # 保存结果
        out_csv_path = memory_manager.save_round_results(round_num, df_eval)
        logger.info(f"本轮结果已保存：{out_csv_path}")
        
        return df_eval
        
    except Exception as e:
        logger.error(f"单轮优化失败: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_best_factors(top_k: int = 5) -> pd.DataFrame:
    """按 val_score 挑选历史最优因子"""
    results_dir = Path(CONFIG['RESULTS_DIR'])
    dfs = []
    for p in sorted(results_dir.glob("round_*_factor_metrics.csv")):
        try:
            df = pd.read_csv(p)
            import re
            df['round'] = int(re.findall(r"round_(\d+)_factor_metrics\.csv", p.name)[0])
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values('val_score', ascending=False).head(top_k)
    return all_df.reset_index(drop=True)


def show_report():
    """读取 iterator 产出的报告"""
    logger = logging.getLogger('main')
    results_dir = Path(CONFIG['RESULTS_DIR'])
    report_file = results_dir / "optimization_report.json"
    summary_file = results_dir / "round_summary_mean.csv"

    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
            _print_report_summary(report)
        except Exception as e:
            logger.warning(f"读取 {report_file} 失败：{e}")
    else:
        logger.info("未找到 optimization_report.json")

    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            logger.info("=== 每轮均值（预览） ===")
            logger.info(df.head(10).to_string(index=False))
        except Exception as e:
            logger.warning(f"读取 {summary_file} 失败：{e}")
    else:
        logger.info("未找到 round_summary_mean.csv")

    all_file = results_dir / "all_rounds_factors.csv"
    if all_file.exists():
        logger.info(f"all_rounds_factors.csv: {all_file}")
    else:
        logger.info("未找到 all_rounds_factors.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative factor optimization")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "single", "best", "print-config", "report"],
                        help="运行模式：full=完整优化；single=单轮；best=查看最优；print-config=打印配置摘要；report=查看汇总报告")
    parser.add_argument("--round", type=int, default=1, help="round编号（single模式）")
    parser.add_argument("--prev", type=int, default=None, help="上一轮编号（single模式）")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    if not check_environment():
        sys.exit(1)

    if args.mode == "print-config":
        print_config_summary()
        sys.exit(0)

    if args.mode == "full":
        result = run_full_optimization()
        print("完成。摘要如下：")
        print(result)
        sys.exit(0)

    if args.mode == "single":
        df = run_single_round(args.round, args.prev)
        if df is None or df.empty:
            print("单轮运行未得到有效结果。")
            sys.exit(2)
        print(df.head(10))
        sys.exit(0)

    if args.mode == "best":
        best = get_best_factors(top_k=10)
        if best is None or best.empty:
            print("没有找到历史评估结果。")
            sys.exit(3)
        print(best)
        sys.exit(0)

    if args.mode == "report":
        show_report()
        sys.exit(0)

    logger.error("未知模式")
    sys.exit(99)


if __name__ == "__main__":
    main()
