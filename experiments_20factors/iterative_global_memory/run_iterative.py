#!/usr/bin/env python3
"""
全局记忆迭代因子优化主程序
- 完整优化：IterativeOptimizer.run_optimization()（使用累积全局记忆）
- 单轮调试：传递累积记忆给 GlobalMemoryAgent
- 单因子测试：用 batch_evaluate([code])
- 结果分析：report 模式读取汇总文件
- 记忆查看：memory 模式查看累积记忆统计
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
from .memory_manager import GlobalMemoryManager
from .positive_agents import GlobalMemoryAgent
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
    for k in ["rounds", "best_round", "best_avg_val", "best_top_val", 
              "summary_file", "total_factors_generated", "generation_strategy"]:
        if k in report:
            logger.info(f"{k}: {report[k]}")
    baseline = report.get("baseline", {})
    if isinstance(baseline, dict) and baseline.get("exists"):
        logger.info(f"baseline: n={baseline.get('n')} avg_train={baseline.get('avg_train')} "
                   f"avg_val={baseline.get('avg_val')} best_val={baseline.get('best_val')}")


def run_full_optimization() -> dict:
    """执行完整多轮优化（使用全局累积记忆）"""
    logger = logging.getLogger('main')
    try:
        logger.info("开始完整迭代优化（全局累积记忆模式）...")
        logger.info("Scoring = (Sharpe + AnnRet + D) / 3; D = 1 - MaxDD; "
                   "Split: Train=1961–2000, Val=2001–2010, Test=2011–2025")
        logger.info(f"生成策略: {CONFIG.get('GENERATION_STRATEGY', 'auto_mix')}")
        logger.info(f"Top-K: {CONFIG.get('GLOBAL_TOP_K', 8)}, Bottom-K: {CONFIG.get('GLOBAL_BOTTOM_K', 8)}")

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


def run_single_round(round_num: int) -> pd.DataFrame:
    """
    运行单轮优化（使用累积全局记忆）
    注意：与 baseline 不同，这里会加载所有历史记忆而非仅上一轮
    """
    logger = logging.getLogger('main')
    try:
        logger.info(f"运行单轮优化（全局记忆模式，轮次 {round_num}）...")
        
        # 加载配置和数据
        from experiments.iterative_global_memory.config import (
            FACTORS_PER_ROUND, RESULTS_DIR, GENERATION_STRATEGY
        )
        g = load_global_config()
        splits = load_splits(
            g["data"]["raw_file"],
            g["schema"]["date_col"],
            g["years"],
            id_col=g["schema"]["id_col"],
            ret_col=g["schema"]["ret_col"]
        )
        
        # 读取累积记忆（关键区别：加载所有历史轮次）
        memory_manager = GlobalMemoryManager()
        positive_agent = GlobalMemoryAgent()
        
        cumulative_memory = memory_manager.get_cumulative_memory(current_round=round_num)
        
        if not cumulative_memory:
            logger.error("累积记忆为空")
            return pd.DataFrame()
        
        # 打印记忆摘要
        mem_summary = memory_manager.get_memory_summary(round_num)
        logger.info(f"累积轮次数: {mem_summary.get('total_rounds', 0)}")
        logger.info(f"累积因子总数: {mem_summary.get('total_factors', 0)}")
        logger.info(f"历史最佳 val_score: {mem_summary.get('overall_best_val_score', 0):.4f}")
        logger.info(f"历史平均 train_score: {mem_summary.get('overall_avg_train_score', 0):.4f}")
        
        # 生成新因子（传入累积记忆）
        new_codes = positive_agent.generate_optimized_factors(
            cumulative_memory=cumulative_memory,
            round_num=round_num,
            generation_strategy=GENERATION_STRATEGY,
            save_response=True,
            n_override=FACTORS_PER_ROUND
        )
        
        if not new_codes:
            logger.error("未生成到新的因子代码")
            return pd.DataFrame()
        
        logger.info(f"生成了 {len(new_codes)} 个新因子")
        
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
        
        # 打印统计
        val_scores = pd.to_numeric(df_eval['val_score'], errors='coerce').dropna()
        train_scores = pd.to_numeric(df_eval['train_score'], errors='coerce').dropna()
        logger.info(f"Val分数  - 平均: {val_scores.mean():.4f}, 最佳: {val_scores.max():.4f}")
        logger.info(f"Train分数 - 平均: {train_scores.mean():.4f}, 最佳: {train_scores.max():.4f}")
        
        return df_eval
        
    except Exception as e:
        logger.error(f"单轮优化失败: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def show_memory_stats(up_to_round: int = None):
    """查看累积记忆统计（全局记忆特有功能）"""
    logger = logging.getLogger('main')
    try:
        memory_manager = GlobalMemoryManager()
        
        # 确定查看到哪一轮
        if up_to_round is None:
            # 自动检测最新轮次
            results_dir = Path(CONFIG['RESULTS_DIR'])
            rounds = []
            for p in results_dir.glob("round_*_factor_metrics.csv"):
                import re
                m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
                if m:
                    rounds.append(int(m.group(1)))
            up_to_round = max(rounds) + 1 if rounds else 1
        
        logger.info(f"=== 累积记忆统计（截至 Round {up_to_round}）===")
        
        # 获取摘要
        summary = memory_manager.get_memory_summary(up_to_round)
        logger.info(f"总轮次数: {summary['total_rounds']}")
        logger.info(f"总因子数: {summary['total_factors']}")
        logger.info(f"最佳 val_score: {summary['overall_best_val_score']:.4f}")
        logger.info(f"平均 val_score: {summary['overall_avg_val_score']:.4f}")
        logger.info(f"最佳 train_score: {summary['overall_best_train_score']:.4f}")
        logger.info(f"平均 train_score: {summary['overall_avg_train_score']:.4f}")
        
        # 各轮详情
        logger.info("\n=== 各轮详情 ===")
        for rd in summary['round_details']:
            logger.info(f"Round {rd['round']} ({rd['type']}): "
                       f"n={rd['count']}, "
                       f"val={rd['avg_val_score']:.4f}±{rd['best_val_score']:.4f}, "
                       f"train={rd['avg_train_score']:.4f}±{rd['best_train_score']:.4f}")
        
        # 使用 flatten_for_gpt 查看当前会喂给 GPT 的因子
        logger.info("\n=== GPT 视角（Top-K采样后）===")
        flattened = memory_manager.flatten_for_gpt(
            current_round=up_to_round,
            top_k_per_round=CONFIG.get('GLOBAL_TOP_K', 8),
            deduplicate=True
        )
        logger.info(f"去重后总因子数: {len(flattened)}")
        if flattened:
            scores = [f.get('val_score', 0) for f in flattened]
            logger.info(f"分数范围: {min(scores):.4f} ~ {max(scores):.4f}")
            logger.info(f"平均分数: {sum(scores)/len(scores):.4f}")
        
    except Exception as e:
        logger.error(f"查看记忆统计失败: {e}")
        logger.error(traceback.format_exc())


def get_best_factors(top_k: int = 5) -> pd.DataFrame:
    """按 val_score 挑选历史最优因子"""
    results_dir = Path(CONFIG['RESULTS_DIR'])
    dfs = []
    
    # 读取 baseline
    baseline_csv = Path(CONFIG['BASELINE_FILE'])
    if baseline_csv.exists():
        try:
            df = pd.read_csv(baseline_csv)
            df['round'] = 0
            dfs.append(df)
        except Exception:
            pass
    
    # 读取各轮
    for p in sorted(results_dir.glob("round_*_factor_metrics.csv")):
        try:
            df = pd.read_csv(p)
            import re
            m = re.search(r"round_(\d+)_factor_metrics\.csv", p.name)
            if m:
                df['round'] = int(m.group(1))
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
            logger.info("=== 每轮均值（预览）===")
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
    parser = argparse.ArgumentParser(
        description="全局记忆迭代因子优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明:
  full         - 完整多轮优化（使用累积全局记忆）
  single       - 单轮调试（传递累积记忆）
  best         - 查看历史最优因子
  memory       - 查看累积记忆统计（全局记忆特有）
  print-config - 打印配置摘要
  report       - 查看优化报告

示例:
  python -m experiments.iterative_global_memory.run_iterative --mode full
  python -m experiments.iterative_global_memory.run_iterative --mode single --round 2
  python -m experiments.iterative_global_memory.run_iterative --mode memory --round 3
  python -m experiments.iterative_global_memory.run_iterative --mode best
        """
    )
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "single", "best", "memory", "print-config", "report"],
                        help="运行模式")
    parser.add_argument("--round", type=int, default=1, 
                        help="轮次编号（single/memory模式）")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        help="日志级别")
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
        print("\n完成。摘要如下：")
        print(result)
        sys.exit(0)

    if args.mode == "single":
        df = run_single_round(args.round)
        if df is None or df.empty:
            print("单轮运行未得到有效结果。")
            sys.exit(2)
        print("\n=== 评估结果预览 ===")
        print(df.head(10))
        sys.exit(0)

    if args.mode == "memory":
        show_memory_stats(args.round)
        sys.exit(0)

    if args.mode == "best":
        best = get_best_factors(top_k=10)
        if best is None or best.empty:
            print("没有找到历史评估结果。")
            sys.exit(3)
        print("\n=== 历史最优因子（Top 10）===")
        print(best[['round', 'factor_id', 'code', 'val_score', 'train_score']].to_string(index=False))
        sys.exit(0)

    if args.mode == "report":
        show_report()
        sys.exit(0)

    logger.error("未知模式")
    sys.exit(99)


if __name__ == "__main__":
    main()