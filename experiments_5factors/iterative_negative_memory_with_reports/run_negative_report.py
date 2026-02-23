# experiments/iterative_negative_memory_with_reports/run_negative_report.py
"""
Iterative Negative Memory WITH REPORTS - 主入口

核心改动：
- 记忆使用 factor_report 而非 train_score
- 负样本两步生成（代码 + 报告）
- 正样本生成基于报告学习
"""

import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入本实验模块
from experiments.iterative_negative_memory_with_reports.config import (
    CONFIG,
    MAX_ROUNDS,
    validate_config,
    print_config_summary,
)
from experiments.iterative_negative_memory_with_reports.iterator import FactorIterator

# 导入通用模块
from core.utils import setup_logger
from shared.config_loader import load_global_config


def main():
    """主函数"""
    start_time = time.time()
    
    # ========== Step 1: 配置校验 ========== #
    print_config_summary()
    errors, warnings = validate_config()
    
    if errors:
        print("\n❌ 配置错误:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    
    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")
    
    # ========== Step 2: 设置日志 ========== #
    log_file = CONFIG["LOGS_DIR"] / "experiment.log"
    logger = setup_logger(
        name="iterative_negative_memory_with_reports",
        level="INFO",
        to_file=str(log_file)
    )
    
    logger.info("=" * 80)
    logger.info("实验开始: Iterative Negative Memory WITH REPORTS")
    logger.info("=" * 80)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"结果目录: {CONFIG['RESULTS_DIR']}")
    
    # ========== Step 3: 初始化迭代器 ========== #
    try:
        iterator = FactorIterator()
    except Exception as e:
        logger.error(f"迭代器初始化失败: {e}")
        sys.exit(1)
    
    # ========== Step 4: 执行迭代 ========== #
    all_round_stats = []
    
    for round_num in range(1, MAX_ROUNDS + 1):
        try:
            stats = iterator.execute_round(round_num)
            all_round_stats.append(stats)
            
            # 检查是否失败
            if stats.get("status") != "success":
                logger.error(f"Round {round_num} 执行失败，停止实验")
                break
                
        except KeyboardInterrupt:
            logger.warning(f"用户中断实验（Round {round_num}）")
            break
        except Exception as e:
            logger.error(f"Round {round_num} 异常: {e}", exc_info=True)
            break
    
    # ========== Step 5: 生成最终报告 ========== #
    duration = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("实验完成")
    logger.info("=" * 80)
    logger.info(f"总耗时: {duration:.2f}s ({duration/60:.1f}min)")
    logger.info(f"完成轮次: {len(all_round_stats)}/{MAX_ROUNDS}")
    
    # 全局统计
    global_stats = iterator.get_global_stats()
    logger.info(f"总生成因子数: {global_stats['total_factors_generated']}")
    logger.info(f"总评估因子数: {global_stats['total_factors_evaluated']}")
    
    # 各轮汇总
    logger.info("\n各轮表现汇总:")
    for stats in all_round_stats:
        round_num = stats["round_num"]
        status = stats.get("status", "unknown")
        
        if status == "success":
            val_mean = stats.get("val_score_mean", 0.0)
            val_max = stats.get("val_score_max", 0.0)
            logger.info(
                f"  Round {round_num}: val_score_mean={val_mean:.4f}, "
                f"val_score_max={val_max:.4f}"
            )
        else:
            error = stats.get("error", "unknown")
            logger.info(f"  Round {round_num}: ❌ {error}")
    
    logger.info("\n结果文件:")
    logger.info(f"  - 聚合结果: {CONFIG['RESULTS_DIR']}/all_rounds_factors.csv")
    logger.info(f"  - 各轮结果: {CONFIG['RESULTS_DIR']}/round_N_factor_metrics.csv")
    logger.info(f"  - 负样本: {CONFIG['NEGATIVE_SAMPLES_DIR']}/")
    
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())