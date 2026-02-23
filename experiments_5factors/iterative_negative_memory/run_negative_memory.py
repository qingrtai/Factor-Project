#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_negative_memory.py — Iterative Negative Memory Experiment Main Script

主控脚本职责：
1. 环境检查和日志初始化
2. Round 0 准备（基于 baseline 重评估）
3. 执行多轮迭代（调用 iterator）
4. 生成最终分析报告
5. 保存实验摘要

设计原则：
- 简洁：只负责主流程控制
- 复用：所有核心功能交给 iterator 和 memory_manager
- 完整：提供充分的日志和错误处理
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# ========== 路径设置 ========== #
parent_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========== 模块导入 ========== #
from .config import (
    CONFIG,
    validate_config,
    print_config_summary,
    RESULTS_DIR,
    LOGS_DIR,
    MAX_ROUNDS,
    FACTORS_PER_ROUND,
    NEGATIVE_SAMPLES_COUNT,
    BASELINE_FILE,
)

from .iterator import FactorIterator
from .memory_manager import MemoryManager


# ===================== 日志设置 =====================

def setup_logging() -> logging.Logger:
    """
    初始化日志系统
    
    策略：
    - 控制台 + 文件双输出
    - 清理旧的 handlers 避免重复
    - 捕获所有 warnings
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 清理已有 handlers
    for name in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(name).handlers.clear()
    logging.getLogger().handlers.clear()
    
    # 日志格式
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    
    # 文件输出
    log_file = os.path.join(
        LOGS_DIR,
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    
    # 捕获 warnings
    logging.captureWarnings(True)
    
    # 返回主 logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


# ===================== 环境检查 =====================

def check_environment(logger: logging.Logger) -> bool:
    """
    环境检查
    
    检查项：
    1. Python 版本
    2. 必要的包
    3. 配置文件
    4. API key（可选）
    """
    try:
        # Python 版本
        if sys.version_info < (3, 8):
            logger.error("Python 版本需要 >= 3.8")
            return False
        
        # 必要的包
        required_packages = ['pandas', 'numpy']
        missing = []
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            logger.error(f"缺少必要的包: {missing}")
            logger.error(f"请运行: pip install {' '.join(missing)}")
            return False
        
        # 配置验证
        try:
            validate_config()
            logger.info("✓ 配置验证通过")
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
        
        # API key 检查（可选）
        api_keys = [
            "SCHOOL_API_KEY",
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "GPT_API_KEY",
        ]
        
        has_key = any(os.getenv(k) for k in api_keys)
        
        if has_key:
            logger.info("✓ 检测到 API key")
            for k in api_keys:
                status = "SET" if os.getenv(k) else "NOT SET"
                logger.info(f"  {k}: {status}")
        else:
            logger.warning("⚠ 未检测到 API key，如需调用 GPT 请在环境变量中设置")
        
        logger.info("✓ 环境检查通过")
        return True
        
    except Exception as e:
        logger.error(f"环境检查失败: {e}")
        logger.error(traceback.format_exc())
        return False


# 删除第 42 行的 DATA_FILE 导入（同上）

# 修改 show_experiment_config() 函数
def show_experiment_config(logger: logging.Logger) -> None:
    """显示实验配置摘要"""
    # 临时读取全局配置（仅用于显示）
    from shared.config_loader import load_global_config
    g = load_global_config()
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 80)
    
    logger.info(f"实验名称: Iterative Negative Memory")
    logger.info(f"数据文件: {g['data']['raw_file']}")  # ← 从全局配置读取
    logger.info(f"Baseline: {BASELINE_FILE}")
    logger.info(f"结果目录: {RESULTS_DIR}")
    logger.info(f"日志目录: {LOGS_DIR}")
    logger.info(f"")
    logger.info(f"迭代轮数: {MAX_ROUNDS}")
    logger.info(f"每轮因子数: {FACTORS_PER_ROUND}")
    logger.info(f"负样本数: {NEGATIVE_SAMPLES_COUNT}")
    logger.info(f"")
    logger.info(f"评分公式: val_score = (sharpe + ann_ret + D) / 3")
    logger.info(f"D 定义: D = 1 - max_dd")
    logger.info(f"时间切分:")
    logger.info(f"  - Train: {g['years']['train'][0]}-{g['years']['train'][1]}")  # ← 从全局配置读取
    logger.info(f"  - Val:   {g['years']['val'][0]}-{g['years']['val'][1]}")      # ← 从全局配置读取
    logger.info(f"  - Test:  {g['years']['test'][0]}-{g['years']['test'][1]}")    # ← 从全局配置读取
    
    logger.info("=" * 80 + "\n")

# ===================== 迭代执行 =====================

def run_iterations(
    iterator: FactorIterator,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    执行多轮迭代
    
    Args:
        iterator: 迭代器实例
        logger: 日志对象
    
    Returns:
        每轮结果列表
    """
    results = []
    
    for round_num in range(1, MAX_ROUNDS + 1):
        round_start = time.time()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"ROUND {round_num} / {MAX_ROUNDS}")
        logger.info("=" * 80)
        
        try:
            # 执行单轮
            round_result = iterator.execute_round(round_num)
            
            # 检查结果
            if round_result.get("status") == "failed":
                logger.error(f"✗ Round {round_num} 失败")
                results.append(round_result)
                
                # 第一轮失败则中止
                if round_num == 1:
                    logger.error("第一轮失败，停止实验")
                    break
                else:
                    logger.warning("继续下一轮...")
                    continue
            
            # 成功
            results.append(round_result)
            
            logger.info(f"✓ Round {round_num} 完成")
            logger.info(f"  耗时: {time.time() - round_start:.2f}s")
            
            # 轮次间隔（避免 API 限流）
            if round_num < MAX_ROUNDS:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"✗ Round {round_num} 异常: {e}")
            logger.error(traceback.format_exc())
            
            results.append({
                "round_num": round_num,
                "status": "failed",
                "error": str(e),
                "duration": time.time() - round_start,
                "factors_generated": 0,
                "factors_evaluated": 0,
            })
            
            # 第一轮失败则中止
            if round_num == 1:
                logger.error("第一轮失败，停止实验")
                break
            else:
                logger.warning("继续下一轮...")
                continue
    
    return results


# ===================== 最终分析 =====================

def generate_round_summary_mean(logger: logging.Logger) -> Optional[str]:
    """
    生成每轮均值汇总
    
    输出: round_summary_mean.csv
    """
    logger.info("生成 round_summary_mean.csv...")
    
    try:
        rows = []
        
        # ========== Round 0: 直接读 baseline ========== #
        if os.path.exists(BASELINE_FILE):  # ← 改这里
            try:
                df = pd.read_csv(BASELINE_FILE)
                
                # 计算均值
                metrics = {"round": 0}
                
                metric_cols = [
                    "train_score",
                    "val_score", "val_coverage", "val_ann_ret", "val_sharpe",
                    "val_max_dd", "val_D", "val_diversity", "val_autocorr", "val_skew"
                ]
                
                for col in metric_cols:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        metrics[col] = float(values.mean()) if len(values) > 0 else np.nan
                    else:
                        metrics[col] = np.nan
                
                rows.append(metrics)
            except Exception as e:
                logger.warning(f"读取 baseline 失败: {e}")
        
        # ========== Round 1+ ========== #
        for round_num in range(1, MAX_ROUNDS + 1):
            csv_path = os.path.join(RESULTS_DIR, f"round_{round_num}_factor_metrics.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"读取 Round {round_num} 失败: {e}")
                continue
            
            # 计算均值
            metrics = {"round": round_num}
            
            for col in metric_cols:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce").dropna()
                    metrics[col] = float(values.mean()) if len(values) > 0 else np.nan
                else:
                    metrics[col] = np.nan
            
            rows.append(metrics)
        
        if not rows:
            logger.warning("没有轮次数据可汇总")
            return None
        
        # 保存
        summary_df = pd.DataFrame(rows).sort_values("round")
        output_path = os.path.join(RESULTS_DIR, "round_summary_mean.csv")
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ 保存 round_summary_mean.csv: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"生成 round_summary_mean.csv 失败: {e}")
        logger.error(traceback.format_exc())
        return None


def generate_final_analysis(
    results: List[Dict[str, Any]],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    生成最终分析报告
    
    包括：
    - 实验统计
    - 最优因子
    - Top 20 因子
    """
    logger.info("生成最终分析报告...")
    
    summary = {
        "rounds_attempted": len(results),
        "rounds_failed": sum(1 for r in results if r.get("status") == "failed"),
        "total_factors_generated": sum(r.get("factors_generated", 0) for r in results),
        "total_factors_evaluated": sum(r.get("factors_evaluated", 0) for r in results),
        "best_factor": None,
        "top_csv": None,
    }
    
    # 读取 all_rounds_factors.csv
    agg_path = os.path.join(RESULTS_DIR, "all_rounds_factors.csv")
    
    if not os.path.exists(agg_path):
        logger.warning(f"未找到聚合文件: {agg_path}")
        return summary
    
    try:
        df = pd.read_csv(agg_path)
        logger.info(f"加载聚合文件: {agg_path} ({len(df)} 行)")
    except Exception as e:
        logger.error(f"读取聚合文件失败: {e}")
        return summary
    
    # 数值化
    for col in ["val_score", "train_score", "val_sharpe", "val_ann_ret", "val_D"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 统计
    summary["total_factors"] = len(df)
    summary["unique_codes"] = int(df["code"].nunique()) if "code" in df.columns else len(df)
    
    # 最优因子
    if "val_score" in df.columns:
        top_df = df.sort_values("val_score", ascending=False).head(20)
        
        if not top_df.empty:
            best = top_df.iloc[0]
            summary["best_factor"] = {
                "factor_id": str(best.get("factor_id", "")),
                "val_score": float(best.get("val_score", np.nan)),
                "train_score": float(best.get("train_score", np.nan)),
                "val_sharpe": float(best.get("val_sharpe", np.nan)),
                "val_D": float(best.get("val_D", np.nan)),
            }
            
            # 保存 Top 20
            output_cols = [
                c for c in [
                    "round", "factor_id", "code",
                    "val_score", "train_score",
                    "val_sharpe", "val_ann_ret", "val_D", "val_max_dd",
                    "val_diversity", "val_coverage"
                ]
                if c in top_df.columns
            ]
            
            top_csv = os.path.join(RESULTS_DIR, "final_top_factors.csv")
            top_df[output_cols].to_csv(top_csv, index=False)
            summary["top_csv"] = top_csv
            
            logger.info(f"✓ 保存 Top 20 因子: {top_csv}")
    
    return summary


def save_experiment_summary(
    results: List[Dict[str, Any]],
    final_analysis: Dict[str, Any],
    duration: float,
    logger: logging.Logger
) -> None:
    """
    保存实验摘要 JSON
    """
    summary = {
        "experiment": "iterative_negative_memory",
        "timestamp": datetime.now().isoformat(),
        "duration_sec": duration,
        "config": {
            "max_rounds": MAX_ROUNDS,
            "factors_per_round": FACTORS_PER_ROUND,
            "negative_samples_count": NEGATIVE_SAMPLES_COUNT,
        },
        "results": results,
        "final_analysis": final_analysis,
    }
    
    output_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 保存实验摘要: {output_path}")
        
    except Exception as e:
        logger.error(f"保存实验摘要失败: {e}")


def print_final_summary(
    results: List[Dict[str, Any]],
    final_analysis: Dict[str, Any],
    duration: float,
    logger: logging.Logger
) -> None:
    """
    打印最终摘要
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"总耗时: {duration:.2f}s ({duration/60:.1f} min)")
    logger.info(f"完成轮数: {len(results)} / {MAX_ROUNDS}")
    logger.info(f"失败轮数: {final_analysis.get('rounds_failed', 0)}")
    logger.info(f"")
    logger.info(f"总生成因子数: {final_analysis.get('total_factors_generated', 0)}")
    logger.info(f"总评估因子数: {final_analysis.get('total_factors_evaluated', 0)}")
    logger.info(f"唯一代码数: {final_analysis.get('unique_codes', 0)}")
    
    # 最优因子
    best = final_analysis.get("best_factor")
    if best:
        logger.info(f"")
        logger.info(f"最优因子:")
        logger.info(f"  ID: {best.get('factor_id', 'N/A')}")
        logger.info(f"  Val Score: {best.get('val_score', np.nan):.4f}")
        logger.info(f"  Train Score: {best.get('train_score', np.nan):.4f}")
        logger.info(f"  Val Sharpe: {best.get('val_sharpe', np.nan):.4f}")
        logger.info(f"  Val D: {best.get('val_D', np.nan):.4f}")
    
    # 输出文件
    logger.info(f"")
    logger.info(f"输出文件:")
    logger.info(f"  - {RESULTS_DIR}/all_rounds_factors.csv")
    logger.info(f"  - {RESULTS_DIR}/round_summary_mean.csv")
    if final_analysis.get("top_csv"):
        logger.info(f"  - {final_analysis['top_csv']}")
    logger.info(f"  - {RESULTS_DIR}/experiment_summary.json")
    
    logger.info("=" * 80 + "\n")


# ===================== 主函数 =====================

def main():
    """
    主函数
    
    流程：
    1. 环境检查
    2. 初始化组件
    3. 执行迭代
    4. 生成报告
    5. 保存摘要
    """
    exp_start = time.time()
    
    print("\n" + "=" * 80)
    print("ITERATIVE NEGATIVE MEMORY EXPERIMENT")
    print("=" * 80 + "\n")
    
    # ========== Step 1: 初始化 ========== #
    logger = setup_logging()
    
    logger.info("步骤 1/5: 环境检查")
    if not check_environment(logger):
        logger.error("环境检查失败，退出")
        sys.exit(1)
    
    # ========== Step 2: 显示配置 ========== #
    logger.info("步骤 2/5: 显示配置")
    show_experiment_config(logger)
    
    # ========== Step 3: 初始化组件 ========== #
    logger.info("步骤 3/5: 初始化组件")
    
    try:
        iterator = FactorIterator()
        memory_manager = MemoryManager()
        logger.info("✓ 组件初始化完成")
    except Exception as e:
        logger.error(f"✗ 组件初始化失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # ========== Step 4: 执行迭代 ========== #
    logger.info("步骤 4/5: 执行迭代")
    
    results = run_iterations(iterator, logger)
    
    logger.info(f"✓ 迭代完成，共 {len(results)} 轮")
    
    # ========== Step 5: 生成报告 ========== #
    logger.info("步骤 5/5: 生成最终报告")
    
    # 每轮均值
    generate_round_summary_mean(logger)
    
    # 最终分析
    final_analysis = generate_final_analysis(results, logger)
    
    # 保存摘要
    duration = time.time() - exp_start
    save_experiment_summary(results, final_analysis, duration, logger)
    
    # 打印摘要
    print_final_summary(results, final_analysis, duration, logger)
    
    logger.info("✓ 实验完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)