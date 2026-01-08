#!/usr/bin/env python3
"""
测试集评估主脚本
功能：
  1. 读取 best_iterations.yaml 配置，获取每个实验的最佳轮次
  2. 从各实验的 all_rounds_factors.csv 中提取最佳轮次的因子
  3. 在测试集上重新评估这些因子
  4. 输出详细结果到 results/test_evaluation/{experiment}_test_detailed.csv

运行：python scripts/evaluate_test_set.py
"""

# ========== 重要：必须先设置路径，再导入 core 模块 ==========
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（确保能找到 core 模块）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 同时添加 scripts 目录（用于导入 test_evaluator_clean）
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
# ==========================================================

# 现在可以安全导入所有模块了
import yaml
import pandas as pd
import numpy as np
from core.data_loader import load_splits
from core.utils import setup_logger

# 导入无数据泄露的评估函数
from test_evaluator_clean import evaluate_on_test_holdout

# 设置日志
logger = setup_logger("TestEvaluator", level="INFO")

# 实验列表
EXPERIMENTS = [
    'factor_baseline',
    'iterative_baseline',
    'iterative_global_memory',
    'iterative_negative_memory',
    'iterative_baseline_with_reports',
    'iterative_negative_memory_with_reports'
]

def load_best_iterations(config_path: str = "configs/best_iterations.yaml") -> dict:
    """加载最佳轮次配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config['experiments']
    except Exception as e:
        logger.error(f"❌ 无法加载配置文件 {config_path}: {e}")
        sys.exit(1)


def load_test_data(global_config_path: str = "configs/global.yaml") -> dict:
    """加载测试集数据"""
    logger.info("📊 加载测试集数据...")
    
    try:
        with open(global_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        splits = load_splits(
            raw_csv=cfg['data']['raw_file'],
            date_col=cfg['schema']['date_col'],
            years=cfg['years'],
            id_col=cfg['schema']['id_col'],
            ret_col=cfg['schema']['ret_col']
        )
        
        logger.info(f"✓ 测试集加载成功：{len(splits['test'])} 行数据")
        return splits, cfg
        
    except Exception as e:
        logger.error(f"❌ 加载测试集失败: {e}")
        sys.exit(1)


def extract_best_round_factors(experiment: str, best_round: int) -> list:
    """从实验结果中提取最佳轮次的因子"""
    # factor_baseline 使用不同的文件名
    if experiment == 'factor_baseline':
        csv_path = Path(f"results/{experiment}/baseline_factor_metrics.csv")
    else:
        csv_path = Path(f"results/{experiment}/all_rounds_factors.csv")
    
    if not csv_path.exists():
        logger.error(f"❌ 找不到文件: {csv_path}")
        # 尝试查找该目录下的其他CSV文件
        exp_dir = Path(f"results/{experiment}")
        if exp_dir.exists():
            csv_files = list(exp_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"   💡 但在该目录下找到了这些CSV文件：")
                for f in csv_files:
                    logger.info(f"      - {f.name}")
            else:
                logger.info(f"   💡 该目录下没有任何CSV文件")
        else:
            logger.error(f"   💡 实验目录不存在: {exp_dir}")
        return []
    
    try:
        # 尝试多种编码格式读取 CSV
        df = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']
        last_error = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                logger.info(f"  ✓ 使用 {encoding} 编码成功读取文件")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                # 其他类型的错误（比如文件损坏）
                logger.error(f"❌ 读取文件时出错 (encoding={encoding}): {e}")
                return []
        
        if df is None:
            logger.error(f"❌ 无法用任何编码读取文件: {csv_path}")
            logger.error(f"   最后一次错误: {last_error}")
            return []
        
        # factor_baseline 可能没有 round 列，直接使用所有数据
        if experiment == 'factor_baseline':
            if 'round' in df.columns:
                best_df = df[df['round'] == best_round].copy()
            else:
                # 没有 round 列，直接使用所有数据
                logger.info(f"  ℹ️  {experiment} 没有 round 列，使用全部因子")
                best_df = df.copy()
        else:
            # 其他实验：筛选最佳轮次的因子
            best_df = df[df['round'] == best_round].copy()
        
        if best_df.empty:
            logger.warning(f"⚠️  {experiment} 的 round {best_round} 没有数据")
            return []
        
        # 转换为因子列表格式（batch_evaluate 需要的格式）
        factors = []
        
        # 检查必需的列是否存在
        if 'code' not in best_df.columns:
            logger.error(f"❌ 文件缺少 'code' 列。当前列名: {list(best_df.columns)}")
            return []
        
        for i, (_, row) in enumerate(best_df.iterrows(), start=1):
            factor_dict = {'code': row['code']}
            # 如果有 factor_id 列就保留，否则使用序号
            if 'factor_id' in best_df.columns and pd.notna(row.get('factor_id')):
                factor_dict['factor_id'] = row['factor_id']
            else:
                factor_dict['factor_id'] = i
            factors.append(factor_dict)
        
        logger.info(f"  ✓ 提取 {len(factors)} 个因子 (round {best_round})")
        return factors
        
    except Exception as e:
        logger.error(f"❌ 读取 {experiment} 失败: {e}")
        return []


def evaluate_experiment(experiment: str, best_round: int, splits: dict, cfg: dict, output_dir: Path):
    """评估单个实验（无数据泄露版本）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 评估实验: {experiment} (最佳轮次: {best_round})")
    logger.info(f"{'='*80}")
    
    # 1. 提取最佳轮次的因子
    factors = extract_best_round_factors(experiment, best_round)
    if not factors:
        logger.warning(f"⚠️  跳过 {experiment}（无有效因子）")
        return
    
    # 2. 在测试集上评估（使用验证集确定方向，无数据泄露）
    logger.info(f"  🧪 在测试集上评估 {len(factors)} 个因子...")
    logger.info(f"  📌 关键：因子方向在验证集上确定，测试集纯净评估")
    
    try:
        # 使用无数据泄露的评估函数
        results_df = evaluate_on_test_holdout(
            factors=factors,
            val_data=splits['val'],    # 验证集：用于确定因子方向
            test_data=splits['test'],  # 测试集：纯净评估
            ret_col=cfg['schema']['ret_col'],
            date_col=cfg['schema']['date_col'],
            periods_per_year=cfg['freq_per_year'],
            id_start=1
        )
        
        # 3. 添加实验信息列
        results_df['experiment_name'] = experiment
        results_df['best_round'] = best_round
        
        # 4. 保存结果
        output_path = output_dir / f"{experiment}_test_detailed.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"  ✓ 结果已保存: {output_path}")
        
        # 5. 显示简要统计
        success_rate = (results_df['status'] == 'success').mean()
        success_df = results_df[results_df['status'] == 'success']
        
        logger.info(f"  📈 成功率: {success_rate:.1%}")
        
        if len(success_df) > 0:
            try:
                scores = success_df['test_score'].values
                mean_score = np.nanmean(scores.astype(float))
                if np.isfinite(mean_score):
                    logger.info(f"  📈 平均 test_score: {mean_score:.4f}")
                else:
                    logger.info(f"  📈 平均 test_score: N/A")
            except Exception as e:
                logger.info(f"  📈 平均 test_score: N/A (计算失败: {e})")
        else:
            logger.info(f"  📈 平均 test_score: N/A (无成功因子)")
        
    except Exception as e:
        logger.error(f"❌ 评估 {experiment} 时出错: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        logger.error(f"❌ 评估 {experiment} 时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 过滤不重要的 numpy 警告（避免日志混乱）
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    # 添加调试输出
    print("=" * 80)
    print("脚本启动...")
    print("=" * 80)
    
    try:
        logger.info("\n" + "="*80)
        logger.info("🚀 开始测试集评估")
        logger.info("="*80)
    except Exception as e:
        print(f"日志初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 1. 创建输出目录
    output_dir = Path("results/test_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 输出目录: {output_dir}")
    
    # 2. 加载配置
    best_iterations = load_best_iterations()
    logger.info(f"\n✓ 加载最佳轮次配置:")
    for exp, round_num in best_iterations.items():
        logger.info(f"  - {exp}: round {round_num}")
    
    # 3. 加载测试集
    splits, cfg = load_test_data()
    
    # 4. 逐个评估实验
    for experiment in EXPERIMENTS:
        if experiment not in best_iterations:
            logger.warning(f"⚠️  配置中缺少 {experiment}，跳过")
            continue
        
        best_round = best_iterations[experiment]
        evaluate_experiment(experiment, best_round, splits, cfg, output_dir)
    
    # 5. 完成
    logger.info("\n" + "="*80)
    logger.info("✅ 测试集评估完成！")
    logger.info(f"📁 所有结果保存在: {output_dir}")
    logger.info("="*80)
    logger.info("\n💡 下一步：运行 python scripts/generate_test_report.py 生成汇总报告")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=" * 80)
        print("脚本执行时发生错误！")
        print("=" * 80)
        print(f"错误信息: {e}")
        print("\n完整错误栈:")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)