#!/usr/bin/env python3
"""
测试集报告生成脚本
功能：
  1. 读取6个实验的详细测试结果
  2. 生成汇总统计表（mean, min, max）
  3. 生成对比表格（精简版）
  4. 生成箱线图（test_score 对比）

运行：python scripts/generate_test_report.py
"""
# ========== 重要：必须先设置路径，再导入 core 模块 ==========
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ==========================================================

# 现在可以安全导入所有模块了
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils import setup_logger

# 设置日志
logger = setup_logger("ReportGenerator", level="INFO")

# 实验列表（按显示顺序）
EXPERIMENTS = [
    'factor_baseline',
    'iterative_baseline',
    'iterative_global_memory',
    'iterative_negative_memory',
    'iterative_baseline_with_reports',
    'iterative_negative_memory_with_reports'
]

# 实验显示名称（用于图表）
EXPERIMENT_LABELS = {
    'factor_baseline': 'Baseline',
    'iterative_baseline': 'Iterative',
    'iterative_global_memory': 'Global Memory',
    'iterative_negative_memory': 'Negative Memory',
    'iterative_baseline_with_reports': 'Iterative + Reports',
    'iterative_negative_memory_with_reports': 'Negative Memory + Reports'
}

# 需要统计的指标
METRICS = [
    'test_score',
    'test_coverage',
    'test_ann_ret',
    'test_sharpe',
    'test_max_dd',
    'test_D',
    'test_diversity',
    'test_autocorr',
    'test_skew'
]


def load_test_results(input_dir: Path) -> dict:
    """加载所有实验的测试结果"""
    logger.info("📂 加载测试结果...")
    
    results = {}
    for experiment in EXPERIMENTS:
        csv_path = input_dir / f"{experiment}_test_detailed.csv"
        
        if not csv_path.exists():
            logger.warning(f"⚠️  找不到文件: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            # 只保留成功的因子
            df_success = df[df['status'] == 'success'].copy()
            results[experiment] = df_success
            logger.info(f"  ✓ {experiment}: {len(df_success)}/{len(df)} 个成功因子")
            
        except Exception as e:
            logger.error(f" 读取 {experiment} 失败: {e}")
    
    return results


def generate_summary_statistics(results: dict, output_dir: Path):
    """生成汇总统计表"""
    logger.info("\n 生成汇总统计表...")
    
    summary_rows = []
    
    for experiment in EXPERIMENTS:
        if experiment not in results or results[experiment].empty:
            logger.warning(f"  跳过 {experiment}（无数据）")
            continue
        
        df = results[experiment]
        row = {'experiment': experiment}
        
        # 计算每个指标的 mean, min, max
        for metric in METRICS:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_min'] = values.min()
                    row[f'{metric}_max'] = values.max()
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan
        
        # 计算成功率
        row['success_rate'] = len(df) / 10.0  # 假设每个实验有10个因子
        row['n_success_factors'] = len(df)
        
        summary_rows.append(row)
    
    # 转换为 DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # 保存
    output_path = output_dir / "test_summary_statistics.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info(f"  ✓ 汇总统计已保存: {output_path}")
    
    return summary_df


def generate_comparison_table(results: dict, output_dir: Path):
    """生成对比表格（精简版）"""
    logger.info("\n 生成对比表格...")
    
    comparison_rows = []
    
    # 核心指标列表
    core_metrics = ['test_score', 'test_sharpe', 'test_ann_ret', 'test_D', 'test_coverage']
    
    for experiment in EXPERIMENTS:
        if experiment not in results or results[experiment].empty:
            continue
        
        df = results[experiment]
        row = {'experiment': experiment}
        
        # 计算核心指标的均值
        for metric in core_metrics:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors='coerce').dropna()
                row[metric] = values.mean() if len(values) > 0 else np.nan
        
        row['n_factors'] = len(df)
        
        comparison_rows.append(row)
    
    # 转换为 DataFrame
    comparison_df = pd.DataFrame(comparison_rows)
    
    # 保存
    output_path = output_dir / "test_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"  ✓ 对比表格已保存: {output_path}")
    
    # 打印到控制台
    logger.info("\n" + "="*80)
    logger.info(" 测试集对比表格（核心指标均值）")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    logger.info("="*80 + "\n")
    
    return comparison_df


def generate_boxplot(results: dict, output_dir: Path):
    """生成 test_score 箱线图"""
    logger.info("\n 生成箱线图...")
    
    # 准备数据
    plot_data = []
    for experiment in EXPERIMENTS:
        if experiment not in results or results[experiment].empty:
            continue
        
        df = results[experiment]
        scores = pd.to_numeric(df['test_score'], errors='coerce').dropna()
        
        for score in scores:
            plot_data.append({
                'experiment': EXPERIMENT_LABELS.get(experiment, experiment),
                'test_score': score
            })
    
    if not plot_data:
        logger.warning("  没有数据，跳过箱线图生成")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # 绘图
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # 箱线图
    ax = sns.boxplot(
        data=plot_df,
        x='experiment',
        y='test_score',
        palette='Set2',
        width=0.6
    )
    
    # 叠加散点（显示每个因子）
    sns.stripplot(
        data=plot_df,
        x='experiment',
        y='test_score',
        color='black',
        alpha=0.3,
        size=4,
        ax=ax
    )
    
    # 添加均值线
    means = plot_df.groupby('experiment')['test_score'].mean()
    x_positions = range(len(means))
    ax.hlines(
        y=means.values,
        xmin=[x - 0.4 for x in x_positions],
        xmax=[x + 0.4 for x in x_positions],
        colors='red',
        linestyles='--',
        linewidth=2,
        label='Mean'
    )
    
    # 设置标题和标签
    ax.set_title('Test Score Comparison Across Experiments', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Score', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "test_score_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ 箱线图已保存: {output_path}")
    
    plt.close()


def generate_additional_metrics_plot(results: dict, output_dir: Path):
    """生成其他关键指标的对比图（可选）"""
    logger.info("\n 生成多指标对比图...")
    
    # 准备数据
    metrics_to_plot = ['test_sharpe', 'test_ann_ret', 'test_D']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics_to_plot):
        plot_data = []
        
        for experiment in EXPERIMENTS:
            if experiment not in results or results[experiment].empty:
                continue
            
            df = results[experiment]
            values = pd.to_numeric(df[metric], errors='coerce').dropna()
            
            for value in values:
                plot_data.append({
                    'experiment': EXPERIMENT_LABELS.get(experiment, experiment),
                    metric: value
                })
        
        if not plot_data:
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        # 绘制箱线图
        ax = axes[idx]
        sns.boxplot(
            data=plot_df,
            x='experiment',
            y=metric,
            palette='Set2',
            ax=ax
        )
        
        ax.set_title(metric.replace('test_', '').replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "test_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ 多指标对比图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    logger.info("\n" + "="*80)
    logger.info(" 开始生成测试集报告")
    logger.info("="*80)
    
    # 输入输出目录
    input_dir = Path("results/test_evaluation")
    output_dir = Path("results/test_evaluation")
    
    if not input_dir.exists():
        logger.error(f" 输入目录不存在: {input_dir}")
        logger.error(" 请先运行: python scripts/evaluate_test_set.py")
        return
    
    # 1. 加载测试结果
    results = load_test_results(input_dir)
    
    if not results:
        logger.error(" 没有找到任何测试结果")
        return
    
    # 2. 生成汇总统计
    summary_df = generate_summary_statistics(results, output_dir)
    
    # 3. 生成对比表格
    comparison_df = generate_comparison_table(results, output_dir)
    
    # 4. 生成箱线图
    generate_boxplot(results, output_dir)
    
    # 5. 生成多指标对比图（可选）
    generate_additional_metrics_plot(results, output_dir)
    
    # 6. 完成
    logger.info("\n" + "="*80)
    logger.info(" 报告生成完成！")
    logger.info("="*80)
    logger.info(f"\n 所有输出文件保存在: {output_dir}")
    logger.info("\n生成的文件：")
    logger.info("  1. test_summary_statistics.csv    - 详细汇总统计（mean, min, max）")
    logger.info("  2. test_comparison.csv             - 精简对比表格")
    logger.info("  3. test_score_comparison.png       - test_score 箱线图")
    logger.info("  4. test_metrics_comparison.png     - 其他指标对比图")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()