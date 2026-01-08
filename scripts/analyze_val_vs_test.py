#!/usr/bin/env python3
"""
验证集 vs 测试集 详细对比分析
诊断为什么 factor_baseline 在测试集上表现好
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_experiment(exp_name: str):
    """分析单个实验"""
    print(f"\n{'='*80}")
    print(f"实验: {exp_name}")
    print(f"{'='*80}")
    
    # 读取验证集结果
    if exp_name == 'factor_baseline':
        val_path = Path(f'results/{exp_name}/baseline_factor_metrics.csv')
    else:
        val_path = Path(f'results/{exp_name}/all_rounds_factors.csv')
    
    test_path = Path(f'results/test_evaluation/{exp_name}_test_detailed.csv')
    
    if not val_path.exists():
        print(f"❌ 找不到验证集文件: {val_path}")
        return
    
    if not test_path.exists():
        print(f"❌ 找不到测试集文件: {test_path}")
        return
    
    # 读取数据（尝试多种编码）
    for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
        try:
            val_df = pd.read_csv(val_path, encoding=encoding)
            break
        except:
            continue
    
    for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
        try:
            test_df = pd.read_csv(test_path, encoding=encoding)
            break
        except:
            continue
    
    # 统计
    print(f"\n【整体统计】")
    print(f"验证集平均 val_score: {val_df['val_score'].mean():.4f}")
    print(f"测试集平均 test_score: {test_df[test_df['status']=='success']['test_score'].mean():.4f}")
    print(f"测试集成功率: {(test_df['status']=='success').mean():.1%}")
    
    # 检查异常
    success_df = test_df[test_df['status'] == 'success']
    
    print(f"\n【测试集异常检查】")
    high_sharpe = (success_df['test_sharpe'] > 3.0).sum()
    print(f"Sharpe > 3.0 的因子数: {high_sharpe}/{len(success_df)}")
    
    if 'test_D' in success_df.columns:
        neutral_D = (success_df['test_D'] == 0.5).sum()
        print(f"D = 0.5 的因子数: {neutral_D}/{len(success_df)}")
        print(f"D < 0.2 的因子数: {(success_df['test_D'] < 0.2).sum()}/{len(success_df)}")
    
    # 失败因子
    failed = test_df[test_df['status'] != 'success']
    if len(failed) > 0:
        print(f"\n【失败因子】")
        print(f"失败数: {len(failed)}/10")
        for idx, row in failed.iterrows():
            print(f"  因子{row['factor_id']}: {row['status']}")
    
    # 逐因子对比
    print(f"\n【逐因子对比】(成功因子)")
    print(f"{'ID':<4} {'val_score':>10} {'test_score':>10} {'变化':>10} {'test_D':>8} {'test_sh':>10}")
    print("-" * 66)
    
    for i in range(min(10, len(test_df))):
        test_row = test_df.iloc[i]
        
        if test_row['status'] != 'success':
            print(f"{i+1:<4} {'N/A':>10} {'FAILED':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10}")
            continue
        
        # 找对应的验证集因子
        if exp_name == 'factor_baseline':
            val_row = val_df.iloc[i]
        else:
            # 其他实验需要匹配 round
            val_row = val_df.iloc[i] if i < len(val_df) else None
        
        if val_row is not None:
            val_score = val_row['val_score']
            test_score = test_row['test_score']
            change = test_score - val_score
            test_D = test_row.get('test_D', np.nan)
            test_sh = test_row.get('test_sharpe', np.nan)
            
            marker = "⚠️" if test_sh > 3.0 else ""
            print(f"{i+1:<4} {val_score:>10.4f} {test_score:>10.4f} {change:>10.4f} {test_D:>8.4f} {test_sh:>10.4f} {marker}")
    
    # 相关性分析
    if exp_name == 'factor_baseline':
        success_indices = test_df[test_df['status'] == 'success'].index
        val_scores = [val_df.iloc[i]['val_score'] for i in success_indices]
        test_scores = [test_df.iloc[i]['test_score'] for i in success_indices]
        
        if len(val_scores) >= 3:
            corr = np.corrcoef(val_scores, test_scores)[0, 1]
            print(f"\n【相关性】")
            print(f"验证集 vs 测试集分数相关性: {corr:.4f}")
            if abs(corr) < 0.3:
                print("⚠️  相关性很低，说明验证集和测试集表现完全不一致！")


def main():
    print("=" * 80)
    print("验证集 vs 测试集 全面对比分析")
    print("=" * 80)
    
    experiments = [
        'factor_baseline',
        'iterative_baseline',
        'iterative_global_memory',
        'iterative_negative_memory',
        'iterative_baseline_with_reports',
        'iterative_negative_memory_with_reports'
    ]
    
    for exp in experiments:
        analyze_experiment(exp)
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()