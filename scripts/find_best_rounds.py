#!/usr/bin/env python3
"""
辅助工具：查找每个实验的最佳轮次
用途：根据验证集 val_score，自动找出每个实验表现最好的轮次
运行：python scripts/find_best_rounds.py
"""

import pandas as pd
from pathlib import Path

# 实验列表（factor_baseline 只有一轮，固定为0，不需要查找）
EXPERIMENTS = [
    'iterative_baseline',
    'iterative_global_memory', 
    'iterative_negative_memory',
    'iterative_baseline_with_reports',
    'iterative_negative_memory_with_reports'
]

def find_best_rounds():
    """查找每个实验的最佳轮次"""
    
    print("=" * 80)
    print("查找最佳轮次（基于验证集 val_score 均值）")
    print("=" * 80)
    print()
    
    results = {}
    
    for exp in EXPERIMENTS:
        csv_path = Path(f'results/{exp}/all_rounds_factors.csv')
        
        if not csv_path.exists():
            print(f"⚠️  {exp:45s} → 文件不存在: {csv_path}")
            continue
        
        try:
            # 尝试多种编码格式读取 CSV
            df = None
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                print(f"❌ {exp:45s} → 错误: 无法用任何编码读取文件")
                continue
            
            # 计算每轮的平均 val_score
            round_scores = df.groupby('round')['val_score'].mean()
            best_round = round_scores.idxmax()
            best_score = round_scores.max()
            
            results[exp] = {
                'best_round': int(best_round),
                'best_score': float(best_score),
                'total_rounds': int(df['round'].max() + 1)
            }
            
            print(f"✓ {exp:45s} → Round {best_round}  (val_score: {best_score:.4f})")
            
        except Exception as e:
            print(f"❌ {exp:45s} → 错误: {e}")
    
    print()
    print("=" * 80)
    print("建议的 best_iterations.yaml 配置：")
    print("=" * 80)
    print()
    print("experiments:")
    print("  factor_baseline: 0  # 基线实验固定为第0轮")
    
    for exp in EXPERIMENTS:
        if exp in results:
            best_round = results[exp]['best_round']
            print(f"  {exp}: {best_round}")
    
    print()
    print("=" * 80)
    print("请将上述配置复制到 configs/best_iterations.yaml")
    print("=" * 80)
    print()
    
    return results


if __name__ == "__main__":
    find_best_rounds()