#!/usr/bin/env python3
"""
验证大数据集
"""

import pandas as pd
import yaml

# 读取配置
with open('configs/global.yaml') as f:
    cfg = yaml.safe_load(f)

# 读取数据（分块读取，避免内存溢出）
print("读取数据...")
df = pd.read_csv(
    cfg['data']['raw_file'],
    parse_dates=[cfg['schema']['date_col']]
)

print("=" * 80)
print("数据基本信息")
print("=" * 80)

# 时间范围
date_col = cfg['schema']['date_col']
print(f"\n日期范围: {df[date_col].min()} 到 {df[date_col].max()}")
print(f"时间跨度: {(df[date_col].max() - df[date_col].min()).days / 365.25:.1f} 年")

# 时间点数
unique_dates = df[date_col].nunique()
print(f"唯一日期数: {unique_dates}")

# 股票数量
id_col = cfg['schema']['id_col']
print(f"股票数量: {df[id_col].nunique()}")

# 数据量
print(f"总行数: {len(df):,}")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# 按时期划分
train_end = cfg['years']['train_end']
val_end = cfg['years']['val_end']
test_end = cfg['years']['test_end']

df['year'] = df[date_col].dt.year

train_df = df[df['year'] <= train_end]
val_df = df[(df['year'] > train_end) & (df['year'] <= val_end)]
test_df = df[(df['year'] > val_end) & (df['year'] <= test_end)]

print("\n" + "=" * 80)
print("数据划分")
print("=" * 80)
print(f"\n训练集 ({df[date_col].min().year}-{train_end}):")
print(f"  行数: {len(train_df):,}")
print(f"  唯一日期: {train_df[date_col].nunique()}")

print(f"\n验证集 ({train_end+1}-{val_end}):")
print(f"  行数: {len(val_df):,}")
print(f"  唯一日期: {val_df[date_col].nunique()}")

print(f"\n测试集 ({val_end+1}-{test_end}):")
print(f"  行数: {len(test_df):,}")
print(f"  唯一日期: {test_df[date_col].nunique()}")

# 缺失值检查
print("\n" + "=" * 80)
print("数据质量")
print("=" * 80)

ret_col = cfg['schema']['ret_col']
print(f"\n收益率缺失: {df[ret_col].isnull().sum() / len(df):.1%}")

# 检查关键财务字段
key_fields = ['atq', 'revtq', 'ibq', 'niq', 'cogsq']
existing_fields = [f for f in key_fields if f in df.columns]

if existing_fields:
    print("\n关键字段缺失率:")
    for field in existing_fields:
        missing_rate = df[field].isnull().sum() / len(df)
        print(f"  {field}: {missing_rate:.1%}")
else:
    print("\n⚠️  未找到标准财务字段，请检查列名")
    print(f"   当前列名: {list(df.columns)[:20]}...")

print("\n" + "=" * 80)
print("验证完成！")
print("=" * 80)