#!/usr/bin/env python3
"""
诊断脚本：检查导入是否正常
运行：python scripts/test_imports.py
"""

print("=" * 80)
print("开始测试导入...")
print("=" * 80)

# 测试1：路径设置
print("\n[1/5] 测试路径设置...")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
print(f"  ✓ 项目根目录: {PROJECT_ROOT}")
print(f"  ✓ Scripts目录: {SCRIPTS_DIR}")

# 测试2：导入 core 模块
print("\n[2/5] 测试导入 core 模块...")
try:
    from core.data_loader import load_splits
    from core.utils import setup_logger
    print("  ✓ core 模块导入成功")
except Exception as e:
    print(f"  ❌ core 模块导入失败: {e}")
    sys.exit(1)

# 测试3：导入 test_evaluator_clean
print("\n[3/5] 测试导入 test_evaluator_clean...")
try:
    from test_evaluator_clean import evaluate_on_test_holdout
    print("  ✓ test_evaluator_clean 导入成功")
except Exception as e:
    print(f"  ❌ test_evaluator_clean 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4：读取配置文件
print("\n[4/5] 测试读取配置文件...")
try:
    import yaml
    with open("configs/best_iterations.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"  ✓ 配置文件读取成功，找到 {len(config.get('experiments', {}))} 个实验")
except Exception as e:
    print(f"  ❌ 配置文件读取失败: {e}")
    sys.exit(1)

# 测试5：测试 logger
print("\n[5/5] 测试日志系统...")
try:
    logger = setup_logger("TestLogger", level="INFO")
    logger.info("日志系统测试")
    print("  ✓ 日志系统正常")
except Exception as e:
    print(f"  ❌ 日志系统失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！环境正常，可以运行 evaluate_test_set.py")
print("=" * 80)