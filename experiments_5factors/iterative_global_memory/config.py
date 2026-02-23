# experiments/iterative_global_memory/config.py
"""
iterative_global_memory 实验配置
核心特点：累积所有历史轮次的记忆（与iterative_baseline的单轮记忆不同）
"""

from shared.paths import results_dir
from pathlib import Path

# =============================================================================
# 实验核心配置
# =============================================================================

# 迭代轮次
MAX_ROUNDS = 3                 # 最大迭代轮次
MIN_ROUNDS = 3                 # 最少运行轮次（强制跑满）
FACTORS_PER_ROUND = 10         # 每轮生成因子数

# =============================================================================
# 路径配置
# =============================================================================

RESULTS_DIR = results_dir("iterative_global_memory")
BASELINE_FILE = results_dir("factor_baseline") / "baseline_factor_metrics.csv"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 全局记忆配置（核心区别！）
# =============================================================================

# 记忆采样策略
GLOBAL_TOP_K = 8               # 每轮从历史中取top-k高分因子
GLOBAL_BOTTOM_K = 8            # 每轮从历史中取bottom-k低分因子（对比学习）
MAX_MEMORY_ITEMS = 2000        # 记忆条目上限（防止token溢出）

# 因子生成策略
REQUIRED_UNDERUSED_RATIO = 0.4  # 低频字段使用比例（鼓励探索新字段）
GENERATION_STRATEGY = "auto_mix"  # evolution | diversification | exploitation | auto_mix

# 记忆评分字段
MEMORY_SCORE_FIELD = "train_score"  # 用于记忆排序的字段
GPT_REFERENCE_SCORE = "val"         # GPT学习的分数类型（告诉GPT关注val_score）

# =============================================================================
# 早停配置
# =============================================================================

EARLY_STOPPING_PATIENCE = 2    # 允许连续2轮无提升（不像baseline设999禁用）
MIN_DELTA = 0.001              # 最小提升阈值
MIN_ROUNDS = 3                 # 最少运行轮数（与MAX_ROUNDS相同则强制跑满）
VAL_SCORE_THRESHOLD = None     # 绝对分数阈值（None表示不使用）

# 早停比较基准
ES_COMPARE_TO = "best"         # "best"（与历史最佳比）或"last"（与上一轮比）
ES_SCORE_TYPE = "val"          # 早停依据的分数类型："val"或"train"

# =============================================================================
# GPT生成配置
# =============================================================================

GPT_MODEL = "gpt-4o-2024-08-06"  # 使用的GPT模型
GPT_TEMPERATURE = 0.7            # 生成温度（0.7保证多样性）
GPT_MAX_TOKENS = 900             # 最大token数
GPT_MAX_RETRIES = 3              # 失败重试次数
GPT_RETRY_DELAY = 2.0            # 重试延迟（秒）

# ← 在这里添加下面这一行
MAX_REFILL_CALLS = 3             # 补货最大次数（评估不足时的重试）

# =============================================================================
# 导出CONFIG字典（供run_iterative.py使用）
# =============================================================================

CONFIG = {
    # 迭代配置
    "MAX_ROUNDS": MAX_ROUNDS,
    "MIN_ROUNDS": MIN_ROUNDS,
    "FACTORS_PER_ROUND": FACTORS_PER_ROUND,
    
    # 路径
    "RESULTS_DIR": RESULTS_DIR,
    "BASELINE_FILE": BASELINE_FILE,
    "LOGS_DIR": LOGS_DIR,
    
    # 全局记忆配置（核心特点）
    "GLOBAL_TOP_K": GLOBAL_TOP_K,
    "GLOBAL_BOTTOM_K": GLOBAL_BOTTOM_K,
    "MAX_MEMORY_ITEMS": MAX_MEMORY_ITEMS,
    "REQUIRED_UNDERUSED_RATIO": REQUIRED_UNDERUSED_RATIO,
    "GENERATION_STRATEGY": GENERATION_STRATEGY,
    "MEMORY_SCORE_FIELD": MEMORY_SCORE_FIELD,
    "GPT_REFERENCE_SCORE": GPT_REFERENCE_SCORE,
    
    # 早停配置
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "MIN_DELTA": MIN_DELTA,
    "VAL_SCORE_THRESHOLD": VAL_SCORE_THRESHOLD,
    "ES_COMPARE_TO": ES_COMPARE_TO,
    "ES_SCORE_TYPE": ES_SCORE_TYPE,
    
    # GPT配置
    "GPT_MODEL": GPT_MODEL,
    "GPT_TEMPERATURE": GPT_TEMPERATURE,
    "GPT_MAX_TOKENS": GPT_MAX_TOKENS,
    "GPT_MAX_RETRIES": GPT_MAX_RETRIES,
    "GPT_RETRY_DELAY": GPT_RETRY_DELAY,
    "MAX_REFILL_CALLS": MAX_REFILL_CALLS,  # ← 添加这一行
}

# =============================================================================
# 校验函数（保持简洁）
# =============================================================================

def validate_config():
    """校验配置是否合法"""
    errors = []
    warnings = []
    
    # 路径检查
    if not BASELINE_FILE.exists():
        errors.append(f"Baseline文件不存在: {BASELINE_FILE}")
    
    # 基本参数检查
    if MAX_ROUNDS <= 0:
        errors.append("MAX_ROUNDS必须大于0")
    if FACTORS_PER_ROUND <= 0:
        errors.append("FACTORS_PER_ROUND必须大于0")
    if MIN_ROUNDS > MAX_ROUNDS:
        warnings.append("MIN_ROUNDS > MAX_ROUNDS，将自动调整")
    
    # 全局记忆参数检查
    if GLOBAL_TOP_K < 0 or GLOBAL_BOTTOM_K < 0:
        errors.append("GLOBAL_TOP_K和GLOBAL_BOTTOM_K必须非负")
    if not 0 <= REQUIRED_UNDERUSED_RATIO <= 1:
        errors.append("REQUIRED_UNDERUSED_RATIO必须在[0,1]范围内")
    
    # 早停参数检查
    if EARLY_STOPPING_PATIENCE < 0:
        errors.append("EARLY_STOPPING_PATIENCE必须非负")
    if ES_COMPARE_TO not in ("best", "last"):
        errors.append("ES_COMPARE_TO必须是'best'或'last'")
    
    return errors, warnings

# =============================================================================
# 配置摘要打印
# =============================================================================

def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("迭代全局记忆优化配置摘要")
    print("=" * 60)
    print(f"轮数: {MAX_ROUNDS}")
    print(f"每轮因子数: {FACTORS_PER_ROUND}")
    print(f"最少轮数: {MIN_ROUNDS}")
    print(f"早停耐心: {EARLY_STOPPING_PATIENCE}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print("\n[全局记忆配置] ← 核心区别")
    print(f"  Top-K高分: {GLOBAL_TOP_K}")
    print(f"  Bottom-K低分: {GLOBAL_BOTTOM_K}")
    print(f"  最大记忆数: {MAX_MEMORY_ITEMS}")
    print(f"  低频字段比例: {REQUIRED_UNDERUSED_RATIO}")
    print(f"  生成策略: {GENERATION_STRATEGY}")
    print(f"  记忆评分字段: {MEMORY_SCORE_FIELD}")
    print(f"  GPT参考分数: {GPT_REFERENCE_SCORE}")
    print("=" * 60)