# experiments/iterative_negative_memory/config.py
"""
iterative_negative_memory 实验配置

对照组实验：使用负向记忆，不分层 top/middle/bottom
核心改动：
1. NEGATIVE_SAMPLES_COUNT = 35% of FACTORS_PER_ROUND (= 7)
2. batch_size = 5（小批量，提高生成成功率）
"""

from shared.paths import results_dir
from pathlib import Path

# =============================================================================
# 实验核心配置
# =============================================================================

MAX_ROUNDS = 3
MIN_ROUNDS = 3
FACTORS_PER_ROUND = 20
NEGATIVE_SAMPLES_COUNT = max(1, int(round(FACTORS_PER_ROUND * 0.35)))  # = 7
MAX_GENERATION_ATTEMPTS = 15

# =============================================================================
# 路径配置
# =============================================================================

RESULTS_DIR = results_dir("iterative_negative_memory")
BASELINE_FILE = results_dir("factor_baseline") / "baseline_factor_metrics.csv"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_SAMPLES_DIR = RESULTS_DIR / "negative_samples"
NEGATIVE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 记忆配置
# =============================================================================

MEMORY_SCORE_FIELD = "train_score"

# =============================================================================
# 负向代理配置
# =============================================================================

NEGATIVE_AGENT_CONFIG = {
    "temperature_first": 0.75,
    "temperature_later": 0.85,
    "max_calls_per_round": 4,
    "overask": 3,
    "min_code_similarity": 0.80,
    "save_samples": True,
    "save_format": "json",
}

# =============================================================================
# 正向代理配置
# =============================================================================

POSITIVE_AGENT_CONFIG = {
    "batch_size": 5,                 # 小批量生成（和 baseline 一致）
    "min_code_similarity": 0.70,
    "use_negative_memory": True,
    "negative_weight": 0.3,
    "max_attempts": 12,
}

# GPT 配置
GPT_TEMPERATURE = 0.50
GPT_MAX_TOKENS = 2200
TIMEOUT = 90

# =============================================================================
# 早停配置
# =============================================================================

EARLY_STOPPING_PATIENCE = 999
MIN_DELTA = 0.001
VAL_SCORE_THRESHOLD = None

# =============================================================================
# 导出 CONFIG 字典
# =============================================================================

CONFIG = {
    "MAX_ROUNDS": MAX_ROUNDS,
    "MIN_ROUNDS": MIN_ROUNDS,
    "FACTORS_PER_ROUND": FACTORS_PER_ROUND,
    "NEGATIVE_SAMPLES_COUNT": NEGATIVE_SAMPLES_COUNT,
    "MAX_GENERATION_ATTEMPTS": MAX_GENERATION_ATTEMPTS,
    "RESULTS_DIR": RESULTS_DIR,
    "BASELINE_FILE": BASELINE_FILE,
    "LOGS_DIR": LOGS_DIR,
    "NEGATIVE_SAMPLES_DIR": NEGATIVE_SAMPLES_DIR,
    "MEMORY_SCORE_FIELD": MEMORY_SCORE_FIELD,
    "NEGATIVE_AGENT_CONFIG": NEGATIVE_AGENT_CONFIG,
    "POSITIVE_AGENT_CONFIG": POSITIVE_AGENT_CONFIG,
    "GPT_TEMPERATURE": GPT_TEMPERATURE,
    "GPT_MAX_TOKENS": GPT_MAX_TOKENS,
    "TIMEOUT": TIMEOUT,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "MIN_DELTA": MIN_DELTA,
    "VAL_SCORE_THRESHOLD": VAL_SCORE_THRESHOLD,
}

# =============================================================================
# 校验函数
# =============================================================================

def validate_config():
    errors = []
    warnings = []
    if not BASELINE_FILE.exists():
        errors.append(f"Baseline文件不存在: {BASELINE_FILE}")
    if MAX_ROUNDS <= 0:
        errors.append("MAX_ROUNDS必须大于0")
    if FACTORS_PER_ROUND <= 0:
        errors.append("FACTORS_PER_ROUND必须大于0")
    if NEGATIVE_SAMPLES_COUNT < 0:
        errors.append("NEGATIVE_SAMPLES_COUNT不能为负")
    if NEGATIVE_SAMPLES_COUNT == 0:
        warnings.append("NEGATIVE_SAMPLES_COUNT为0，将退化为baseline")
    return errors, warnings

def print_config_summary():
    print("=" * 60)
    print("Iterative Negative Memory 配置摘要")
    print("=" * 60)
    print(f"轮数: {MAX_ROUNDS}")
    print(f"每轮因子数: {FACTORS_PER_ROUND}")
    print(f"负样本数: {NEGATIVE_SAMPLES_COUNT} (35% of {FACTORS_PER_ROUND})")
    print(f"GPT Temperature: {GPT_TEMPERATURE}")
    print(f"最大尝试次数: {MAX_GENERATION_ATTEMPTS}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"结果目录: {RESULTS_DIR}")
    print("=" * 60)
