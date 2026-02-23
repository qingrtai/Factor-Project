from shared.paths import results_dir
from pathlib import Path

MAX_ROUNDS = 3
FACTORS_PER_ROUND = 5          # ← 改：10 → 5

RESULTS_DIR = results_dir("iterative_baseline")
BASELINE_FILE = results_dir("factor_baseline") / "baseline_factor_metrics.csv"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_SCORE_FIELD = "train_score"

EARLY_STOPPING_PATIENCE = 999
MIN_DELTA = 0.001
MIN_ROUNDS = 3
VAL_SCORE_THRESHOLD = None

CONFIG = {
    "MAX_ROUNDS": MAX_ROUNDS,
    "FACTORS_PER_ROUND": FACTORS_PER_ROUND,
    "RESULTS_DIR": RESULTS_DIR,
    "BASELINE_FILE": BASELINE_FILE,
    "LOGS_DIR": LOGS_DIR,
    "MEMORY_SCORE_FIELD": MEMORY_SCORE_FIELD,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "MIN_DELTA": MIN_DELTA,
    "MIN_ROUNDS": MIN_ROUNDS,
    "VAL_SCORE_THRESHOLD": VAL_SCORE_THRESHOLD,
    "TOP_K_FACTORS": 5,          # ← 改：10 → 5
}

def validate_config():
    errors = []
    warnings = []
    if not BASELINE_FILE.exists():
        errors.append(f"Baseline文件不存在: {BASELINE_FILE}")
    if MAX_ROUNDS <= 0:
        errors.append("MAX_ROUNDS必须大于0")
    if FACTORS_PER_ROUND <= 0:
        errors.append("FACTORS_PER_ROUND必须大于0")
    return errors, warnings

def print_config_summary():
    print("=" * 60)
    print("迭代因子优化配置摘要")
    print("=" * 60)
    print(f"轮数: {MAX_ROUNDS}")
    print(f"每轮因子数: {FACTORS_PER_ROUND}")
    print(f"最少轮数: {MIN_ROUNDS}")
    print(f"早停耐心: {EARLY_STOPPING_PATIENCE}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print("=" * 60)
