# experiments/iterative_negative_memory_with_reports/config.py
"""
iterative_negative_memory_with_reports 实验配置
核心特点：在负向记忆基础上，使用 factor_report 代替 train_score
"""

from shared.paths import results_dir
from pathlib import Path

# =============================================================================
# 实验核心配置
# =============================================================================

# 迭代轮次
MAX_ROUNDS = 3                    # 最大迭代轮次
MIN_ROUNDS = 3                    # 最少运行轮次（强制跑满）
FACTORS_PER_ROUND = 10            # 每轮生成因子数
NEGATIVE_SAMPLES_COUNT = 5        # 每轮生成负样本数（减少到5个，更精准）
MAX_GENERATION_ATTEMPTS = 8       # 每轮最大生成尝试次数

# =============================================================================
# 路径配置
# =============================================================================

RESULTS_DIR = results_dir("iterative_negative_memory_with_reports")
BASELINE_FILE = results_dir("factor_baseline") / "baseline_factor_metrics.csv"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_SAMPLES_DIR = RESULTS_DIR / "negative_samples"
NEGATIVE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 记忆配置（关键区别：使用 report）
# =============================================================================

# 记忆字段：factor_report 代替 train_score
MEMORY_FIELDS = ["code", "factor_report"]  # ← 核心改动
MEMORY_SCORE_FIELD = "train_score"         # 仍用于排序筛选

# =============================================================================
# 负向代理配置（两步法：代码 + 报告）
# =============================================================================

NEGATIVE_AGENT_CONFIG = {
    # 代码生成参数
    "code_temperature": 0.65,
    "code_max_tokens": 800,
    
    # 报告生成参数
    "report_temperature": 0.80,
    "report_max_tokens": 500,
    "report_max_chars": 200,        # 负样本报告字数限制
    
    # 质量控制
    "enforce_distinct": True,        # 强制反模式类型不重复
    "max_calls_per_round": 4,
    "overask": 2,
    "min_code_similarity": 0.80,
    
    # 保存配置
    "save_samples": True,
    "save_format": "csv",
}

# =============================================================================
# 正向代理配置（with reports）
# =============================================================================

POSITIVE_AGENT_CONFIG = {
    # 生成策略混合比例
    "mix_exploit": 4,                # 利用（低温，基于top因子）
    "mix_evolve": 5,                 # 演化（中温，改进现有）
    "mix_diversify": 1,              # 探索（高温，多样化）
    
    # 温度设置
    "temp_exploit": 0.55,
    "temp_evolve": 0.70,
    "temp_diversify": 0.85,
    
    # 质量控制
    "batch_size": 15,
    "min_code_similarity": 0.50,
    "use_negative_memory": True,     # 使用负样本黑名单
    "negative_weight": 0.3,
    "max_attempts": 8,
    
    # 稳健化参数
    "antipattern_topk": 2,           # 只避免最常见的top-K反模式
    "min_unique_fields": 2,          # 因子至少涉及2个不同字段
}

# =============================================================================
# 报告生成配置
# =============================================================================

REPORT_CONFIG = {
    # 正向报告（详细）
    "positive_report_max_tokens": 600,
    "positive_report_temperature": 0.70,
    
    # 负向报告（简洁）
    "negative_report_max_tokens": 300,
    "negative_report_temperature": 0.75,
    
    # 报告模板路径
    "template_file": Path(__file__).parent.parent.parent / "reports" / "report_template.txt",
}

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
    # 迭代配置
    "MAX_ROUNDS": MAX_ROUNDS,
    "MIN_ROUNDS": MIN_ROUNDS,
    "FACTORS_PER_ROUND": FACTORS_PER_ROUND,
    "NEGATIVE_SAMPLES_COUNT": NEGATIVE_SAMPLES_COUNT,
    "MAX_GENERATION_ATTEMPTS": MAX_GENERATION_ATTEMPTS,
    
    # 路径
    "RESULTS_DIR": RESULTS_DIR,
    "BASELINE_FILE": BASELINE_FILE,
    "LOGS_DIR": LOGS_DIR,
    "NEGATIVE_SAMPLES_DIR": NEGATIVE_SAMPLES_DIR,
    
    # 记忆配置
    "MEMORY_FIELDS": MEMORY_FIELDS,
    "MEMORY_SCORE_FIELD": MEMORY_SCORE_FIELD,
    
    # 代理配置
    "NEGATIVE_AGENT_CONFIG": NEGATIVE_AGENT_CONFIG,
    "POSITIVE_AGENT_CONFIG": POSITIVE_AGENT_CONFIG,
    "REPORT_CONFIG": REPORT_CONFIG,
    
    # 早停配置
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "MIN_DELTA": MIN_DELTA,
    "VAL_SCORE_THRESHOLD": VAL_SCORE_THRESHOLD,
}

# =============================================================================
# 校验函数
# =============================================================================

def validate_config():
    """校验配置"""
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
    
    # 检查报告模板
    if not REPORT_CONFIG["template_file"].exists():
        warnings.append(f"报告模板不存在: {REPORT_CONFIG['template_file']}")
    
    return errors, warnings

def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("Iterative Negative Memory WITH REPORTS 配置摘要")
    print("=" * 60)
    print(f"轮数: {MAX_ROUNDS}")
    print(f"每轮因子数: {FACTORS_PER_ROUND}")
    print(f"负样本数: {NEGATIVE_SAMPLES_COUNT}")
    print(f"记忆字段: {MEMORY_FIELDS}")  # ← 显示使用 report
    print(f"Baseline: {BASELINE_FILE}")
    print(f"结果目录: {RESULTS_DIR}")
    print("=" * 60)