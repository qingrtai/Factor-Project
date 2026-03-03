from shared.paths import results_dir
from pathlib import Path

# === 基础配置（复用 iterative_baseline）===
MAX_ROUNDS = 3
FACTORS_PER_ROUND = 20
RESULTS_DIR = results_dir("iterative_baseline_with_reports_v2")
BASELINE_FILE = results_dir("factor_baseline") / "baseline_factor_metrics.csv"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# === 报告配置（改进版）===
REPORT_TEMPLATE_FILE = Path(__file__).resolve().parents[2] / "reports" / "report_template.txt"
REPORT_MAX_TOKENS = 600     # 增加到 600（原来 200 太少）
REPORT_TEMPERATURE = 0.7    
GENERATE_REPORTS_FOR = "ranked"  # 改为 "ranked"：top 3 详细报告 + bottom 3 简要报告

# === 因子生成配置 ===
FACTOR_GENERATION_MAX_TOKENS = 1200  # 增加到 1200（原来 900）
GENERATION_TEMPERATURE = 0.8  # 新增：生成因子时的温度

# === 因子分配方案（新增）===
# "A" = 有 middle（全部带报告）: top 35% + middle 30% + bottom 35%
# "C" = 无 middle（丢弃中间 30%）: top 35% + bottom 35%
ALLOCATION_SCHEME = "A"

# === 报告策略（新增）===
TOP_K_DETAILED = 3  # 前 K 个生成详细报告
BOTTOM_K_BRIEF = 3  # 后 K 个生成简要报告（作为负例）
INCLUDE_COMPARATIVE_ANALYSIS = True  # 在提示词中包含对比分析

# === 因子生成配置 ===
MAX_REFILL_CALLS = 3  
REQUIRE_FULL_ROUND = False  

# === 报告失败处理 ===
REPORT_ON_FAILURE = "fallback"  # 改为 fallback（提供基本信息）

# === 记忆配置 ===
MEMORY_SCORE_FIELD = "train_score"  
GPT_MEMORY_FIELD = "factor_report"  

# === 早停配置 ===
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
    # 报告配置
    "REPORT_TEMPLATE_FILE": REPORT_TEMPLATE_FILE,
    "REPORT_MAX_TOKENS": REPORT_MAX_TOKENS,
    "REPORT_TEMPERATURE": REPORT_TEMPERATURE,
    "GENERATE_REPORTS_FOR": GENERATE_REPORTS_FOR,
    "TOP_K_DETAILED": TOP_K_DETAILED,
    "BOTTOM_K_BRIEF": BOTTOM_K_BRIEF,
    "INCLUDE_COMPARATIVE_ANALYSIS": INCLUDE_COMPARATIVE_ANALYSIS,
    # 生成配置
    "FACTOR_GENERATION_MAX_TOKENS": FACTOR_GENERATION_MAX_TOKENS,
    "GENERATION_TEMPERATURE": GENERATION_TEMPERATURE,
    # 因子分配方案
    "ALLOCATION_SCHEME": ALLOCATION_SCHEME,
    # 记忆配置
    "MEMORY_SCORE_FIELD": MEMORY_SCORE_FIELD,
    "GPT_MEMORY_FIELD": GPT_MEMORY_FIELD,
    # 早停配置
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "MIN_DELTA": MIN_DELTA,
    "MIN_ROUNDS": MIN_ROUNDS,
    "VAL_SCORE_THRESHOLD": VAL_SCORE_THRESHOLD,
    "MAX_REFILL_CALLS": MAX_REFILL_CALLS,
    "REQUIRE_FULL_ROUND": REQUIRE_FULL_ROUND,
    "REPORT_ON_FAILURE": REPORT_ON_FAILURE,
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
    if not REPORT_TEMPLATE_FILE.exists():
        errors.append(f"报告模板文件不存在: {REPORT_TEMPLATE_FILE}")
    if ALLOCATION_SCHEME not in ("A", "C"):
        errors.append(f"ALLOCATION_SCHEME 必须是 'A' 或 'C'，当前: {ALLOCATION_SCHEME}")
    
    return errors, warnings

def print_config_summary():
    print("=" * 60)
    print("迭代因子优化配置摘要（改进版报告生成）")
    print("=" * 60)
    print(f"轮数: {MAX_ROUNDS}")
    print(f"每轮因子数: {FACTORS_PER_ROUND}")
    print(f"最少轮数: {MIN_ROUNDS}")
    print(f"早停耐心: {EARLY_STOPPING_PATIENCE}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    
    print("\n[因子分配方案]")
    print(f"  方案: {ALLOCATION_SCHEME}")
    if ALLOCATION_SCHEME == "A":
        print(f"  Top 35% (strengths报告) + Middle 30% (neutral报告) + Bottom 35% (weaknesses报告)")
    else:
        print(f"  Top 35% (strengths报告) + Bottom 35% (weaknesses报告), 丢弃中间30%")
    
    print("\n[报告生成配置]")
    print(f"  模板文件: {REPORT_TEMPLATE_FILE}")
    print(f"  报告 max tokens: {REPORT_MAX_TOKENS} (提升到 600)")
    print(f"  生成 max tokens: {FACTOR_GENERATION_MAX_TOKENS} (提升到 1200)")
    print(f"  生成温度: {GENERATION_TEMPERATURE}")
    print(f"  报告策略: Top {TOP_K_DETAILED} 详细 + Bottom {BOTTOM_K_BRIEF} 简要")
    print(f"  对比分析: {INCLUDE_COMPARATIVE_ANALYSIS}")
    print(f"  GPT 记忆字段: {GPT_MEMORY_FIELD}")
    print("=" * 60)
