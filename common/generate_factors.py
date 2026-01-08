# common/generate_factors.py
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from common.column_desc import COLUMN_DESC
from common.gpt_runner import call_gpt
try:
    # 如果你在 gpt_runner 里实现了 clean_code，就用那一份
    from common.gpt_runner import clean_code as _clean_code_external
except Exception:
    _clean_code_external = None

from core.factor_exec import safe_execute
from shared.config_loader import load_global_config

# -------------------------
# 可调参数（也可由 experiments.yaml 控制，这里给默认值）
# -------------------------
DEFAULT_SAMPLE_ROWS = 5000
MAX_ATTEMPTS_PER_FACTOR = 5
BANNED_TOKENS = (
    " import ", " from ", " eval(", " exec(", " open(", " os.", " sys.",
    " subprocess.", " socket.", " pathlib.", " requests.", " urllib.", " httpx.",
    " boto3", " sqlalchemy", " sklearn", " torch", " tensorflow", " dask", " joblib", " pickle",
    " read_csv(", " read_parquet(", " read_excel(", " to_csv(", " to_parquet(", " to_excel(", " to_json(", " to_pickle("
)
ALLOWED_LIBS = "numpy as np, pandas as pd"
PCT_CHANGE_FIX = True  # 遇到 .pct_change() 自动改为 .pct_change(fill_method=None)
CORR_DEDUP = False     # 是否进行相关性去重（默认关，防误杀）


# -------------------------
# 小工具
# -------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _results_dir() -> Path:
    p = _project_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _codes_txt_path() -> Path:
    return _results_dir() / "factor_codes.txt"

def _strip_code_fences(code: str) -> str:
    code = re.sub(r"^\s*```(?:python)?\s*|\s*```\s*$", "", code, flags=re.IGNORECASE|re.MULTILINE)
    return code.strip()

def _normalize_code(code: str) -> str:
    # 归一化用于去重：去围栏、统一空白与引号
    code = _strip_code_fences(code)
    trans = str.maketrans({
        "“": '"', "”": '"', "„": '"', "‟": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "＇": "'",
        "（": "(", "）": ")", "，": ",", "：": ":", "。": ".",
    })
    code = code.translate(trans)
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()

# 在 generate_factors.py 中，替换现有的 _local_clean_code 函数

def _local_clean_code(code: str) -> str:
    """
    强化清洗：移除注释、描述性文字，保留2-10行纯逻辑代码
    专为GPT迭代优化场景设计
    """
    # 1. 基础清洗：去围栏、统一标点
    code = _strip_code_fences(code)
    # 替换第78-83行
    # 使用 replace 链式调用代替 maketrans
    code = (code
        .replace(""", '"').replace(""", '"').replace("„", '"').replace("‟", '"').replace("＂", '"')
        .replace("'", "'").replace("'", "'").replace("‚", "'").replace("‛", "'").replace("＇", "'")
        .replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":").replace("。", ".")
    )

    
    # 2. 逐行处理：移除注释、空行
    lines = []
    for line in code.splitlines():
        # 移除行内注释（# 后面的所有内容）
        line = re.sub(r'#.*', '', line).strip()
        if line:  # 跳过空行
            lines.append(line)
    
    if not lines:
        raise ValueError("Code is empty after cleaning")
    
    code = '\n'.join(lines)
    
    # 3. 拒绝明显的描述性句子（允许少量变量名，因为 Prompt 会控制）
    forbidden_patterns = [
        r'this\s+(calculates|measures|represents)',  # "this calculates..."
        r'(higher|lower)\s+(is\s+)?better',          # "higher is better"
        r'#.*\b(profitability|efficiency)\b',        # 注释中的描述
    ]
    code_lower = code.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, code_lower):
            raise ValueError(f"Code contains descriptive text: matched '{pattern}'")

    # 如果检测到多个描述性变量名，也拒绝（说明 GPT 没遵守规则）
    descriptive_vars = ['profitability', 'efficiency', 'solvency', 'liquidity', 
                        'leverage', 'growth', 'quality', 'momentum', 'value']
    var_count = sum(1 for word in descriptive_vars if word in code_lower)
    if var_count >= 2:  # 容忍1个，但2个以上就拒绝
        raise ValueError(f"Code uses too many descriptive variable names ({var_count} found)")
    
    # 4. 行数限制
    if len(lines) < 1:
        raise ValueError("Code has no valid lines")
    if len(lines) > 3:
        raise ValueError(f"Code too long ({len(lines)} lines), keep it concise (2-3 lines)")
    
    # 5. 必须有核心赋值
    if not re.search(r"data\s*\[\s*['\"]factor_score['\"]\s*\]\s*=", code):
        raise ValueError("Missing data['factor_score'] assignment")
    
    # 6. 修复 pct_change（保持原有逻辑）
    if PCT_CHANGE_FIX:
        code = code.replace(".pct_change()", ".pct_change(fill_method=None)")
        def _add_fill(m: re.Match) -> str:
            inner = m.group(1)
            if "fill_method" in inner:
                return f".pct_change({inner})"
            inner = inner.strip()
            if inner == "":
                return ".pct_change(fill_method=None)"
            return f".pct_change({inner}, fill_method=None)"
        code = re.sub(r"\.pct_change\((.*?)\)", _add_fill, code)
    
    # 7. 统一空白（但保留换行）
    code = re.sub(r'[ \t]+', ' ', code)  # 多个空格/tab -> 单空格
    
    return code.strip()


# 同时更新 _clean_code 函数（优先使用外部的，回退到本地的）
def _clean_code(code: str) -> str:
    """清洗代码：优先使用 gpt_runner 的版本，回退到本地强化版"""
    if _clean_code_external:
        try:
            return _clean_code_external(code)
        except Exception:
            pass
    return _local_clean_code(code)

def _has_banned_tokens(code: str) -> bool:
    c = " " + code.replace("\n", " ") + " "
    return any(tok in c for tok in BANNED_TOKENS)

def _ensure_output_var(code: str) -> bool:
    # 必须写入 data['factor_score'] =
    return bool(re.search(r"data\s*\[\s*['\"]factor_score['\"]\s*\]\s*=", code))

def _preview(code: str, width: int = 80) -> str:
    c = _strip_code_fences(code)
    return (c[:width] + "...") if len(c) > width else c

def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan
    try:
        return float(np.corrcoef(x[m], y[m])[0, 1])
    except Exception:
        return np.nan


# -------------------------
# Prompt
def _build_generation_prompt(sample_cols: List[str]) -> str:
    allowed = ", ".join(f"`{c}`" for c in sample_cols)
    return f"""
You are a quantitative finance expert.
TASK: Generate exactly ONE Python factor on DataFrame `data`.

HARD RULES:
1) OUTPUT ONLY executable Python code (NO markdown fences, NO comments, NO explanations).
2) The FIRST line MUST start with: data['factor_score'] = 
3) The LAST line MUST be: data['factor_score'] = data['factor_score'].fillna(0)
4) Use ONLY the columns listed below: {allowed}
5) Handle division-by-zero via np.where(denominator==0, 0, numerator/denominator).
6) If using .pct_change(), MUST specify fill_method=None (e.g., .pct_change(4, fill_method=None)).
7) Do NOT create intermediate variables - write the formula directly in one line.
8) Total code: 2-3 lines MAXIMUM (formula line + fillna line, optionally one intermediate step).
9) Do NOT import anything (np, pd already available).

GOOD EXAMPLES:
data['factor_score'] = np.where(data['saleq']==0, 0, data['cogsq']/data['saleq'])
data['factor_score'] = data['factor_score'].fillna(0)

data['factor_score'] = (data['niq']/data['atq']).pct_change(4, fill_method=None)
data['factor_score'] = data['factor_score'].fillna(0)

data['factor_score'] = data['ibq'].rolling(4, min_periods=1).mean() / data['atq']
data['factor_score'] = data['factor_score'].fillna(0)

BAD EXAMPLES (DO NOT USE):
profitability = data['niq'] / data['atq']  # ❌ NO intermediate variables
data['factor_score'] = profitability       # ❌ NO descriptive names

Now produce ONE factor (2-3 lines total).
"""


# -------------------------
# 采样数据（用于快速运行校验）
# -------------------------
def _load_sample_df(n_rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    g = load_global_config()
    raw_file = g["data"]["raw_file"]
    date_col = g["schema"]["date_col"]

    # 只加载需要的列（尽量轻）
    want_cols = set(COLUMN_DESC.keys())
    usecols = None  # 若 CSV 列很大，可改为 list(want_cols ∪ {date_col})
    df = pd.read_csv(raw_file, parse_dates=[date_col], low_memory=False, usecols=usecols)

    # 子集 + 打乱（稳定性可用固定 seed）
    if len(df) > n_rows:
        df = df.sample(n_rows, random_state=2025).sort_values(date_col)
    # 只保留工程允许的列，防止引用出圈
    keep = [c for c in df.columns if c in want_cols or c == date_col]
    return df[keep].reset_index(drop=True)


# -------------------------
# 验证：运行 + 质量门槛
# -------------------------
def _run_and_check(code: str, df_sample: pd.DataFrame, min_coverage: float = 0.05) -> Tuple[bool, Optional[pd.Series], str]:
    """
    返回：(是否通过, 因子序列或 None, 失败原因)
    """
    # 基础规则检查
    if _has_banned_tokens(code):
        return False, None, "contains banned tokens or IO calls"
    if not _ensure_output_var(code):
        return False, None, "missing data['factor_score'] assignment"

    # 执行
    try:
        s = safe_execute(code, df_sample)
    except Exception as e:
        return False, None, f"exec error: {e}"

    # 质量门槛：覆盖率 + 非常数
    cov = float(1.0 - pd.isna(s).mean())
    if cov < min_coverage:
        return False, None, f"low coverage ({cov:.1%})"
    if s.dropna().nunique() <= 1:
        return False, None, "degenerate (<=1 unique non-NaN)"

    return True, s, ""


# -------------------------
# 主流程
# -------------------------
def generate_factors(
    n: int = 10,
    *,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_attempts_per_factor: int = MAX_ATTEMPTS_PER_FACTOR,
    correlation_dedup: bool = CORR_DEDUP,
    save_codes: bool = True,
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """
    生成 n 个可运行的因子代码：
      - 使用 GPT 生成，并对每段代码进行本地快速校验（运行 + 质量门槛）
      - 去重策略：代码签名（规范化后精确匹配）；可选：与已通过因子的样本相关性去重
      - 通过后加入结果池；最终返回 [{'code': code_str}, ...]
      - 如 save_codes=True，会将代码按顺序写入 results/factor_codes.txt（便于追溯）
    """
    if verbose:
        print("[generate_factors] Loading sample data...")
    df_sample = _load_sample_df(n_rows=sample_rows)
    if verbose:
        print(f"[generate_factors] Loaded {len(df_sample)} rows for testing\n")

    # 允许列（传给 prompt）
    g = load_global_config()
    date_col = g["schema"]["date_col"]
    sample_cols = [c for c in df_sample.columns if c != date_col and c in COLUMN_DESC]

    results: List[Dict[str, str]] = []
    accepted_codes_norm: set[str] = set()
    accepted_series: List[pd.Series] = []

    if verbose:
        print(f"[generate_factors] Generating {n} factors (one by one)...")
        print(f"[generate_factors] Correlation check: {bool(correlation_dedup)}\n")

    total_attempts = 0

    while len(results) < n:
        idx = len(results) + 1
        if verbose:
            print("=" * 60)
            print(f"[Factor {idx}/{n}] Generating...")

        prompt = _build_generation_prompt(sample_cols)

        success = False
        for attempt in range(1, max_attempts_per_factor + 1):
            total_attempts += 1
            if verbose:
                print(f"  [Attempt {attempt}] Calling GPT...", flush=True)
            raw = call_gpt(prompt=prompt, temperature=0.6, max_tokens=400)
            raw = raw or ""
            raw = raw.strip()

            # 预清洗 + 快速拦截
            print(f"\n{'='*60}")
            print(f"[DEBUG] Raw code from GPT (attempt {attempt}):")
            print(raw)
            print(f"{'='*60}\n")
            code = _clean_code(raw)

            if _has_banned_tokens(code):
                if verbose:
                    print("  [REJECTED] Contains banned tokens")
                continue
            if not _ensure_output_var(code):
                if verbose:
                    print("  [REJECTED] Missing data['factor_score'] assignment")
                continue

            # 代码签名去重
            norm = _normalize_code(code)
            if norm in accepted_codes_norm:
                if verbose:
                    print("  [DUPLICATE] Code already exists in pool")
                    print("  [REJECTED] Validation failed")
                continue

            # 运行校验
            ok, s, err = _run_and_check(code, df_sample)
            if not ok:
                if verbose:
                    print(f"  [REJECTED] {err}")
                continue

            # 相关性去重（可选）
            if correlation_dedup and accepted_series:
                dup = False
                for prev in accepted_series:
                    r = _pearson_corr(s, prev)
                    if np.isfinite(r) and abs(r) >= 0.98:
                        dup = True
                        break
                if dup:
                    if verbose:
                        print("  [DUPLICATE] Highly correlated with an accepted factor")
                        print("  [REJECTED] Validation failed")
                    continue

            # 通过
            accepted_codes_norm.add(norm)
            accepted_series.append(s)
            results.append({"code": code})
            if verbose:
                print("  [SUCCESS] Code validated")
                print("  [ACCEPTED] ✓")
                print(f"  Code preview: {_preview(code, 80)}")

            success = True
            break  # 下一因子

        if not success:
            # 没有通过，继续尝试下一个“位次”的因子（保持总量尽量接近 n）
            if verbose:
                print("  [FAIL] Max attempts reached; moving on.")

    # 汇总
    if verbose:
        print("\n" + "=" * 60)
        print("[generate_factors] ========== SUMMARY ==========")
        print(f"[generate_factors] Requested: {n}")
        print(f"[generate_factors] Generated: {len(results)}")
        print(f"[generate_factors] Total attempts: {total_attempts}")
        rate = 100.0 * len(results) / max(1, total_attempts)
        print(f"[generate_factors] Success rate: {rate:.1f}%")
        print("[generate_factors] ================================")
        print("")

    # 可选：把代码落盘，便于 compute / 复现实验
    if save_codes and results:
        out = _codes_txt_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for i, r in enumerate(results, start=1):
                f.write(f"# ===== Factor {i} =====\n")
                f.write(r["code"].strip())
                f.write("\n\n")
        if verbose:
            print(f"[generate_factors] Saved codes -> {out}")

    return results


if __name__ == "__main__":
    # 方便单独调试
    _ = generate_factors(n=10, correlation_dedup=False, save_codes=True, verbose=True)
