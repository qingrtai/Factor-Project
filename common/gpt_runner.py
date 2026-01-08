# gpt_runner.py — secure TLS + split timeouts + retries + robust cleaning
from __future__ import annotations
import os
import re
import json
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from common.column_desc import COLUMN_DESC

load_dotenv()

# ---------------- env & defaults ----------------
API_URL = os.getenv("GPT_API_URL", "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions")
API_KEY = os.getenv("SCHOOL_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
MODEL   = os.getenv("GPT_MODEL", "gpt-4o-2024-08-06")

# Split timeouts (seconds)
CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "5"))   # 建议 3~8
READ_TIMEOUT    = float(os.getenv("LLM_READ_TIMEOUT", "12"))     # 建议 8~20

# TLS verification
REQUESTS_CA = os.getenv("REQUESTS_CA_BUNDLE", "")
USE_INSECURE = os.getenv("ALLOW_INSECURE_SSL", "").strip() in {"1", "true", "True"}

# ---------------- HTTPS session with retries ----------------
SESSION = requests.Session()

# TLS verify (prefer certifi)
try:
    import certifi
    CA_FILE = REQUESTS_CA if REQUESTS_CA else certifi.where()
except Exception:
    CA_FILE = REQUESTS_CA if REQUESTS_CA else True

if USE_INSECURE:
    SESSION.verify = False
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
else:
    SESSION.verify = CA_FILE

# Retries: connect/read/5xx
retries = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.6,                 # 0.6 -> 1.2
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset({"POST"}),
)
adapter = HTTPAdapter(max_retries=retries, pool_maxsize=10)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# Headers
SESSION.headers.update({"Content-Type": "application/json"})
if API_KEY:
    SESSION.headers.update({"Authorization": f"Bearer {API_KEY}"})


# ================= core call =================
def _strip_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"```(?:python|json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()

def call_gpt(prompt: str,
             temperature: float = 0.8,
             max_tokens: int = 600,
             timeout: Optional[float] = None,
             model: Optional[str] = None) -> str:
    """
    调用 Chat Completions，返回纯字符串（message.content）。
    - 分离 (connect, read) 超时；若上层传单一 timeout，则对两侧同时取最小值。
    - 轻微约束 temperature/max_tokens 防止异常参数。
    """
    if not API_KEY:
        print("[gpt_runner] ERROR: API key not set (SCHOOL_API_KEY / OPENAI_API_KEY / GPT_API_KEY)")
        return ""

    # clamp params
    temperature = float(min(max(temperature, 0.0), 1.0))
    max_tokens  = int(max(64, min(max_tokens, 4096)))

    ct = CONNECT_TIMEOUT
    rt = READ_TIMEOUT
    if timeout is not None:
        t = float(timeout)
        ct = min(ct, t)
        rt = min(rt, t)

    payload = {
        "model": model or MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        print(f"[gpt_runner] → POST {API_URL}  (ct={ct}s, rt={rt}s, verify={'False' if USE_INSECURE else CA_FILE})")
        resp = SESSION.post(API_URL, json=payload, timeout=(ct, rt), stream=False)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        msg = _strip_fences(msg)
        print(f"[gpt_runner] ← OK ({len(msg)} chars)")
        return msg
    except requests.exceptions.SSLError as e:
        print("[gpt_runner] SSL ERROR:", e)
        if not USE_INSECURE and not REQUESTS_CA:
            print("[gpt_runner] Hint: set REQUESTS_CA_BUNDLE=path\\to\\rootCA.pem  或  ALLOW_INSECURE_SSL=1（仅调试用）")
        return ""
    except requests.exceptions.ConnectTimeout:
        print(f"[gpt_runner] TIMEOUT: connect > {ct}s")
        return ""
    except requests.exceptions.ReadTimeout:
        print(f"[gpt_runner] TIMEOUT: read > {rt}s")
        return ""
    except requests.exceptions.RequestException as e:
        code = getattr(e.response, "status_code", None)
        txt  = getattr(e.response, "text", "")[:300] if getattr(e, "response", None) else ""
        print(f"[gpt_runner] REQUEST ERROR: {e}  status={code}  body={txt}")
        return ""
    except Exception as e:
        print("[gpt_runner] ERROR:", e)
        return ""


# ================= code cleaning & validation =================
_CODE_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def _normalize_unicode_punct(code: str) -> str:
    trans_table = str.maketrans({
        "“": '"', "”": '"', "„": '"', "‟": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "＇": "'",
        "（": "(", "）": ")", "，": ",", "：": ":", "。": ".",
    })
    return code.translate(trans_table)

def _strip_non_ascii(code: str) -> str:
    return re.sub(r"[^\x20-\x7E\n]+", "", code)

def check_bracket_balance(code: str) -> bool:
    count = 0
    for c in code:
        if c == "(": count += 1
        elif c == ")": count -= 1
        if count < 0: return False
    return count == 0

def auto_fix_code(code: str) -> str:
    # 安全的微修补（不改变语义）：断行、去 zscore、简单多余括号收缩
    code = re.sub(r"\)\s*(?=data\['factor_score'\])", r")\n", code)
    code = code.replace(".zscore()", "")
    code = re.sub(r"\)\)\)", "))", code)
    return code

def clean_code(code: str) -> str:
    """
    清洗 GPT 返回的代码字符串（强化版）：
    - 去掉 markdown 围栏、统一中英文标点与引号、移除非 ASCII
    - 移除所有注释（包括行内注释）和空行
    - 拒绝描述性文字（profitability、efficiency等）
    - 修补 .pct_change 的 fill_method=None
    - 括号匹配检查
    - 限制行数（2-10行合理）
    """
    if not isinstance(code, str):
        code = str(code or "")
    
    # 1. 去围栏
    code = _CODE_FENCE_RE.sub("", code).strip()
    
    # 2. 统一标点
    code = _normalize_unicode_punct(code)
    
    # 3. 移除非ASCII（但保留换行）
    code = _strip_non_ascii(code)

    # 4. 移除注释与空行
    lines = []
    for line in code.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 移除行内注释
        line = re.sub(r"#.*", "", line).strip()
        if line:
            lines.append(line)
    
    if not lines:
        raise ValueError("code_empty: no valid lines after cleaning")
    
    code = "\n".join(lines)
    
    # 5. 拒绝描述性文字（GPT 常加的解释）
    forbidden_words = [
        'profitability', 'efficiency', 'solvency', 'liquidity',
        'leverage', 'growth', 'quality', 'momentum', 'value',
        'this calculates', 'this measures', 'this represents',
        'higher is better', 'lower is better'
    ]
    code_lower = code.lower()
    for word in forbidden_words:
        if word in code_lower:
            raise ValueError(f"forbidden_text: contains '{word}' (should be pure code only)")
    
    # 6. 行数限制
    if len(lines) > 10:
        raise ValueError(f"code_too_long: {len(lines)} lines (max 10 for clarity)")

    # 7. 修补 pct_change — 强制 fill_method=None
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

    # 8. 基础自动修补 + 括号检查
    code = auto_fix_code(code)
    if not check_bracket_balance(code):
        raise ValueError("syntax_error: unmatched brackets")

    return code.strip()

def normalize_code_for_dedup(code: str) -> str:
    """用于去重的规范化：去围栏、统一空白与标点、压缩空行。"""
    code = _CODE_FENCE_RE.sub("", code or "").strip()
    code = _normalize_unicode_punct(code)
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()

def is_duplicate(code: str, code_list: List[str]) -> bool:
    return hash((normalize_code_for_dedup(code))) in {hash(normalize_code_for_dedup(x)) for x in code_list}

def validate_column_references(code: str, allowed_cols: List[str], df: Optional[pd.DataFrame] = None) -> Tuple[bool, Optional[str]]:
    """
    列校验（大小写无关）：只能引用 allowed_cols ∩ df.columns 以及执行过程中创建的新列与 factor_score。
    如果 df 提供，则与其列做交集；否则仅使用 allowed_cols。
    """
    allowed = {c.lower() for c in allowed_cols}
    if isinstance(df, pd.DataFrame):
        allowed &= {c.lower() for c in df.columns}

    created = set()
    for line in code.splitlines():
        if line.strip().startswith("#"):
            continue
        # 捕捉新创建的列
        m = re.match(r"\s*data\[['\"](.+?)['\"]\]\s*=", line)
        if m:
            created.add(m.group(1).lower())

        # 遍历引用
        refs = re.findall(r"data\[['\"](.+?)['\"]\]", line)
        for col in refs:
            col_lower = col.lower()
            # 忽略当前行赋值的目的列
            if m and col_lower == m.group(1).lower():
                continue
            if col_lower not in allowed and col_lower not in created and col_lower != "factor_score":
                return False, f"Invalid reference: {col}"

    return True, None


# ================= optional local exec helpers (for quick tests) =================
def safe_exec_factor_code(code_str: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    仅用于本地快速测试（不是生产评分链路）。
    注意：真正的安全执行在 core.factor_exec.safe_execute。
    """
    code_str = code_str.replace(".pct_change()", ".pct_change(fill_method=None)")
    local_env = {"data": data.copy(), "np": np, "pd": pd}
    try:
        exec(code_str, {"__builtins__": {}}, local_env)  # 用 local_env 做 locals
        out = local_env["data"]
        if "factor_score" not in out.columns:
            raise RuntimeError("no_output: data['factor_score'] not set")
        return out
    except Exception as e:
        raise RuntimeError(f"执行失败：{e}")

def test_factor_execution(df: pd.DataFrame, code: str, verbose: bool = True) -> Tuple[bool, str]:
    """
    供开发期快速验证的工具：不影响主流程。
    - 清洗 + 列校验 + 本地安全执行（简化版本）
    """
    try:
        code = clean_code(code)
        code = auto_fix_code(code)

        allowed_cols = [c for c in COLUMN_DESC.keys() if c in df.columns]
        is_valid, err = validate_column_references(code, allowed_cols, df)
        if not is_valid:
            if verbose: print("Column validation failed:", err)
            return False, "invalid_column"

        if not check_bracket_balance(code):
            return False, "syntax_error: unmatched brackets"

        # 一些常见的禁止字段（可按需调整）
        banned_fields = ['seq']  # 示例：真的禁用再加
        if any(bad.lower() in code.lower() for bad in banned_fields):
            return False, "invalid_column: banned field used"

        if verbose:
            print(" Running GPT Code:\n", code)

        result_df = safe_exec_factor_code(code, df)

        if 'factor_score' not in result_df.columns:
            return False, "no_output"
        fs = pd.to_numeric(result_df['factor_score'], errors="coerce")
        if fs.isna().all():
            return False, "nan_output"
        if fs.dropna().nunique() <= 1:
            return False, "constant_or_nan"

        return True, ""
    except ValueError as ve:
        return False, str(ve)
    except Exception as e:
        print("Execution exception:", e)
        os.makedirs("results", exist_ok=True)
        try:
            with open("results/last_failed_code.txt", "w", encoding="utf-8") as f:
                f.write(code or "")
        except Exception:
            pass
        return False, "code_exception"


# ================= multi-generate helper (optional) =================
def call_gpt_and_extract_factor_code(prompt: str, n_generate: int = 20, timeout: int = 60) -> List[Dict[str, str]]:
    """
    多次调用 GPT，尝试提取多段代码（用于老脚本的兼容/批量试错）。
    generate_factors.py 已有更完善的流程；本函数可用于临时批量探索。
    """
    seen: List[str] = []
    results: List[Dict[str, str]] = []
    for _ in range(n_generate):
        raw = call_gpt(prompt, timeout=timeout)
        if not raw:
            continue
        try:
            code = clean_code(raw)
            if not code or is_duplicate(code, seen):
                continue
            seen.append(code)
            results.append({"code": code, "style": "unknown"})
        except Exception:
            continue
    return results
