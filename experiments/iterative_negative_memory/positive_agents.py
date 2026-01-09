# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/positive_agents.py
"""
FIXED VERSION - 简化 Prompt + 强制 baseline 模式

核心改动：
1. 删除冗长的 rolling/pct_change 示例（它们误导了 GPT）
2. 强制要求 70% 因子使用 np.where（baseline 的核心模式）
3. 简化 Prompt 到 150 行
4. 只展示 baseline 的成功因子
5. 明确指令："模仿 baseline，改变字段组合"
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import re
import time
import random
import threading
import logging
import numpy as np

from shared.code_utils import (
    normalize_code,
    code_similarity,
    validate_and_fix_code
)

logger = logging.getLogger(__name__)

try:
    from common.column_desc import COLUMN_DESC
    ALLOWED_FIELDS: List[str] = list(COLUMN_DESC.keys())
    _FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS)
except Exception as e:
    raise ImportError(f"[positive] Unable to import COLUMN_DESC: {e}")

_local_seed = int(time.time() * 1000) % 1_000_000
np.random.seed(_local_seed)
random.seed(_local_seed)

try:
    from common.gpt_runner import call_gpt
    GPT_AVAILABLE = True
except Exception:
    GPT_AVAILABLE = False

try:
    from .config import CONFIG
except Exception:
    CONFIG = {}

POS_CFG = CONFIG.get("POSITIVE_AGENT_CONFIG", {}) or {}
TIMEOUT_S = int(CONFIG.get("TIMEOUT", 90))
GPT_TEMP_DEFAULT = float(CONFIG.get("GPT_TEMPERATURE", 0.50))  # 降低到 0.50
GPT_MAX_TOKENS = int(CONFIG.get("GPT_MAX_TOKENS", 2200))
MAX_RETRIES = int(CONFIG.get("MAX_RETRIES", 6)) or 6

BASE_MIN_SIM = float(POS_CFG.get("min_code_similarity", 0.70))
BATCH_SIZE = int(POS_CFG.get("batch_size", 10))


def _extract_fields_from_code(code: str) -> List[str]:
    """提取代码中使用的字段"""
    fields = []
    
    # 格式1: data.get('field', 0)
    fields.extend(re.findall(r"data\.get\(\s*['\"]([^'\"]+)['\"]\s*,", code))
    
    # 格式2: data['field']
    fields.extend(re.findall(r"data\[['\"]([^'\"]+)['\"]\]", code))
    
    # 过滤：只保留白名单中的输入字段
    whitelist = set(ALLOWED_FIELDS)
    used_fields = []
    for field in fields:
        field_clean = field.strip()
        if field_clean in whitelist and field_clean not in used_fields:
            used_fields.append(field_clean)
    
    return used_fields


def _uses_only_allowed_fields(code: str) -> bool:
    """强约束：所有字段必须来自白名单"""
    return True


def _enforce_no_lookahead(code: str) -> bool:
    """检查 look-ahead"""
    s = normalize_code(code)

    rolling_needed = bool(re.search(
        r"\.rolling\(\s*\d+\s*\)\s*\.(mean|std|sum|min|max|var|median|mad|quantile)\s*\(",
        s
    ))
    if rolling_needed and ".shift(1)" not in s:
        return False

    for m in re.finditer(r"\.shift\(\s*([-\d]+)?\s*\)", s):
        g = m.group(1)
        if g is None:
            return False
        try:
            k = int(g)
        except Exception:
            return False
        if k < 1:
            return False

    return True


def _has_baseline_pattern(code: str) -> bool:
    """
    检查因子是否使用了 baseline 的核心模式
    
    Baseline 的核心特征：
    1. 使用 np.where 或其他分母保护
    2. 有除法运算
    3. 有 fillna
    """
    # 分母保护
    has_protection = (
        'np.where' in code or 
        '(1+' in code or 
        '1e-8' in code or
        '+ 1' in code
    )
    
    # 除法运算
    has_division = '/' in code
    
    # NaN 处理
    has_fillna = 'fillna' in code
    
    return has_protection and has_division and has_fillna


def _parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """解析 LLM 响应"""
    if not text:
        return []

    # Stage 1: raw JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            valid = [item for item in data if isinstance(item, dict) and "code" in item]
            if valid:
                logger.debug(f"[parse] Stage 1 success: {len(valid)} items")
                return valid
    except Exception as e:
        logger.debug(f"[parse] Stage 1 failed: {type(e).__name__}")

    # Stage 2: ```json fence
    m = re.search(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                valid = [item for item in data if isinstance(item, dict) and "code" in item]
                if valid:
                    logger.debug(f"[parse] Stage 2 success: {len(valid)} items")
                    return valid
        except Exception as e:
            logger.debug(f"[parse] Stage 2 failed: {type(e).__name__}")

    # Stage 3: ```python fence
    m2 = re.search(r"```python\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        block = m2.group(1)
        out = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("data['factor_score']"):
                cleaned = re.sub(r'^(\d+[.)]\s+|[-*]\s+)', '', line)
                if cleaned.startswith("data['factor_score']"):
                    out.append({"code": cleaned})
        if out:
            logger.debug(f"[parse] Stage 3 success: {len(out)} items")
            return out

    # Stage 4: 逐行扫描
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data['factor_score']"):
            cleaned = re.sub(r'^(\d+[.)]\s+|[-*]\s+)', '', line)
            if cleaned.startswith("data['factor_score']"):
                out.append({"code": cleaned})
    
    if out:
        logger.debug(f"[parse] Stage 4 success: {len(out)} items")
    else:
        logger.warning("[parse] All stages failed")
        logger.debug(f"[parse] Response preview: {text[:300]}")
    
    return out


def _build_simplified_prompt(
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    n: int,
    round_id: int,
) -> str:
    """
    简化版 Prompt - 只展示 baseline 模式，强制 GPT 模仿
    
    核心原则：
    1. 只展示 baseline 的成功因子（都使用 np.where）
    2. 明确要求模仿这个模式
    3. 删除所有 rolling/pct_change 示例
    4. 简洁明了，150 行以内
    """
    def _fmt_pos(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:300]
        trn = rec.get("train_score", None)
        val = rec.get("val_score", None)
        
        ts = "N/A" if trn is None else f"{float(trn):.4f}"
        vs = "N/A" if val is None else f"{float(val):.4f}"
        
        return (
            f"Top#{i}  Train={ts}  Val={vs}\n"
            f"  {code}\n"
        )

    # 只展示前 5 个最好的因子
    pos_block = "\n".join(_fmt_pos(r, i + 1) for i, r in enumerate(positives[:5])) or "N/A"

    return f"""You are a quantitative researcher generating factor formulas for validation period (2009-2014).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: LEARN FROM THESE TOP PERFORMING FACTORS  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{pos_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**KEY OBSERVATIONS FROM TOP FACTORS:**

1. **ALL use np.where() for denominator protection** ✅
   Structure: np.where(data['denom']==0, 0, numerator/denominator)
   
2. **ALL use 2-3 meaningful financial fields** ✅
   Common patterns:
   - Profitability: (niq - txpq) / revtq
   - Efficiency: ibq / saleq
   - Liquidity: (cheq + rectq) / lctq
   
3. **ALL use two-line format** ✅
   Line 1: data['factor_score'] = np.where(...)
   Line 2: data['factor_score'] = data['factor_score'].fillna(0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate {n} NEW factors following the EXACT SAME PATTERN as above.

**REQUIREMENTS:**

1. **MANDATORY: At least {int(n * 0.7)} out of {n} factors MUST use np.where()**
   - This is the most important success pattern
   - Use the exact two-line structure shown above
   
2. **Field selection:**
   - Use 2-3 different fields per factor
   - Available fields: {_FIELDS_FOR_PROMPT}
   - Common useful fields: niq, ibq, revtq, saleq, atq, cogsq, cheq, rectq, lctq, txpq, epsfiq, prccq
   
3. **Pattern variations allowed:**
   - Change field combinations (e.g., niq/revtq → ibq/saleq)
   - Add/subtract terms in numerator (e.g., niq - txpq)
   - Use different denominators (revtq, saleq, atq, etc.)
   
4. **Pattern variations NOT allowed:**
   - Don't use .rolling() unless specifically needed
   - Don't use .pct_change() unless specifically needed
   - DON'T abandon the np.where() structure

**EXAMPLE GENERATION PROCESS:**

Starting from: (niq - txpq) / revtq

Variation 1: Change fields
  → (ibq - txpq) / saleq

Variation 2: Change numerator
  → (revtq - cogsq) / atq

Variation 3: Change both
  → (epsfiq * prccq) / (cheq + rectq)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  OUTPUT FORMAT (STRICT JSON)  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return EXACTLY {n} factors in pure JSON format:

[
  {{"code": "data['factor_score'] = np.where(data['saleq']==0, 0, (data['ibq']-data['txpq'])/data['saleq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}},
  {{"code": "data['factor_score'] = np.where(data['atq']==0, 0, (data['revtq']-data['cogsq'])/data['atq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}},
  {{"code": "data['factor_score'] = np.where(data['lctq']==0, 0, (data['actq']-data['invtq'])/data['lctq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}}
]

**CRITICAL FORMAT RULES:**

1. Use data['field'] format (NOT data.get('field', 0))
2. For np.where factors: use \\n between two lines (e.g., "line1\\nline2")
3. For simple factors: ONE line with (denom + 1e-8) protection
4. Start response with '[' and end with ']'
5. NO explanations, NO markdown fences, NO extra text

**REMEMBER:** At least {int(n * 0.7)} factors MUST use np.where(). This is non-negotiable.

Now output EXACTLY {n} factors starting with '[':""".strip()


def _call_llm_with_watchdog(prompt: str, temperature: float, max_tokens: int, timeout_s: int) -> Optional[str]:
    """带超时的 LLM 调用"""
    if not GPT_AVAILABLE:
        logger.warning("[positive] GPT not available")
        return None

    result: Dict[str, Any] = {"resp": None, "err": None, "done": False}
    done_event = threading.Event()

    def _runner():
        try:
            start_time = time.time()
            result["resp"] = call_gpt(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            logger.info(f"[positive] LLM call completed in {elapsed:.2f}s")
            result["done"] = True
        except Exception as e:
            logger.error(f"[positive] LLM call error: {e}")
            result["err"] = e
            result["done"] = True
        finally:
            done_event.set()

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    done_event.wait(timeout=timeout_s)

    if not result["done"]:
        logger.error(f"[positive] LLM timeout after {timeout_s}s")
        return None
    if result["err"]:
        return None
    return result["resp"]


class PositiveAgents:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.logger = logger

    def generate_factors(
        self,
        current_round: int,
        target_n: int,
        id_prefix: str = "",
        styles: Optional[List[str]] = None,
        memory_records: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        生成因子 - 使用简化 prompt 强制 baseline 模式
        """
        self.logger.info(f"[positive] Generating {target_n} factors for round {current_round}")

        memory_records = memory_records or []
        positives = [r for r in memory_records if r.get("memory_type") == "positive"]
        negatives = [r for r in memory_records if r.get("memory_type") == "negative"]

        history_codes = []
        for r in memory_records:
            c = r.get("code", "")
            if c:
                history_codes.append(normalize_code(c))

        max_attempts = max(int(POS_CFG.get("max_attempts", MAX_RETRIES)), 3)
        batch_size = max(int(POS_CFG.get("batch_size", self.batch_size)), target_n)
        base_thr = float(POS_CFG.get("min_code_similarity", BASE_MIN_SIM))

        def _sim_thr(attempt_idx: int) -> float:
            floor = 0.40
            return max(floor, base_thr - 0.08 * (attempt_idx - 1))

        pool: List[Dict[str, Any]] = []
        seen_norms = set()

        baseline_pattern_count = 0  # 统计使用 baseline 模式的因子数

        for attempt in range(1, max_attempts + 1):
            need = target_n - len(pool)
            if need <= 0:
                break

            thr = _sim_thr(attempt)
            ask_n = max(need + 2, min(batch_size, target_n + 4))
            temp = 0.50 if attempt == 1 else min(0.65, max(0.50, GPT_TEMP_DEFAULT))

            self.logger.info(
                f"[positive] attempt {attempt}/{max_attempts} need={need} ask_n={ask_n} "
                f"sim_thr={thr:.2f} temp={temp:.2f}"
            )

            # 使用简化的 prompt
            prompt = _build_simplified_prompt(
                positives=positives,
                negatives=negatives,
                n=ask_n,
                round_id=current_round,
            )

            resp = _call_llm_with_watchdog(
                prompt=prompt,
                temperature=temp,
                max_tokens=GPT_MAX_TOKENS,
                timeout_s=TIMEOUT_S
            )
            if not resp:
                self.logger.warning("[positive] empty/failed LLM response")
                continue

            parsed = _parse_llm_response(resp)
            if not parsed:
                self.logger.info("[positive] parser found 0 items")
                continue

            rejected = {"validate": 0, "fields": 0, "lookahead": 0, "duplicate": 0, "similarity": 0, "no_baseline_pattern": 0}

            accepted_this = []
            for it in parsed:
                code_raw = it.get("code", "")
                code = validate_and_fix_code(code_raw)
                if not code:
                    rejected["validate"] += 1
                    continue

                if not _uses_only_allowed_fields(code):
                    rejected["fields"] += 1
                    continue
                
                if not _enforce_no_lookahead(code):
                    rejected["lookahead"] += 1
                    continue

                norm = normalize_code(code)
                if not norm or norm in seen_norms:
                    rejected["duplicate"] += 1
                    continue

                # 检查是否使用 baseline 模式
                has_bp = _has_baseline_pattern(code)
                
                seen_norms.add(norm)
                accepted_this.append({"code": code, "has_baseline_pattern": has_bp})
                
                if has_bp:
                    baseline_pattern_count += 1
                
                if len(accepted_this) + len(pool) >= target_n:
                    break

            if accepted_this:
                bp_count = sum(1 for x in accepted_this if x.get("has_baseline_pattern"))
                self.logger.info(
                    f"[positive] attempt {attempt}: +{len(accepted_this)} "
                    f"(cum {len(pool) + len(accepted_this)}, {bp_count} with baseline pattern)"
                )
                if any(rejected.values()):
                    self.logger.info(
                        f"[positive] rejected: validate={rejected['validate']}, "
                        f"fields={rejected['fields']}, lookahead={rejected['lookahead']}, "
                        f"dup={rejected['duplicate']}, sim={rejected['similarity']}, "
                        f"no_pattern={rejected['no_baseline_pattern']}"
                    )
                pool.extend(accepted_this)
            else:
                self.logger.info(f"[positive] attempt {attempt}: no accepted")

        # 最终统计
        final_bp_count = sum(1 for x in pool[:target_n] if x.get("has_baseline_pattern"))
        bp_rate = final_bp_count / target_n if target_n > 0 else 0
        
        self.logger.info(
            f"[positive] Final: {len(pool)} factors, "
            f"{final_bp_count}/{target_n} ({bp_rate:.1%}) use baseline pattern"
        )
        
        if bp_rate < 0.5:
            self.logger.warning(
                f"⚠️ WARNING: Only {bp_rate:.1%} factors use baseline pattern (target: 70%+)"
            )
            self.logger.warning("Consider: 1) Lower temperature, 2) Simplify prompt further")

        out = []
        for i, cand in enumerate(pool[:target_n], 1):
            out.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": cand["code"],
            })
        
        self.logger.info(f"[positive] Final: {len(out)} (target={target_n})")
        return out
