# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/positive_agents.py
"""
FIXED VERSION V3 - 强制 np.where，禁止 (denom + 1e-8)

核心改动：
1. 完全删除 (denom + 1e-8) 的示例
2. 100% 强制使用 np.where（不是 70%）
3. 在代码验证前就检查 np.where
4. 如果没有 np.where 直接拒绝
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
GPT_TEMP_DEFAULT = float(CONFIG.get("GPT_TEMPERATURE", 0.40))  # 进一步降低到 0.40
GPT_MAX_TOKENS = int(CONFIG.get("GPT_MAX_TOKENS", 2200))
MAX_RETRIES = int(CONFIG.get("MAX_RETRIES", 6)) or 6

BASE_MIN_SIM = float(POS_CFG.get("min_code_similarity", 0.70))
BATCH_SIZE = int(POS_CFG.get("batch_size", 10))


def _must_have_np_where(code: str) -> bool:
    """
    强制检查：代码必须包含 np.where
    
    这是最关键的检查，在所有其他检查之前
    """
    return 'np.where' in code


def _validate_two_line_format(code: str) -> bool:
    """
    验证是否为正确的两行格式
    
    正确格式：
    data['factor_score'] = np.where(...)
    data['factor_score'] = data['factor_score'].fillna(0)
    """
    lines = code.strip().split('\n')
    
    if len(lines) != 2:
        return False
    
    line1 = lines[0].strip()
    line2 = lines[1].strip()
    
    # 第一行必须有 np.where
    if 'np.where' not in line1:
        return False
    
    # 第二行必须有 fillna
    if 'fillna' not in line2:
        return False
    
    return True


def _simple_validate(code: str) -> Optional[str]:
    """
    简化的代码验证（绕过 validate_and_fix_code）
    
    只做基本检查：
    1. 必须有 np.where
    2. 必须是两行格式
    3. 第一行赋值，第二行 fillna
    """
    if not code:
        return None
    
    # 强制检查 np.where
    if not _must_have_np_where(code):
        return None
    
    # 检查两行格式
    if not _validate_two_line_format(code):
        return None
    
    # 基本语法检查
    if "data['factor_score']" not in code:
        return None
    
    # 检查是否有明显的语法错误
    if code.count('(') != code.count(')'):
        return None
    
    if code.count('[') != code.count(']'):
        return None
    
    return code


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


def _build_ultra_strict_prompt(
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    n: int,
    round_id: int,
) -> str:
    """
    超严格版 Prompt - 100% 强制 np.where，禁止 (denom + 1e-8)
    """
    def _fmt_pos(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:300]
        trn = rec.get("train_score", None)
        val = rec.get("val_score", None)
        
        ts = "N/A" if trn is None else f"{float(trn):.4f}"
        vs = "N/A" if val is None else f"{float(val):.4f}"
        
        return (
            f"#{i}  Train={ts}  Val={vs}\n"
            f"{code}\n"
        )

    # 展示所有10个因子
    pos_block = "\n".join(_fmt_pos(r, i + 1) for i, r in enumerate(positives))

    # ========== 新增：负样本展示 ========== #
    neg_block = ""
    if negatives:
        def _fmt_neg(rec: Dict[str, Any], i: int) -> str:
            """格式化负样本"""
            code = str(rec.get("code", ""))[:250]
            trn = rec.get("train_score", None)
            ts = "N/A" if trn is None else f"{float(trn):.4f}"
            return f"BAD #{i}  Train={ts}\n{code}\n"
        
        neg_lines = "\n".join(_fmt_neg(r, i + 1) for i, r in enumerate(negatives))
        
        neg_block = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ❌ FAILED FACTORS (AVOID THESE PATTERNS - LOW TRAIN SCORES)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    {neg_lines}
    
    These factors have LOW train scores and demonstrate common pitfalls:
    - Missing denominator checks (division by zero)
    - Poor normalization
    - Noise amplification
    - Weak economic intuition
    
    DO NOT copy these patterns. Learn what to AVOID.
    """
    # ========================================= #
    
    return f"""You are generating factor formulas. Follow these examples EXACTLY:
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TOP PERFORMING FACTORS (COPY THEIR STRUCTURE)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    {pos_block}
    {neg_block}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️  CRITICAL: MANDATORY STRUCTURE (NO EXCEPTIONS)  ⚠️
    
    EVERY factor MUST follow this EXACT 2-line structure:
    
    Line 1: data['factor_score'] = np.where(data['DENOM']==0, 0, EXPRESSION/data['DENOM'])
    Line 2: data['factor_score'] = data['factor_score'].fillna(0)
    
    Where:
    - DENOM = denominator field (revtq, saleq, atq, etc.)
    - EXPRESSION = numerator (can be: single field, sum, difference, etc.)
    
    EXAMPLES OF VALID VARIATIONS:
    
    Example 1 (difference in numerator):
    data['factor_score'] = np.where(data['revtq']==0, 0, (data['niq']-data['txpq'])/data['revtq'])
    data['factor_score'] = data['factor_score'].fillna(0)
    
    Example 2 (sum in numerator):
    data['factor_score'] = np.where(data['lctq']==0, 0, (data['cheq']+data['rectq'])/data['lctq'])
    data['factor_score'] = data['factor_score'].fillna(0)
    
    Example 3 (single field in numerator):
    data['factor_score'] = np.where(data['saleq']==0, 0, data['ibq']/data['saleq'])
    data['factor_score'] = data['factor_score'].fillna(0)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ❌ ABSOLUTELY FORBIDDEN (WILL BE REJECTED)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. ONE-LINE formats like:
       data['factor_score'] = (data['niq']-data['txpq']) / (data['revtq'] + 1e-8)
       ❌ WRONG - missing np.where, missing second line
    
    2. Using (denom + 1e-8) instead of np.where:
       data['factor_score'] = EXPRESSION / (data['DENOM'] + 1e-8)
       ❌ WRONG - must use np.where
    
    3. Any format without np.where:
       ❌ REJECTED IMMEDIATELY
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📋 YOUR TASK
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Generate {n} factors using the MANDATORY 2-line np.where structure above.
    
    REQUIREMENTS:
    
    1. **Structure** (100% compliance):
       - Line 1: MUST start with "data['factor_score'] = np.where("
       - Line 2: MUST be "data['factor_score'] = data['factor_score'].fillna(0)"
       
    2. **Fields**:
       - Available: {_FIELDS_FOR_PROMPT}
       - Use 2-3 fields per factor
       - Common: niq, ibq, revtq, saleq, atq, cogsq, cheq, rectq, lctq, txpq
       
    3. **Variations** (change these, keep structure):
       - Numerator: can be niq-txpq, ibq+cogsq, single field, etc.
       - Denominator: revtq, saleq, atq, lctq, etc.
       - Fields: use different combinations
    
    4. **Format** (CRITICAL):
       - Use \\n to separate two lines in JSON
       - Use data['field'] format (NOT data.get)
       - No markdown, no explanations, just JSON
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️  OUTPUT FORMAT (JSON ONLY)  ⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    [
      {{"code": "data['factor_score'] = np.where(data['saleq']==0, 0, (data['ibq']-data['txpq'])/data['saleq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}},
      {{"code": "data['factor_score'] = np.where(data['atq']==0, 0, (data['revtq']+data['niq'])/data['atq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}},
      {{"code": "data['factor_score'] = np.where(data['lctq']==0, 0, data['cheq']/data['lctq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}}
    ]
    
    Start output with '[' immediately. No explanations. Exactly {n} factors.
    
    REMEMBER: Every factor MUST have np.where. No exceptions. No (denom + 1e-8).""".strip()


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
        生成因子 - 强制 np.where，禁止 (denom + 1e-8)
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

        np_where_count = 0

        for attempt in range(1, max_attempts + 1):
            need = target_n - len(pool)
            if need <= 0:
                break

            thr = _sim_thr(attempt)
            ask_n = max(need + 2, min(batch_size, target_n + 4))
            temp = 0.40 if attempt == 1 else min(0.50, max(0.40, GPT_TEMP_DEFAULT))

            self.logger.info(
                f"[positive] attempt {attempt}/{max_attempts} need={need} ask_n={ask_n} "
                f"sim_thr={thr:.2f} temp={temp:.2f}"
            )

            # 使用超严格的 prompt
            prompt = _build_ultra_strict_prompt(
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

            rejected = {
                "no_np_where": 0,
                "wrong_format": 0,
                "lookahead": 0,
                "duplicate": 0,
                "similarity": 0
            }

            accepted_this = []
            for it in parsed:
                code_raw = it.get("code", "")
                
                # ========== 第一关：必须有 np.where ========== #
                if not _must_have_np_where(code_raw):
                    rejected["no_np_where"] += 1
                    continue
                
                # ========== 第二关：使用简化验证 ========== #
                code = _simple_validate(code_raw)
                if not code:
                    rejected["wrong_format"] += 1
                    continue
                
                # ========== 第三关：检查 look-ahead ========== #
                if not _enforce_no_lookahead(code):
                    rejected["lookahead"] += 1
                    continue

                # ========== 第四关：去重 ========== #
                norm = normalize_code(code)
                if not norm or norm in seen_norms:
                    rejected["duplicate"] += 1
                    continue

                # ========== 通过所有检查 ========== #
                seen_norms.add(norm)
                accepted_this.append({"code": code})
                np_where_count += 1
                
                if len(accepted_this) + len(pool) >= target_n:
                    break

            if accepted_this:
                self.logger.info(
                    f"[positive] attempt {attempt}: +{len(accepted_this)} "
                    f"(cum {len(pool) + len(accepted_this)}, all with np.where ✅)"
                )
                if any(rejected.values()):
                    self.logger.info(
                        f"[positive] rejected: no_np_where={rejected['no_np_where']}, "
                        f"wrong_format={rejected['wrong_format']}, "
                        f"lookahead={rejected['lookahead']}, "
                        f"dup={rejected['duplicate']}, sim={rejected['similarity']}"
                    )
                pool.extend(accepted_this)
            else:
                self.logger.info(f"[positive] attempt {attempt}: no accepted")
                if parsed:
                    self.logger.warning(
                        f"[positive] parsed {len(parsed)} but all rejected: "
                        f"no_np_where={rejected['no_np_where']}, "
                        f"wrong_format={rejected['wrong_format']}, "
                        f"lookahead={rejected['lookahead']}, "
                        f"dup={rejected['duplicate']}"
                    )

        # 最终统计
        np_where_rate = np_where_count / target_n if target_n > 0 else 0
        
        self.logger.info(
            f"[positive] Final: {len(pool)} factors, "
            f"{np_where_count}/{target_n} ({np_where_rate:.1%}) use np.where"
        )
        
        if np_where_rate < 1.0:
            self.logger.warning(
                f"⚠️ WARNING: Only {np_where_rate:.1%} factors use np.where (target: 100%)"
            )
            self.logger.warning("All factors MUST use np.where. Check prompt and GPT response.")

        out = []
        for i, cand in enumerate(pool[:target_n], 1):
            out.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": cand["code"],
            })
        
        self.logger.info(f"[positive] Final: {len(out)} (target={target_n})")
        return out
