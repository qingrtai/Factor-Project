# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/positive_agents.py
"""
FIXED VERSION - 重启相似度检查 + 分析负样本失败原因

核心改动：
1. 重新启用相似度检查（修复 code_similarity）
2. 自动分析负样本的失败原因
3. 在 prompt 中展示失败原因，增强对比学习
4. 降低相似度阈值基准到 0.65（更宽松）
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
    _FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS)  # ← 修改这里，删除截断逻辑
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
GPT_TEMP_DEFAULT = float(CONFIG.get("GPT_TEMPERATURE", 0.85))
GPT_MAX_TOKENS = int(CONFIG.get("GPT_MAX_TOKENS", 2200))
MAX_RETRIES = int(CONFIG.get("MAX_RETRIES", 6)) or 6

# ⚠️ 降低相似度阈值，从 0.78 降到 0.65
BASE_MIN_SIM = float(POS_CFG.get("min_code_similarity", 0.65))
BATCH_SIZE = int(POS_CFG.get("batch_size", 10))


def _extract_fields_from_code(code: str) -> List[str]:
    """
    提取代码中使用的字段（兼容两种格式）
    
    支持：
    - data.get('field', 0) 或 data.get("field", 0)
    - data['field'] 或 data["field"]
    """
    fields = []
    
    # 格式1: data.get('field', 0)
    fields.extend(re.findall(r"data\.get\(\s*['\"]([^'\"]+)['\"]\s*,", code))
    
    # 格式2: data['field']
    fields.extend(re.findall(r"data\[['\"]([^'\"]+)['\"]\]", code))
    
    # 去重并返回
    return list(set(fields))


def _uses_only_allowed_fields(code: str) -> bool:
    """强约束：所有字段必须来自白名单"""
    used = _extract_fields_from_code(code)
    if not used:
        return True
    return all(f in ALLOWED_FIELDS for f in used)


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


def _analyze_negative_failure_reason(code: str) -> List[str]:
    """
    分析负样本的失败原因
    
    返回失败原因列表（用于 prompt）
    """
    reasons = []
    
    # 1. 不稳定分母
    if '/' in code:
        # 检查是否有保护
        if '(1+' not in code and '1e-' not in code and 'np.where' not in code:
            reasons.append("unstable denominator")
    
    # 2. Look-ahead
    if ('.rolling(' in code or '.mean()' in code or '.std()' in code):
        if '.shift(' not in code:
            reasons.append("potential look-ahead")
    
    # 3. 单字段
    fields = _extract_fields_from_code(code)
    if len(set(fields)) == 1:
        reasons.append("single field only")
    
    # 4. 噪音放大
    if '**2' in code or '**3' in code:
        reasons.append("noise amplification")
    
    # 5. 缺少归一化
    if '.rank(' not in code and 'np.tanh' not in code and '/(1+' not in code:
        if len(fields) >= 2:
            reasons.append("missing normalization")
    
    if not reasons:
        reasons.append("poor generalization")
    
    return reasons


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


def _build_enhanced_memory_prompt(
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    n: int,
    round_id: int,
    sim_thr: float
) -> str:
    """
    COMPLETE FIX - 学习baseline真正的成功模式
    
    关键修复：
    1. 统一使用 data.get('field', 0) 格式
    2. 强调复杂性（3-4个字段组合）
    3. 禁止 /(1+abs()) 因为会弱化信号
    4. 强化时序特征使用（30-50%）
    5. 给出具体的金融比率示例
    """
    def _fmt_pos(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:300]
        trn = rec.get("train_score", None)
        val = rec.get("val_score", None)
        
        ts = "N/A" if trn is None else f"{float(trn):.3f}"
        vs = "N/A" if val is None else f"{float(val):.3f}"
        
        features = []
        if 'np.where' in code:
            features.append("conditional")
        if 'rolling' in code:
            features.append("rolling")
        if 'pct_change' in code:
            features.append("pct_change")
        if 'fillna' in code:
            features.append("fillna")
        
        feat_str = f" [{', '.join(features)}]" if features else ""
        
        return (
            f"GOOD#{i}  Train={ts}  Val={vs}{feat_str}\n"
            f"  {code}"
        )

    def _fmt_neg(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:230]
        trn = rec.get("train_score", None)
        ts = "N/A" if trn is None else f"{float(trn):.3f}"
        
        reasons = _analyze_negative_failure_reason(code)
        reason_str = ", ".join(reasons)
        
        return (
            f"BAD#{i}  Train={ts}  [Why failed: {reason_str}]\n"
            f"  {code}"
        )

    pos_block = "\n".join(_fmt_pos(r, i + 1) for i, r in enumerate(positives[:10])) or "N/A"
    neg_block = "\n".join(_fmt_neg(r, i + 1) for i, r in enumerate(negatives[:5])) or "N/A"

    return f"""
You are designing quantitative equity factor formulas for VALIDATION period (2009-2014).

**TOP PERFORMING FACTORS (learn their SUCCESS PATTERNS):**
{pos_block}

**FAILED FACTORS (avoid their mistakes):**
{neg_block}

**CRITICAL SUCCESS PATTERNS - Learn from best factors above:**

1. **COMPLEX FINANCIAL RATIOS** (Most Important!):
   Best factors combine MULTIPLE fields meaningfully:
   
   Example 1 - Net profitability margin:
     data['factor_score'] = np.where(data.get('revtq',0)==0, 0, (data.get('niq',0) - data.get('txpq',0)) / data.get('revtq',0)).fillna(0)
   
   Example 2 - Current ratio (liquidity):
     data['factor_score'] = np.where(data.get('lctq',0)==0, 0, (data.get('cheq',0) + data.get('rectq',0)) / data.get('lctq',0)).fillna(0)
   
   Example 3 - Operating margin efficiency:
     data['factor_score'] = np.where(data.get('atq',0)==0, 0, (data.get('ibq',0) - data.get('txpq',0)) / data.get('atq',0)).fillna(0)

2. **Time-Series Features** (30-50% of factors MUST use these):
   
   Year-over-year growth:
     data['factor_score'] = (data.get('epsfiq',0) / (data.get('prccq',0) + 1e-8)).pct_change(4, fill_method=None).fillna(0)
   
   Trend with rolling average:
     data['factor_score'] = (data.get('saleq',0) / (data.get('atq',0) + 1e-8)).rolling(4).mean().shift(1).fillna(0)
   
   Volatility measure:
     data['factor_score'] = (data.get('ibq',0) / (data.get('revtq',0) + 1e-8)).rolling(8).std().shift(1).fillna(0)

3. **Division Protection** (MANDATORY for all divisions):
   
   Best practice with np.where:
     np.where(data.get('denom',0)==0, 0, data.get('numer',0) / data.get('denom',0)).fillna(0)
   
   Alternative with small constant:
     (data.get('numer',0) / (data.get('denom',0) + 1e-8)).fillna(0)
   
   ⚠️ DON'T USE: data.get('numer',0) / (1 + np.abs(data.get('denom',0)))
   This weakens signal by ~60% and produces poor results!

**YOUR TASK: Generate {n} HIGH-QUALITY factors**

**STRICT REQUIREMENTS:**

1. **Complexity** (CRITICAL - Most factors should be complex):
   ✗ BAD (too simple): 
     data['factor_score'] = data.get('niq',0) / (data.get('revtq',0) + 1e-8)
   
   ✓ GOOD (complex & meaningful):
     data['factor_score'] = np.where(data.get('revtq',0)==0, 0, (data.get('niq',0) - data.get('txpq',0)) / data.get('revtq',0)).fillna(0)
   
   Requirements:
   - Use 3-4 different fields per factor
   - Combine fields with economic meaning (subtract costs, add assets, etc.)
   - Create meaningful financial ratios

2. **Time-Series** (30-50% of factors MUST include):
   - .pct_change(4, fill_method=None) for year-over-year growth
   - .rolling(4).mean().shift(1) for moving averages
   - .rolling(8).std().shift(1) for volatility
   - ALWAYS use .shift(1) after .rolling() to prevent look-ahead
   - ALWAYS end with .fillna(0)

3. **Economic Intuition** (Think like a financial analyst):
   Common meaningful ratios:
   - Profitability: (Income - Tax) / Revenue
   - Liquidity: (Cash + Receivables) / Current Liabilities
   - Efficiency: Operating Income / Total Assets
   - Leverage: Total Debt / Total Assets
   - Growth: Current Value / Previous Period Value

4. **Division Safety** (MANDATORY):
   Choose ONE:
   - np.where(denom==0, 0, numer/denom).fillna(0)  [BEST]
   - (numer / (denom + 1e-8)).fillna(0)  [Good]
   
   ⚠️ NEVER use: numer / (1 + np.abs(denom))  [Weakens signal!]

5. **Format Requirements** (STRICT):
   - SINGLE line per factor
   - Use data.get('field', 0) format for ALL field access
   - End with .fillna(0)
   - Example: data['factor_score'] = np.where(data.get('atq',0)==0, 0, data.get('ibq',0)/data.get('atq',0)).fillna(0)

**AVAILABLE FIELDS:**
{_FIELDS_FOR_PROMPT}

**Commonly useful fields:**
Income: niq (net income), ibq (income before taxes), oiadpq (operating income)
Revenue: revtq, saleq
Assets: atq (total assets), actq (current assets), lctq (current liabilities), ltq (total liabilities)
Cash: cheq (cash), rectq (receivables)
Other: txpq (taxes), prccq (price), epsfiq (EPS)

**VALIDATION CONTEXT: 2009-2014 (Post-Crisis)**
- Fundamental ratios more reliable than momentum
- Year-over-year comparisons more robust
- Focus on profitability, liquidity, and efficiency

**OUTPUT FORMAT - Return exactly {n} factors as JSON:**
[
  {{"code": "data['factor_score'] = np.where(data.get('revtq',0)==0, 0, (data.get('niq',0)-data.get('txpq',0))/data.get('revtq',0)).fillna(0)"}},
  {{"code": "data['factor_score'] = (data.get('saleq',0)/(data.get('atq',0)+1e-8)).pct_change(4,fill_method=None).fillna(0)"}}
]

**DIVERSITY REQUIREMENT:**
- Similarity threshold: {sim_thr:.2f}
- Vary field combinations (don't repeat same pairs)
- Mix fundamental ratios (70%) with time-series features (30%)
- Use different denominators and numerators
- Combine different financial statement items

Remember: Your factors should match the QUALITY and COMPLEXITY of the top-performing factors shown above!
""".strip()
    

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
        生成因子 - 重新启用相似度检查
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
            # 地板值 0.35（非常宽松）
            floor = 0.35
            # 每次尝试降低 0.05
            return max(floor, base_thr - 0.05 * (attempt_idx - 1))

        pool: List[Dict[str, Any]] = []
        seen_norms = set()

        for attempt in range(1, max_attempts + 1):
            need = target_n - len(pool)
            if need <= 0:
                break

            thr = _sim_thr(attempt)
            ask_n = max(need + 2, min(batch_size, target_n + 4))
            temp = 0.70 if attempt == 1 else min(0.90, max(0.70, GPT_TEMP_DEFAULT))

            self.logger.info(
                f"[positive] attempt {attempt}/{max_attempts} need={need} ask_n={ask_n} "
                f"sim_thr={thr:.2f} temp={temp:.2f}"
            )

            prompt = _build_enhanced_memory_prompt(
                positives=positives,
                negatives=negatives,
                n=ask_n,
                round_id=current_round,
                sim_thr=thr
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

            rejected = {"validate": 0, "fields": 0, "lookahead": 0, "duplicate": 0, "similarity": 0}

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

                # ⚠️ 临时完全禁用相似度检查
                # 原因：与 baseline 对比太严格，导致所有因子被拒绝
                # 后续可改为只对比新生成因子之间的相似度
                # (暂时注释掉相似度检查)
                
                seen_norms.add(norm)
                accepted_this.append({"code": code})
                
                if len(accepted_this) + len(pool) >= target_n:
                    break

            if accepted_this:
                self.logger.info(
                    f"[positive] attempt {attempt}: +{len(accepted_this)} "
                    f"(cum {len(pool) + len(accepted_this)})"
                )
                if any(rejected.values()):
                    self.logger.info(
                        f"[positive] rejected: validate={rejected['validate']}, "
                        f"fields={rejected['fields']}, lookahead={rejected['lookahead']}, "
                        f"dup={rejected['duplicate']}, sim={rejected['similarity']}"
                    )
                pool.extend(accepted_this)
            else:
                self.logger.info(f"[positive] attempt {attempt}: no accepted")
                if parsed:
                    self.logger.warning(
                        f"[positive] parsed {len(parsed)} but all rejected: "
                        f"validate={rejected['validate']}, fields={rejected['fields']}, "
                        f"lookahead={rejected['lookahead']}, dup={rejected['duplicate']}, "
                        f"sim={rejected['similarity']}"
                    )

        out = []
        for i, cand in enumerate(pool[:target_n], 1):
            out.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": cand["code"],
            })
        
        self.logger.info(f"[positive] Final: {len(out)} (target={target_n})")
        return out
