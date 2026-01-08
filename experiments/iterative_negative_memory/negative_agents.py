# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/negative_agents.py
"""
FIXED VERSION - 使用 common pitfalls 策略而非 deliberately bad

核心改动：
1. 不再生成"故意坏"的因子
2. 改为生成"看起来合理但有常见陷阱"的因子
3. 明确列出 5 种失败模式
4. 增加分母保护要求
"""

from typing import List, Dict, Any, Optional
import logging
import re
import json

from shared.code_utils import (
    normalize_code,
    code_similarity,
    to_single_line,
    is_valid_single_line,
    extract_content
)

try:
    from common.gpt_runner import call_gpt
    GPT_AVAILABLE = True
except Exception:
    GPT_AVAILABLE = False

try:
    from .config import CONFIG, NEGATIVE_AGENT_CONFIG
except Exception:
    CONFIG = {}
    NEGATIVE_AGENT_CONFIG = {}

try:
    from common.column_desc import COLUMN_DESC
    ALLOWED_FIELDS: List[str] = list(COLUMN_DESC.keys())
    _FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS) if len(ALLOWED_FIELDS) <= 200 else (", ".join(ALLOWED_FIELDS[:200]) + ", ...")
except Exception as e:
    raise ImportError(f"[negative_agents] Failed to import COLUMN_DESC: {e}")

logger = logging.getLogger(__name__)

NEG_CONFIG = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)
NEG_TEMPERATURE_FIRST = float(NEG_CONFIG.get("temperature_first", 0.85))
NEG_TEMPERATURE_LATER = float(NEG_CONFIG.get("temperature_later", 0.95))
MAX_TOTAL_NEG_CALLS = int(NEG_CONFIG.get("max_calls_per_round", 4))
NEG_OVERASK = int(NEG_CONFIG.get("overask", 3))
NEG_MIN_CODE_SIM = float(NEG_CONFIG.get("min_code_similarity", 0.80))
GPT_MAX_TOKENS = 2200
TIMEOUT_SECONDS = 90


def _parse_as_json_list(text: str) -> List[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for it in data:
                if isinstance(it, dict) and "code" in it:
                    out.append(str(it["code"]))
                elif isinstance(it, str):
                    out.append(it)
            return out
    except Exception:
        pass
    return []


def _parse_fenced(text: str, lang: str) -> str:
    import re as _re
    pattern = _re.compile(rf"```{lang}\s*(.*?)\s*```", _re.DOTALL | _re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1) if m else ""


def _parse_candidates(text: str) -> List[str]:
    if not text:
        return []
    
    # 1) raw JSON
    codes = _parse_as_json_list(text)
    if codes:
        return codes
    
    # 2) ```json
    j = _parse_fenced(text, "json")
    if j:
        codes = _parse_as_json_list(j)
        if codes:
            return codes
    
    # 3) ```python
    py = _parse_fenced(text, "python")
    if py:
        return [py]
    
    # 4) 行级扫描
    lines = []
    for ln in text.splitlines():
        if "factor_score" in ln:
            lines.append(ln.strip())
    return lines


class NegativeAgent:
    """负向代理：生成具有常见陷阱的因子（而非故意坏）"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        neg_config = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)
        self.temperature_first = float(neg_config.get("temperature_first", 0.85))
        self.temperature_later = float(neg_config.get("temperature_later", 0.95))
        self.max_calls_per_round = int(neg_config.get("max_calls_per_round", 4))
        
        self.logger.info(f"[negatives] ALLOWED_FIELDS loaded: {len(ALLOWED_FIELDS)} fields")

    def generate_negative_factors(
        self,
        current_round: int,
        target_n: int,
        context_positives: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """生成具有常见陷阱的因子"""
        context_positives = context_positives or []

        prompt = self._build_improved_prompt(
            current_round=current_round,
            need=target_n,
            positives=context_positives
        )

        collected: List[str] = []
        calls = 0

        while len(collected) < target_n and calls < self.max_calls_per_round:
            need = target_n - len(collected)
            ask = need + max(1, NEG_OVERASK)
            calls += 1
            temp = self.temperature_first if calls == 1 else self.temperature_later

            self.logger.info(f"[negatives] round={current_round} call#{calls} need={need} ask={ask} temp={temp:.2f}")

            if not GPT_AVAILABLE:
                self.logger.warning("[negatives] GPT not available")
                break
            
            try:
                try:
                    resp = call_gpt(prompt, temperature=temp, max_tokens=GPT_MAX_TOKENS, timeout=TIMEOUT_SECONDS)
                except TypeError:
                    resp = call_gpt(prompt, temperature=temp, max_tokens=GPT_MAX_TOKENS)
                text = extract_content(resp)
            except Exception as e:
                self.logger.error(f"[negatives] GPT error: {e}")
                continue

            raw_codes = _parse_candidates(text)
            accepted = []
            
            for raw in raw_codes:
                single = to_single_line(raw)
                if not single or not is_valid_single_line(single):
                    continue
                
                if any(code_similarity(single, x) >= NEG_MIN_CODE_SIM for x in collected):
                    continue
                
                accepted.append(single)

            if accepted:
                collected.extend([s for s in accepted if s not in collected])
                self.logger.info(f"[negatives] +{len(accepted)} accepted (now {len(collected)}/{target_n})")

        # 补齐（如果需要）
        if len(collected) < target_n:
            self.logger.warning(f"[negatives] only {len(collected)}/{target_n}")
            collected.extend(self._improved_bad_templates()[:(target_n - len(collected))])

        out = [{"code": c} for c in collected[:target_n]]
        self.logger.info(f"[negatives] final {len(out)}/{target_n}")
        return out

    def _build_improved_prompt(self, current_round: int, need: int, positives: List[Dict[str, Any]]) -> str:
        """
        改进版 prompt - 强调常见陷阱而非故意生成坏因子
        """
        def _fmt_pos(i: int, rec: Dict[str, Any]) -> str:
            code = str(rec.get("code", ""))[:220]
            ts = rec.get("train_score", None)
            ts_str = "N/A" if ts is None else f"{float(ts):.3f}"
            return f"GOOD#{i}  TrainScore={ts_str}\n  {code}"

        pos_block = "\n".join(_fmt_pos(i + 1, r) for i, r in enumerate(positives[:10])) or "N/A"

        prompt = f"""
You are a quantitative researcher generating factor formulas that exhibit **COMMON PITFALLS in factor investing**.

These factors should:
1. **Look reasonable at first glance** (proper syntax, use financial ratios)
2. **But contain subtle flaws** that lead to poor out-of-sample performance
3. Be syntactically valid and executable

Context (Training: 1961-2000, Validation: 2001-2010):

Top-performing factors from previous round (for contrast):
{pos_block}

Generate {need} factors that demonstrate **COMMON PITFALLS** (each category below):

**Pitfall Category 1: Unstable Denominators** (MUST include 1-2 factors)
- Division by variables that can be near-zero WITHOUT proper protection
- Example: `data['factor_score'] = data.get('niq',0) / data.get('ibq',0)`
  (WRONG - no protection when ibq is near zero)
- Note: This is a pitfall to demonstrate, not a best practice

**Pitfall Category 2: Poor Normalization** (MUST include 1-2 factors)
- Mixing variables with vastly different scales without normalization
- Missing rank transforms on skewed distributions
- Example: `data['factor_score'] = data.get('atq',0) - data.get('saleq',0) * 100`

**Pitfall Category 3: Overfitting Specific Periods** (MUST include 1 factor)
- Using very specific time windows that may not generalize
- Example: `data['factor_score'] = (data.get('saleq',0).rolling(13).mean() - data.get('saleq',0).rolling(17).mean()).shift(1)`

**Pitfall Category 4: Noise Amplification** (MUST include 1 factor)
- Using very short horizons without smoothing
- Squaring or cubing noisy variables
- Example: `data['factor_score'] = (data.get('rectq',0) - data.get('rectq',0).shift(1))**2`

**Pitfall Category 5: Weak Economic Intuition**
- Combining unrelated variables without clear rationale
- Using transformations that destroy signal
- Example: `data['factor_score'] = np.tanh(data.get('atq',0) + data.get('ltq',0))`

**CRITICAL - These are demonstrations of what NOT to do in real factors!**

**Constraints:**
- Use ONLY `data.get('<field>', 0)` with fields from: {_FIELDS_FOR_PROMPT}
- Time operations: `.shift(k)` with k>=1; `.rolling(w).mean()/.std()` MUST be followed by `.shift(1)`
- Allowed transforms: `np.tanh()`, `np.sign()`, `.rank(pct=True)`, `np.sqrt(np.abs())`, `/(1+np.abs())`
- NO imports, NO multiple statements, NO comments, NO backticks
- Each factor MUST demonstrate ONE of the above pitfalls clearly

**Return format (STRICT JSON):**
[
  {{"code": "data['factor_score'] = data.get('niq',0) / data.get('ibq',0)"}},
  {{"code": "data['factor_score'] = data.get('atq',0) - data.get('saleq',0) * 100"}}
]

Return exactly {need} factors as pure JSON (no explanation, no backticks).
Make sure to include factors from DIFFERENT pitfall categories for diversity.
""".strip()
        
        return prompt

    def _improved_bad_templates(self) -> List[str]:
        """改进的坏模板 - 明确的常见陷阱"""
        return [
            # Unstable denominator
            "data['factor_score'] = data.get('niq',0) / data.get('ibq',0)",
            # Poor normalization
            "data['factor_score'] = data.get('atq',0) - data.get('saleq',0) * 100",
            # Overfitting
            "data['factor_score'] = (data.get('saleq',0).rolling(13).mean() - data.get('saleq',0).rolling(17).mean()).shift(1)",
            # Noise amplification
            "data['factor_score'] = (data.get('rectq',0) - data.get('rectq',0).shift(1))**2",
            # Weak intuition
            "data['factor_score'] = np.tanh(data.get('atq',0) + data.get('ltq',0))",
        ]
