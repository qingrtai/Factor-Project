# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/positive_agents.py
"""
positive_agents.py — Train-only memory; validation-robust generation; no internal fallback.
（中文注释；英文 prompt）

要点（与全流程对齐）：
- Prompt 仅展示上一轮正样本与当轮负样本的 train_score + code；不泄露任何验证期数值。
- 目标是在验证集（2001–2015）稳健，而训练集（1961–2000）的 train_score 仅作方向与稳健性参考。
- 生成严格为“单行可执行”：
    data['factor_score'] = <expr>
  不允许 import / 多语句 / 注释 / 反引号。
- 字段白名单严格使用 column_desc.COLUMN_DESC 的键；并在解析阶段强制校验，防止生成不存在字段。
- 不在正向内部做 fallback；由 iterator 的外层循环“持续调用 GPT 直到凑满目标数”（或达全局上限）。

最终结果（给人看的）列顺序（不暴露给模型）：
round, train_score, val_score, diversity, coverage, ann_ret, sharpe, D, max_dd, autocorr, skew
其中 val_score = (sharpe + ann_ret + D)/3，且 D = 1 - max_drawdown；上述小分均为验证集口径。
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import re
import difflib
import time
import random
import os
import socket
import threading
import logging
import numpy as np

# ========== 导入共享工具 ========== #
from shared.code_utils import (
    normalize_code,
    code_similarity,
    validate_and_fix_code
)

logger = logging.getLogger(__name__)

# ============= 字段白名单：严格使用 column_desc =============
try:
    from common.column_desc import COLUMN_DESC
    ALLOWED_FIELDS: List[str] = list(COLUMN_DESC.keys())
    # 为避免 prompt 过长，仅在 prompt 中预览前 N 个字段名称
    _FIELDS_FOR_PROMPT = (
        ", ".join(ALLOWED_FIELDS) if len(ALLOWED_FIELDS) <= 160
        else (", ".join(ALLOWED_FIELDS[:160]) + ", ...")
    )
except Exception as e:
    raise ImportError(
        f"[positive] Unable to import COLUMN_DESC from common.column_desc: {e}"
    )

# ========= 随机种（非严格复现；严格复现请在 main/config 控制） =========
_local_seed = int(time.time() * 1000) % 1_000_000
np.random.seed(_local_seed)
random.seed(_local_seed)

# ============= GPT runner =============
try:
    from common.gpt_runner import call_gpt
    GPT_AVAILABLE = True
except Exception:
    GPT_AVAILABLE = False

# ============= Config =============
try:
    from .config import CONFIG
except Exception:
    CONFIG = {}

POS_CFG = CONFIG.get("POSITIVE_AGENT_CONFIG", {}) or {}

TIMEOUT_S = int(CONFIG.get("TIMEOUT", 90))
GPT_TEMP_DEFAULT = float(CONFIG.get("GPT_TEMPERATURE", 0.85))
GPT_MAX_TOKENS = int(CONFIG.get("GPT_MAX_TOKENS", 2200))
MAX_RETRIES = int(CONFIG.get("MAX_RETRIES", 6)) or 6
FORCE_LLM = bool(CONFIG.get("FORCE_LLM", True))

# Similarity & batch knobs
BASE_MIN_SIM = float(POS_CFG.get("min_code_similarity", 0.78))  # progressively relax to 0.75
BATCH_SIZE = int(POS_CFG.get("batch_size", 10))


# ===================== Utilities =====================

def _extract_fields_from_code(code: str) -> List[str]:
    """
    提取 code 中以 data.get('FIELD', ...) / data.get("FIELD", ...) 方式访问的字段名。
    """
    fields = re.findall(r"data\.get\(\s*['\"]([^'\"]+)['\"]\s*,", code)
    return fields


def _uses_only_allowed_fields(code: str) -> bool:
    """强约束：所有使用的字段必须来自 ALLOWED_FIELDS。"""
    used = _extract_fields_from_code(code)
    if not used:
        # 允许没有字段（纯常数/函数表达式也可以）
        return True
    return all(f in ALLOWED_FIELDS for f in used)


def _enforce_no_lookahead(code: str) -> bool:
    """
    近似静态检查：
    - 若出现 .rolling(...).(mean|std|sum|min|max|var|median|mad|quantile) 则必须在表达式中随后出现 .shift(1)
    - 所有 .shift(k) 要求 k>=1（未给数字或 k=0 视作不通过）
    """
    s = normalize_code(code)

    # 1) rolling 操作后必须 shift(1)
    rolling_needed = bool(re.search(
        r"\.rolling\(\s*\d+\s*\)\s*\.(mean|std|sum|min|max|var|median|mad|quantile)\s*\(",
        s
    ))
    if rolling_needed and ".shift(1)" not in s:
        return False

    # 2) shift 参数检查
    for m in re.finditer(r"\.shift\(\s*([-\d]+)?\s*\)", s):
        g = m.group(1)
        if g is None:
            return False  # 未明确给 k
        try:
            k = int(g)
        except Exception:
            return False
        if k < 1:
            return False

    return True


def _parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """
    解析 LLM 响应，优先 JSON 格式
    """
    if not text:
        return []

    # ========== Stage 1: 纯 JSON ========== #
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            valid = [item for item in data if isinstance(item, dict) and "code" in item]
            if valid:
                logger.debug(f"[parse] Stage 1 (raw JSON) success: {len(valid)} items")  # ← 添加
                return valid
    except Exception as e:
        logger.debug(f"[parse] Stage 1 (raw JSON) failed: {type(e).__name__}")  # ← 添加

    # ========== Stage 2: ```json fence ========== #
    m = re.search(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                valid = [item for item in data if isinstance(item, dict) and "code" in item]
                if valid:
                    logger.debug(f"[parse] Stage 2 (json fence) success: {len(valid)} items")  # ← 添加
                    return valid
        except Exception as e:
            logger.debug(f"[parse] Stage 2 (json fence) failed: {type(e).__name__}")  # ← 添加

    # ========== Stage 3: ```python fence（严格提取）========== #
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
            logger.debug(f"[parse] Stage 3 (python fence) success: {len(out)} items")  # ← 添加
            return out
        else:
            logger.debug("[parse] Stage 3 (python fence) found fence but no valid lines")  # ← 添加

    # ========== Stage 4: 逐行扫描（最严格）========== #
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data['factor_score']"):
            cleaned = re.sub(r'^(\d+[.)]\s+|[-*]\s+)', '', line)
            if cleaned.startswith("data['factor_score']"):
                out.append({"code": cleaned})
    
    if out:
        logger.debug(f"[parse] Stage 4 (line scan) success: {len(out)} items")  # ← 添加
    else:
        logger.warning("[parse] All stages failed, returning empty")  # ← 添加
        logger.debug(f"[parse] Response preview (first 300 chars): {text[:300]}")  # ← 添加
    
    return out

def _build_enhanced_memory_prompt(
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    n: int,
    round_id: int,
    sim_thr: float
) -> str:
    """
    构造英文 prompt：
    - 正/负样本仅展示 train_score + code；
    - 强化"禁止前视、rolling 后 shift(1)、字段白名单、单行可执行、返回 JSON 列表"等硬约束；
    - 加入字段白名单预览（来自 column_desc）。
    """
    def _fmt_pos(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:230]
        trn = rec.get("train_score", None)
        ts = "N/A" if trn is None else f"{float(trn):.3f}"
        return f"POS#{i}  TrainScore={ts}\n  {code}"

    def _fmt_neg(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:230]
        trn = rec.get("train_score", None)
        ts = "N/A" if trn is None else f"{float(trn):.3f}"
        return f"NEG#{i}  TrainScore={ts}  (validation underperformed; numbers hidden)\n  {code}"

    pos_block = "\n".join(_fmt_pos(r, i + 1) for i, r in enumerate(positives[:10])) or "N/A"
    neg_block = "\n".join(_fmt_neg(r, i + 1) for i, r in enumerate(negatives[:5])) or "N/A"

    return f"""
You are designing **single-line quantitative equity factor formulas** that are robust on VALIDATION (2001–2010).
Do NOT use any validation numbers from memory; you only see TRAIN (1961–2000) train_score and code snippets.

Context and evaluation protocol (numbers hidden):
- Train: 1961–2000 → produces train_score (robustness / overfit checks).
- Validation: 2001–2010 → used for iteration decisions and final scoring.
- Test: 2011–2025 → holdout.
- Final results columns (for humans) are ordered as:
  round, train_score, val_score, diversity, coverage, ann_ret, sharpe, D, max_dd, autocorr, skew
  where val_score = (sharpe + ann_ret + D)/3, D = 1 - max_drawdown.

Top examples (train_score + code):
{pos_block}

Anti-patterns to avoid (these underperformed on validation — numbers hidden):
{neg_block}

HARD CONSTRAINTS FOR EACH CANDIDATE:
- Return exactly {n} items as a pure JSON LIST, no prose, no backticks.
- Each item has ONLY one key: "code".
- "code" MUST be a **single line** of valid Python:
    data['factor_score'] = <expression>
- No imports; no multiple statements; no comments; no backticks; no variable definitions.
- Allowed field access: ONLY `data.get('<field>', 0)` from our schema (full list from project):
  {_FIELDS_FOR_PROMPT}
- Time ops must be explicit and safe: `.shift(k)` with k>=1; rolling stats like `.rolling(w).mean()` or `.std()` MUST be followed by `.shift(1)` to avoid look-ahead.
- **CRITICAL: `.shift()` can ONLY be used directly on pandas Series. Do NOT chain `.shift()` after numpy functions like `np.where()`, `np.tanh()`, `np.sign()`, etc.**
  Examples: ✓ `data.get('field',0).shift(1)` ✓ `data.get('field',0).rolling(4).mean().shift(1)` ✗ `np.where(...).shift(1)` ✗ `np.tanh(...).shift(1)`
- Allowed transforms: `np.tanh(x/2)`, `x/(1+np.abs(x))`, `np.sign(x)*np.sqrt(np.abs(x))`, `.rank(pct=True)`
- **NO LOOK-AHEAD**: At time t, use only information available at or before t (e.g., after rolling, always `.shift(1)`).

VALIDATION-ROBUSTNESS GUIDELINES (qualitative, not numbers):
- Prefer ratios over raw differences; scale by atq/saleq/ceqq or totals for comparability.
- Use year-over-year anchors (t vs t-4) where appropriate to control seasonal effects in quarterly series.
- Keep formulas concise and interpretable; avoid kitchen-sink combos and overfitting.
- Encourage diversity (variables, transforms, horizons). Two codes are considered too similar if textual similarity ≥ {sim_thr:.2f}.

Return JSON list ONLY:
[
  {{"code":"data['factor_score'] = (data.get('atq',0) - data.get('ltq',0)) / (1 + np.abs(data.get('saleq',0)))"}},
  {{"code":"data['factor_score'] = (data.get('saleq',0) / (1 + np.abs(data.get('atq',0)))).shift(1)"}}
]
(Do NOT use placeholders like '.' or '...'; return only valid, executable single-line code.)
""".strip()


def _call_llm_with_watchdog(prompt: str, temperature: float, max_tokens: int, timeout_s: int) -> Optional[str]:
    """
    带超时看门狗的 LLM 调用。超时/异常返回 None。
    """
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
        logger.error(f"[positive] LLM call timeout after {timeout_s}s")
        return None
    if result["err"]:
        return None
    return result["resp"]


# ===================== Main Class =====================

class PositiveAgents:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.logger = logger

    def analyze_memory(self, memory_records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        切分记忆（仅用于 prompt 展示；不暴露验证分）：
          positives: 上一轮的 10 个正样本（order preserved）
          negatives: 本轮由负向代理 GPT 直接生成的“最差因子”（order preserved）
        """
        if not memory_records:
            return [], []
        positives = [r for r in memory_records if r.get("memory_type") == "positive"]
        negatives = [r for r in memory_records if r.get("memory_type") == "negative"]
        return positives, negatives

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
        多次尝试 + 轻过采样 + 相似度递进放宽。
        不做内部 fallback；不足由 iterator 外层循环继续调用直至凑满。
        """
        self.logger.info(f"[positive] Generating {target_n} factors for round {current_round}")

        memory_records = memory_records or []
        positives = [r for r in memory_records if r.get("memory_type") == "positive"]
        negatives = [r for r in memory_records if r.get("memory_type") == "negative"]

        # For similarity checks against memory
        history_codes = []
        for r in memory_records:
            c = r.get("code", "")
            if c:
                history_codes.append(normalize_code(c))

        max_attempts = max(int(POS_CFG.get("max_attempts", MAX_RETRIES)), 3)
        batch_size = max(int(POS_CFG.get("batch_size", self.batch_size)), target_n)
        base_thr = float(POS_CFG.get("min_code_similarity", BASE_MIN_SIM))

        def _sim_thr(attempt_idx: int) -> float:
                # 地板值设为 0.40（允许更高的相似度）
            floor = 0.40
            # 每次尝试降低 0.05（更快放宽）
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
                if FORCE_LLM:
                    continue
                else:
                    continue

            parsed = _parse_llm_response(resp)
            if not parsed:
                self.logger.info("[positive] parser found 0 items from LLM text")
                continue


            # ========== 新增：拒绝统计 ========== #
            rejected = {"validate": 0, "fields": 0, "lookahead": 0, "duplicate": 0}
            # ==================================== #

            accepted_this = []
            for it in parsed:
                code_raw = it.get("code", "")
                code = validate_and_fix_code(code_raw)
                if not code:
                    rejected["validate"] += 1
                    continue

                # ====== 强校验：字段白名单 & no-look-ahead 规则 ======
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

                # ========== 临时禁用相似度检查 ========== #
                # （先跑通实验，后面再诊断 code_similarity 函数）
                # ======================================== #

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
                        f"[positive] attempt {attempt} rejected: "
                        f"validate={rejected['validate']}, fields={rejected['fields']}, "
                        f"lookahead={rejected['lookahead']}, dup={rejected['duplicate']}"
                    )
                pool.extend(accepted_this)
            else:
                self.logger.info(f"[positive] attempt {attempt}: no accepted candidates")
                if parsed:
                    self.logger.warning(
                        f"[positive] attempt {attempt} parsed {len(parsed)} but rejected all: "
                        f"validate={rejected['validate']}, fields={rejected['fields']}, "
                        f"lookahead={rejected['lookahead']}, dup={rejected['duplicate']}"
                    )


           

        # 不在正向内部做 fallback：仅返回 LLM 真实产出；由 iterator 再次调用直到凑满
        out = []
        for i, cand in enumerate(pool[:target_n], 1):
            out.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": cand["code"],
            })
        self.logger.info("[positive] Final LLM-derived candidates: %d (target=%d)", len(out), target_n)
        return out
