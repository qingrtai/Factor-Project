# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/positive_agents.py
"""
正向因子生成代理（对照组版本）

改动 vs 旧版 V3:
- 去掉严格2行格式验证（_validate_two_line_format, _simple_validate, _is_valid_format）
- 新增 _force_single_assignment() 自动修复格式（从 baseline 移植）
- 新增 5 层代码质量过滤（numpy array / forbidden / bare column / assignment / whitelist）
- batch_size=5 小批量生成，提高成功率
- Prompt / temperature / 学习策略 不变（对照组，不影响 val_score）
"""

from typing import List, Dict, Any, Optional
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
GPT_TEMP_DEFAULT = float(CONFIG.get("GPT_TEMPERATURE", 0.40))
GPT_MAX_TOKENS = int(CONFIG.get("GPT_MAX_TOKENS", 2200))
MAX_RETRIES = int(CONFIG.get("MAX_RETRIES", 6)) or 6

BASE_MIN_SIM = float(POS_CFG.get("min_code_similarity", 0.70))
BATCH_SIZE = int(POS_CFG.get("batch_size", 5))

# ============================================================
# 列名白名单集合（用于 _all_columns_allowed）
# ============================================================
_ALLOWED_SET = set(ALLOWED_FIELDS)


# ============================================================
# 解析 LLM 响应
# ============================================================

def _parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """解析 LLM 响应，支持多种格式"""
    if not text:
        return []

    # Stage 1: raw JSON array
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            valid = [item for item in data if isinstance(item, dict) and "code" in item]
            if valid:
                return valid
    except Exception:
        pass

    # Stage 2: ```json fence
    m = re.search(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                valid = [item for item in data if isinstance(item, dict) and "code" in item]
                if valid:
                    return valid
        except Exception:
            pass

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
            return out

    # Stage 4: 逐行扫描
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data['factor_score']"):
            cleaned = re.sub(r'^(\d+[.)]\s+|[-*]\s+)', '', line)
            if cleaned.startswith("data['factor_score']"):
                out.append({"code": cleaned})

    if not out:
        logger.warning("[parse] All stages failed")
        logger.debug(f"[parse] Response preview: {text[:300]}")

    return out


# ============================================================
# Prompt 构建（保持和旧版一致，不影响 val_score）
# ============================================================

def _build_prompt(
    positives: List[Dict[str, Any]],
    negatives: List[Dict[str, Any]],
    n: int,
    round_id: int,
) -> str:
    """
    构建 prompt — 保持旧版的 flat 展示 + 对比学习结构

    唯一新增: numpy array 方法警告（防止 np.where().rank() 错误）
    """
    def _fmt_pos(rec: Dict[str, Any], i: int) -> str:
        code = str(rec.get("code", ""))[:300]
        trn = rec.get("train_score", None)
        val = rec.get("val_score", None)
        ts = "N/A" if trn is None else f"{float(trn):.4f}"
        vs = "N/A" if val is None else f"{float(val):.4f}"
        return f"#{i}  Train={ts}  Val={vs}\n{code}\n"

    pos_block = "\n".join(_fmt_pos(r, i + 1) for i, r in enumerate(positives[:20]))

    neg_block = ""
    if negatives:
        def _fmt_neg(rec: Dict[str, Any], i: int) -> str:
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

These factors have LOW train scores. DO NOT copy these patterns.
"""

    return f"""You are generating factor formulas for quantitative investing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP PERFORMING FACTORS (LEARN FROM THESE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{pos_block}
{neg_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each factor must be a SINGLE LINE in this exact form:
  data['factor_score'] = <EXPRESSION>

Where <EXPRESSION> is a financial ratio or transformation.

Examples:
  data['factor_score'] = np.where(data['atq']==0, 0, data['niq']/data['atq'])
  data['factor_score'] = data['saleq'].rolling(4).mean().shift(1)
  data['factor_score'] = np.where(data['saleq'].shift(4)==0, 0, (data['saleq']-data['saleq'].shift(4))/data['saleq'].shift(4))

**CRITICAL - AVOID NUMPY ARRAY METHODS:**
- NEVER write: np.where(...).rank() or np.where(...).rolling() or np.where(...).pct_change()
- np.where() returns a numpy array, which does NOT have .rank(), .rolling(), .shift(), .pct_change() methods
- If you need these transformations, apply them BEFORE np.where or on pandas Series directly:
  ✓ CORRECT: (data['col1']/data['col2']).rolling(4).mean()
  ✗ WRONG: np.where(data['col']==0, 0, expr).rank()

Constraints:
- Use ONLY data['field'] with fields from: {_FIELDS_FOR_PROMPT}
- For division, use np.where(denom==0, 0, numer/denom)
- Time operations: .shift(k) with k>=1; .rolling(w).mean()/.std() MUST be followed by .shift(1)
- NO imports, NO multiple statements, NO comments

OUTPUT: Pure JSON array, no explanations:
[
  {{"code": "data['factor_score'] = np.where(data['saleq']==0, 0, data['ibq']/data['saleq'])"}},
  {{"code": "data['factor_score'] = data['niq'].rolling(4).mean().shift(1)"}}
]

Generate exactly {n} UNIQUE factors. Start with '[' immediately.""".strip()


# ============================================================
# LLM 调用
# ============================================================

def _call_llm_with_watchdog(prompt: str, temperature: float, max_tokens: int, timeout_s: int) -> Optional[str]:
    """带超时的 LLM 调用"""
    if not GPT_AVAILABLE:
        logger.warning("[positive] GPT not available")
        return None

    result: Dict[str, Any] = {"resp": None, "err": None, "done": False}
    done_event = threading.Event()

    def _runner():
        try:
            result["resp"] = call_gpt(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
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


# ============================================================
# PositiveAgents 类
# ============================================================

class PositiveAgents:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.logger = logger
        self.blacklist_words = ['price', 'volume', 'high', 'low']

    # ----------------------------------------------------------
    # 核心: 自动格式修复（从 baseline 移植）
    # ----------------------------------------------------------
    def _force_single_assignment(self, code: str) -> Optional[str]:
        """
        把任何格式的因子代码修复为单行 data['factor_score']=... 格式

        处理:
        1. JSON 污染清理 ("},{"  尾部垃圾)
        2. 多行合并为单行
        3. 去掉第二行 fillna（会在外层统一包装）
        4. 去掉 data.get() 转为 data['field']
        """
        if not code or "factor_score" not in code:
            return None

        # --- JSON 污染清理 ---
        # GPT 有时在 code 值里混入下一个 JSON 对象的内容
        for sep in ['"},{"', '"}, {"', '"}  ,  {"']:
            if sep in code:
                code = code[:code.index(sep)]

        # 清理尾部常见垃圾
        for tail in ['"}', '",', '"}]', '", "id']:
            if code.rstrip().endswith(tail):
                code = code.rstrip()[:-len(tail)].rstrip()

        # --- 多行处理 ---
        lines = [ln.strip() for ln in code.split('\n') if ln.strip()]

        # 如果是2行格式（line1=赋值, line2=fillna），只取第一行
        if len(lines) == 2 and 'fillna' in lines[1]:
            code = lines[0]
        elif len(lines) >= 1:
            # 取包含 factor_score 赋值的那行
            for ln in lines:
                if ln.startswith("data['factor_score']") and '=' in ln:
                    code = ln
                    break
            else:
                code = lines[0]

        code = code.strip()

        # --- 必须以 data['factor_score'] 开头 ---
        if not code.startswith("data['factor_score']"):
            return None

        # --- 提取等号右边的表达式 ---
        eq_pos = code.find('=')
        if eq_pos < 0:
            return None
        expr = code[eq_pos + 1:].strip()

        if not expr:
            return None

        # 去掉已有的 .fillna(0) / .replace(...)
        expr = re.sub(r'\.fillna\([^)]*\)\s*$', '', expr).strip()
        expr = re.sub(r'\.replace\(\[np\.inf,\s*-np\.inf\],\s*np\.nan\)\s*$', '', expr).strip()

        # --- data.get('field', 0) → data['field'] ---
        expr = re.sub(r"data\.get\(\s*'([^']+)'\s*,\s*0\s*\)", r"data['\1']", expr)

        if not expr:
            return None

        # --- 重新包装为标准单行 ---
        final = (
            f"data['factor_score']=pd.Series({expr},"
            f"index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)"
        )

        return final

    # ----------------------------------------------------------
    # 5 层代码质量过滤
    # ----------------------------------------------------------
    def _check_numpy_array_methods(self, code: str, rejected: list) -> bool:
        """检测 np.where() 返回值上调用 pandas 方法（会报错）"""
        bad_patterns = [
            r'np\.where\([^)]*\)\s*\.rank\s*\(',
            r'np\.where\([^)]*\)\s*\.rolling\s*\(',
            r'np\.where\([^)]*\)\s*\.shift\s*\(',
            r'np\.where\([^)]*\)\s*\.pct_change\s*\(',
            r'np\.where\([^)]*\)\s*\.fillna\s*\(',
            r'np\.where\([^)]*\)\s*\.replace\s*\(',
        ]
        for pat in bad_patterns:
            if re.search(pat, code):
                rejected.append("numpy_array_method")
                return False
        return True

    def _forbidden_scan(self, code: str, rejected: list) -> bool:
        """检测禁用 token、lookahead、黑名单字段"""
        low = code.lower()

        # --- 禁用 token ---
        forbidden_tokens = [
            'import ', 'open(', 'exec(', 'eval(', '__', 'os.',
            'sys.', 'subprocess', 'lambda', 'def ', 'class ',
        ]
        for tok in forbidden_tokens:
            if tok in low:
                rejected.append(f"forbidden:{tok.strip()}")
                return False

        # --- lookahead 检查 ---
        lookahead_tokens = [
            '.shift(0)', '.shift(-',
        ]
        for tok in lookahead_tokens:
            if tok in code:
                rejected.append(f"lookahead:{tok}")
                return False

        # rolling 必须跟 shift(1)
        if '.rolling(' in code:
            # 找 rolling(...).agg(...) 但没有 .shift(
            if '.shift(' not in code:
                rejected.append("lookahead:rolling_no_shift")
                return False

        # --- blacklist_words（软警告，不拒绝）---
        for w in self.blacklist_words:
            if w in low:
                rejected.append(f"keyword:{w}")
                # 只警告，不 return False

        return True

    def _no_bare_column_refs(self, code: str, rejected: list) -> bool:
        """
        确保所有列引用使用 data['col'] 格式，不能裸用列名

        检测模式: 独立出现的已知列名（不在 data['...'] 内）
        """
        # 先去掉所有 data['xxx'] 引用
        cleaned = re.sub(r"data\['[^']+'\]", "___COL___", code)

        # 检查是否有裸列名
        for field in ALLOWED_FIELDS[:50]:  # 只检查常用字段
            if field in ['at', 'do', 'pi']:  # 跳过太短的
                continue
            pattern = r'\b' + re.escape(field) + r'\b'
            if re.search(pattern, cleaned):
                rejected.append(f"bare_column:{field}")
                return False

        return True

    def _all_columns_allowed(self, code: str, rejected: list) -> bool:
        """检查所有引用的列名是否在白名单中"""
        refs = re.findall(r"data\['([^']+)'\]", code)
        for col in refs:
            if col == 'factor_score':
                continue
            if col not in _ALLOWED_SET:
                rejected.append(f"unknown_column:{col}")
                return False
        return True

    def _sanitize_one(self, raw_code: str, rejected: list) -> Optional[str]:
        """
        5 层过滤管线:
        1. _force_single_assignment (格式修复)
        2. _check_numpy_array_methods
        3. _forbidden_scan
        4. _no_bare_column_refs
        5. _all_columns_allowed
        """
        # --- 第 1 层: 格式修复 ---
        single = self._force_single_assignment(raw_code)
        if not single:
            rejected.append("format_fix_failed")
            return None

        c = single.strip()

        # --- 第 2 层: numpy array 方法检查 ---
        if not self._check_numpy_array_methods(c, rejected):
            return None

        # --- 第 3 层: 禁用 token + lookahead ---
        if not self._forbidden_scan(c, rejected):
            return None

        # --- 第 4 层: 裸列名引用 ---
        if not self._no_bare_column_refs(c, rejected):
            return None

        # --- 第 5 层: 列名白名单 ---
        if not self._all_columns_allowed(c, rejected):
            return None

        # --- 基本语法检查 ---
        if c.count('(') != c.count(')'):
            rejected.append("unbalanced_parens")
            return None

        return c

    # ----------------------------------------------------------
    # 主生成方法
    # ----------------------------------------------------------
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
        生成因子（保持和旧版相同的接口和学习策略）
        """
        self.logger.info(f"[positive] Generating {target_n} factors for round {current_round}")

        memory_records = memory_records or []
        positives = [r for r in memory_records if r.get("memory_type") == "positive"]
        negatives = [r for r in memory_records if r.get("memory_type") == "negative"]

        max_attempts = max(int(POS_CFG.get("max_attempts", MAX_RETRIES)), 3)
        base_thr = float(POS_CFG.get("min_code_similarity", BASE_MIN_SIM))

        pool: List[Dict[str, Any]] = []
        seen_norms = set()

        for attempt in range(1, max_attempts + 1):
            need = target_n - len(pool)
            if need <= 0:
                break

            # 每次请求 batch_size 个（默认5）
            batch = max(1, min(self.batch_size, need))
            temp = 0.40 if attempt == 1 else min(0.50, max(0.40, GPT_TEMP_DEFAULT))

            self.logger.info(
                f"[positive] attempt {attempt}/{max_attempts} need={need} "
                f"ask={batch} temp={temp:.2f}"
            )

            prompt = _build_prompt(
                positives=positives,
                negatives=negatives,
                n=batch,
                round_id=current_round,
            )

            resp = _call_llm_with_watchdog(
                prompt=prompt,
                temperature=temp,
                max_tokens=GPT_MAX_TOKENS,
                timeout_s=TIMEOUT_S,
            )
            if not resp:
                self.logger.warning("[positive] empty/failed LLM response")
                continue

            parsed = _parse_llm_response(resp)
            if not parsed:
                self.logger.info("[positive] parser found 0 items")
                continue

            rejected_reasons = []
            accepted_this = 0

            for it in parsed:
                code_raw = it.get("code", "")
                if not code_raw:
                    continue

                # --- 5 层过滤 ---
                rejected_this = []
                code = self._sanitize_one(code_raw, rejected_this)
                if not code:
                    rejected_reasons.extend(rejected_this)
                    continue

                # --- 去重 ---
                norm = normalize_code(code)
                if not norm or norm in seen_norms:
                    rejected_reasons.append("duplicate")
                    continue

                # --- 通过 ---
                seen_norms.add(norm)
                pool.append({"code": code})
                accepted_this += 1

                if len(pool) >= target_n:
                    break

            self.logger.info(
                f"[positive] attempt {attempt}: +{accepted_this} "
                f"(cum {len(pool)}/{target_n})"
            )
            if rejected_reasons:
                # 统计拒绝原因
                from collections import Counter
                counts = Counter(rejected_reasons)
                top_reasons = ", ".join(f"{k}={v}" for k, v in counts.most_common(5))
                self.logger.info(f"[positive] rejected: {top_reasons}")

        # --- 最终输出 ---
        out = []
        for i, cand in enumerate(pool[:target_n], 1):
            out.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": cand["code"],
            })

        self.logger.info(f"[positive] Final: {len(out)} (target={target_n})")
        return out
