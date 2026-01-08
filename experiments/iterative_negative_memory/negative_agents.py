# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory/negative_agents.py
"""
negative_agents.py — 由 GPT 直接“生成最差因子”的负向代理（Prompt 英文，注释中文）

改动要点：
- 移除 prompt 示例中的“...”/“.” 占位，避免模型照抄无效代码。
- _is_valid_single_line 增加语法编译校验，并拒绝 "."、"..." 等空表达式。
- generate_negative_factors 在数量不足时，无论 FORCE_LLM 与否都用本地坏模板补齐，防止 0/5 中断。
"""

from typing import List, Dict, Any, Optional
import logging
import re
import json
import difflib

from shared.code_utils import (
    normalize_code,
    code_similarity,
    to_single_line,
    is_valid_single_line,
    extract_content
)

# ===== GPT 调用器（兼容不同签名） =====
# GPT 调用器（从 common 导入）
try:
    from common.gpt_runner import call_gpt
    GPT_AVAILABLE = True
except Exception:
    GPT_AVAILABLE = False

# 配置读取（从同目录导入）
try:
    from .config import CONFIG, NEGATIVE_AGENT_CONFIG
except Exception:
    CONFIG = {}
    NEGATIVE_AGENT_CONFIG = {}

# 字段白名单（从 common 导入）
try:
    from common.column_desc import COLUMN_DESC
    ALLOWED_FIELDS: List[str] = list(COLUMN_DESC.keys())
    _FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS) if len(ALLOWED_FIELDS) <= 200 else (", ".join(ALLOWED_FIELDS[:200]) + ", ...")
except Exception as e:
    raise ImportError(
        f"[negative_agents] Failed to import COLUMN_DESC from common.column_desc: {e}"
    )

logger = logging.getLogger(__name__)

# ---- 运行参数 ----
# 从 NEGATIVE_AGENT_CONFIG 读取（如果没有则用默认值）
NEG_CONFIG = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)

# 负向代理特定参数
NEG_TEMPERATURE_FIRST = float(NEG_CONFIG.get("temperature_first", 0.85))
NEG_TEMPERATURE_LATER = float(NEG_CONFIG.get("temperature_later", 0.95))
MAX_TOTAL_NEG_CALLS   = int(NEG_CONFIG.get("max_calls_per_round", 4))
NEG_OVERASK           = int(NEG_CONFIG.get("overask", 3))
NEG_MIN_CODE_SIM      = float(NEG_CONFIG.get("min_code_similarity", 0.80))

# 通用 GPT 参数（使用合理默认值）
GPT_MAX_TOKENS  = 2200   # 可以从 common.gpt_runner 的默认值读取
TIMEOUT_SECONDS = 90     # 合理默认值

# ====== 工具函数（中文注释） ======
def _parse_as_json_list(text: str) -> List[str]:
    """按 JSON 列表解析，返回 code 列表"""
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
    """解析 ```lang ... ``` 块"""
    import re as _re
    pattern = _re.compile(rf"```{lang}\s*(.*?)\s*```", _re.DOTALL | _re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1) if m else ""

def _split_by_separator(text: str, sep: str = "---FACTOR_SEPARATOR---") -> List[str]:
    """按自定义分隔符拆分"""
    if sep in text:
        parts = [p.strip() for p in text.split(sep)]
        return [p for p in parts if p]
    return []

def _parse_candidates(text: str) -> List[str]:
    """多路兜底解析，返回原始 code 片段列表（未清洗）"""
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
    # 3) 分隔符
    parts = _split_by_separator(text)
    if parts:
        return parts
    # 4) ```python
    py = _parse_fenced(text, "python")
    if py:
        return [py]
    # 5) 行级扫描（抓包含 factor_score 的行）
    lines = []
    for ln in text.splitlines():
        if "factor_score" in ln:
            lines.append(ln.strip())
    return lines

# ====== 核心类 ======

class NegativeAgent:
    """负向代理：直接调用 GPT 生成“最差因子”"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 从配置读取参数
        neg_config = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)
        self.temperature_first = float(neg_config.get("temperature_first", 0.85))
        self.temperature_later = float(neg_config.get("temperature_later", 0.95))
        self.max_calls_per_round = int(neg_config.get("max_calls_per_round", 4))
        
        # 打印白名单信息（便于诊断）
        self.logger.info(f"[negatives] ALLOWED_FIELDS loaded: {len(ALLOWED_FIELDS)} fields")

    # ====== 对外主方法 ======

    def generate_negative_factors(
        self,
        current_round: int,
        target_n: int,
        context_positives: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        调用 GPT 直接“生成”最差因子（不是挑选），返回长度 <= target_n 的 list[{"code": ...}]
        """
        context_positives = context_positives or []

        # 构建英文 prompt（仅展示 train_score + code）
        prompt = self._build_prompt_en(
            current_round=current_round,
            need=target_n,
            positives=context_positives
        )

        collected: List[str] = []
        calls = 0
        last_err: Optional[Exception] = None

        # 内部循环：多次尝试直到收够 target_n 或达到 cap
        while len(collected) < target_n and calls < self.max_calls_per_round:
            need = target_n - len(collected)
            ask = need + max(1, NEG_OVERASK)  # 轻过采样
            calls += 1
            temp = self.temperature_first if calls == 1 else self.temperature_later

            self.logger.info(f"[negatives] round={current_round} call#{calls} need={need} ask={ask} temp={temp:.2f}")

            # 调用 GPT
            if not GPT_AVAILABLE:
                self.logger.warning("[negatives] GPT not available; break")
                break
            try:
                try:
                    resp = call_gpt(prompt, temperature=temp, max_tokens=GPT_MAX_TOKENS, timeout=TIMEOUT_SECONDS)
                except TypeError:
                    resp = call_gpt(prompt, temperature=temp, max_tokens=GPT_MAX_TOKENS)
                text = extract_content(resp)
            except Exception as e:
                last_err = e
                self.logger.error(f"[negatives] GPT error: {e}")
                continue

            # 解析 & 清洗
            raw_codes = _parse_candidates(text)
            accepted = []
            for raw in raw_codes:
                single = to_single_line(raw)
                if not single:
                    continue
                if not is_valid_single_line(single):
                    continue
                # 与已收集去重
                if any(code_similarity(single, x) >= NEG_MIN_CODE_SIM for x in collected):
                    continue
                accepted.append(single)

            # 纳入收集
            if accepted:
                for s in accepted:
                    if s not in collected:
                        collected.append(s)
                self.logger.info(f"[negatives] +{len(accepted)} accepted (now {len(collected)}/{target_n})")
            else:
                self.logger.info("[negatives] 0 accepted from this call")

        # 若仍不足：统一补齐（避免 0/5 导致上层中断）
        if len(collected) < target_n:
            msg = f"[negatives] only {len(collected)}/{target_n} after {calls} calls; last_err={last_err}"
            self.logger.warning(msg + " (topping up with local bad templates)")
            collected.extend(self._bad_templates()[: (target_n - len(collected))])

        out = [{"code": c} for c in collected[:target_n]]
        self.logger.info(f"[negatives] final {len(out)}/{target_n}")
        return out

    # ====== Prompt 组装（英文，强调“坏模式”与硬约束） ======

    def _build_prompt_en(self, current_round: int, need: int, positives: List[Dict[str, Any]]) -> str:
        """构造英文 prompt（只展示 train_score + code；字段白名单取自 column_desc）"""
        def _fmt_pos(i: int, rec: Dict[str, Any]) -> str:
            code = str(rec.get("code", ""))[:220]
            ts = rec.get("train_score", None)
            ts_str = "N/A" if ts is None else f"{float(ts):.3f}"
            return f"POS#{i}  TrainScore={ts_str}\n  {code}"

        pos_block = "\n".join(_fmt_pos(i + 1, r) for i, r in enumerate(positives[:10])) or "N/A"

        prompt = f"""
You are a quantitative researcher tasked to generate **DELIBERATELY BAD single-line factor formulas**.
These factors are expected to perform poorly on the **validation period (2001–2015)**, while remaining syntactically valid and executable.

Context for reference (NO validation numbers are shown):
Training period: 1961–2000 (we show train_score only).
Validation: 2001–2010 (used for decision-making; DO NOT reveal nor assume exact numbers).
Test: 2011–2025 (holdout).

Top examples from the previous round (for contrast; show TrainScore + code):
{pos_block}

Your task now:
- Directly **generate {need} BAD factors** (not select), each encoded as a **single-line** Python statement:
    data['factor_score'] = <expression>
- The factors should **exhibit failure modes** likely to underperform on validation, such as:
  • extremely short horizons (excessive noise amplification),
  • unstable denominators (near-zero risks),
  • lack of normalization/scaling,
  • economically weak or contradictory relationships,
  • kitchen-sink style combinations without clear rationale,
  • overly simple (single variable) OR overly complex constructions.
- However, code MUST remain valid and **NO LOOK-AHEAD** is allowed.

ALLOWED operations / safety constraints:
- Use ONLY `data.get('<field>', 0)` to access inputs. The available fields come **exactly** from our project schema:
  { _FIELDS_FOR_PROMPT }
- Time ops must be explicit and safe: `.shift(k)` with k>=1; rolling stats like `.rolling(w).mean()` or `.std()` MUST be followed by `.shift(1)` to avoid look-ahead.
- Allowed transforms: `np.tanh(x/2)`, `x/(1+np.abs(x))`, `np.sign(x)*np.sqrt(np.abs(x))`, `.rank(pct=True)`
- **Prohibited**: imports, multiple statements, comments, backticks, variable assignments other than `data['factor_score'] = ...`.

Return format (STRICT):
- Return **exactly {need} items** as a **pure JSON list** of objects with a single key "code".
- Do NOT include any explanation or backticks.

Example (structure only; do NOT include this example in your output):
[
  {{"code": "data['factor_score'] = (data.get('saleq',0)-data.get('saleq',0).shift(1))**2"}},
  {{"code": "data['factor_score'] = data.get('niq',0) / (1e-9 + np.abs(data.get('ibq',0)))"}}
]
""".strip()
        return prompt

    # ====== “坏模式”模板（仅在兜底时使用） ======

    def _bad_templates(self) -> List[str]:
        """典型“坏模式”但语法合法的单行模板（用于兜底补齐）"""
        return [
            "data['factor_score'] = (data.get('saleq',0) - data.get('saleq',0).shift(1))**2",
            "data['factor_score'] = data.get('niq',0) / (data.get('ibq',0) + 1e-9)",
            "data['factor_score'] = data.get('atq',0) + data.get('saleq',0) + data.get('ltq',0)",
            "data['factor_score'] = (data.get('actq',0) - data.get('lctq',0)).rank(pct=True) - (data.get('actq',0)-data.get('lctq',0)).rank(pct=True).shift(1)",
            "data['factor_score'] = np.tanh((data.get('prccq',0) * data.get('cshoq',0)) / (1 + np.abs(data.get('ceqq',0))))",
            "data['factor_score'] = (data.get('rectq',0) + data.get('invtq',0) - data.get('ppentq',0)) / (1 + np.abs(data.get('saleq',0)))",
        ]
