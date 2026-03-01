# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory_with_reports_v2/negative_agents_reports.py
"""
Negative Agents WITH REPORTS (v2 精简版)

核心改动 vs 旧版:
- 去掉 anti-pattern 类别系统（旧版只有5类，不够7个负样本）
- 一步生成 code + reason（旧版两步：先代码再逐个生成报告，N+1次GPT调用）
- 只按代码内容去重（不按类别去重）
- GPT 自由生成差因子，不受类别限制
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
import logging

# 共享工具
from shared.code_utils import (
    to_single_line,
    is_valid_single_line,
    extract_content
)

# GPT 调用器
from common.gpt_runner import call_gpt

# 配置
try:
    from .config import CONFIG, NEGATIVE_AGENT_CONFIG
except Exception:
    CONFIG = {}
    NEGATIVE_AGENT_CONFIG = {}

# 字段白名单
try:
    from common.column_desc import COLUMN_DESC
    ALLOWED_FIELDS: List[str] = list(COLUMN_DESC.keys())
    _FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS[:200]) + (", ..." if len(ALLOWED_FIELDS) > 200 else "")
except Exception as e:
    raise ImportError(f"[negative] Failed to import COLUMN_DESC: {e}")

logger = logging.getLogger(__name__)


# ========================================================================
# Fallback 负样本（兜底用）
# ========================================================================

FALLBACK_NEGATIVES = [
    {
        "code": "data['factor_score'] = (data.get('saleq',0) - data.get('saleq',0).shift(1))**2",
        "reason": "Squared difference of same field amplifies noise without economic meaning."
    },
    {
        "code": "data['factor_score'] = data.get('niq',0) / (data.get('ibq',0) + 1e-9)",
        "reason": "Net income divided by income before tax is near-constant ratio, no predictive signal."
    },
    {
        "code": "data['factor_score'] = data.get('atq',0) + data.get('saleq',0) + data.get('ltq',0)",
        "reason": "Raw sum of unrelated accounting items without normalization, dominated by firm size."
    },
    {
        "code": "data['factor_score'] = (data.get('actq',0) - data.get('lctq',0)).rank(pct=True) - (data.get('actq',0)-data.get('lctq',0)).rank(pct=True).shift(1)",
        "reason": "First-difference of ranks is extremely noisy and mean-reverting, unlikely to carry signal."
    },
    {
        "code": "data['factor_score'] = (data.get('rectq',0) + data.get('invtq',0) - data.get('ppentq',0)) / (1 + np.abs(data.get('saleq',0)))",
        "reason": "Mixing current assets with fixed assets in numerator has no coherent economic interpretation."
    },
    {
        "code": "data['factor_score'] = data.get('cheq',0) * data.get('dpq',0) - data.get('xintq',0)",
        "reason": "Multiplying cash by depreciation is economically meaningless, unstable across industries."
    },
    {
        "code": "data['factor_score'] = (data.get('cogsq',0) / (data.get('invtq',0) + 1e-9)) ** 3",
        "reason": "Cubing inventory turnover amplifies outliers massively, creating extreme values."
    },
    {
        "code": "data['factor_score'] = data.get('oiadpq',0) - data.get('revtq',0) + data.get('atq',0)",
        "reason": "Subtracting revenue from operating income then adding total assets mixes scales and concepts."
    },
    {
        "code": "data['factor_score'] = np.where(data.get('niq',0) > 0, 1, -1) * data.get('saleq',0)",
        "reason": "Binary sign of income times sales is a crude threshold that discards magnitude information."
    },
    {
        "code": "data['factor_score'] = (data.get('dlttq',0) - data.get('dlcq',0)) / (data.get('seqq',0) + data.get('cheq',0) + 1e-9)",
        "reason": "Long-term minus short-term debt over equity plus cash lacks clear economic rationale."
    },
]


# ========================================================================
# 核心类：NegativeAgent
# ========================================================================

class NegativeAgent:
    """
    负向代理（一步法 WITH REPORTS）
    
    一次 GPT 调用同时生成 code + reason
    只按代码内容去重，不按类别限制
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 统一读取配置
        neg_cfg = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)
        
        # 生成参数
        self.temperature = float(neg_cfg.get("code_temperature", 0.65))
        self.max_tokens = int(neg_cfg.get("code_max_tokens", 800))
        self.report_max_chars = int(neg_cfg.get("report_max_chars", 200))
        self.max_calls_per_round = int(neg_cfg.get("max_calls_per_round", 4))
        
        self.logger.info(f"[negative] Initialized (fields: {len(ALLOWED_FIELDS)})")
    
    # ========== 对外主方法 ========== #
    
    def generate_negative_factors_with_reports(
        self,
        current_round: int,
        target_n: int,
        context_positives: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        一步法生成负样本（代码 + reason）
        
        Args:
            current_round: 当前轮次
            target_n: 目标负样本数
            context_positives: 正样本上下文
                格式: [{"code": "...", "factor_report": "..."}, ...]
        
        Returns:
            [{"code": ..., "factor_report": ..., "negative_id": ...}, ...]
        """
        context_positives = context_positives or []
        
        self.logger.info(
            f"[negative] Round {current_round}: 生成 {target_n} 个负样本"
        )
        
        # ========== 生成 code + reason ========== #
        collected = self._generate_negative_batch(
            current_round=current_round,
            target_n=target_n,
            context_positives=context_positives
        )
        
        # ========== 组装返回 ========== #
        results = []
        for i, item in enumerate(collected, 1):
            results.append({
                "code": item["code"],
                "factor_report": item["reason"],
                "negative_id": f"neg_{current_round:02d}_{i:02d}",
            })
        
        self.logger.info(
            f"[negative] Round {current_round}: 最终生成 {len(results)}/{target_n} 负样本"
        )
        
        return results
    
    # ========== 核心生成逻辑 ========== #
    
    def _generate_negative_batch(
        self,
        current_round: int,
        target_n: int,
        context_positives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量生成负样本，只按代码去重
        
        Returns:
            [{"code": ..., "reason": ...}, ...]
        """
        out: List[Dict[str, Any]] = []
        seen_codes: set = set()
        
        for attempt in range(1, self.max_calls_per_round + 1):
            need = target_n - len(out)
            if need <= 0:
                break
            
            # 构建 Prompt
            prompt = self._build_negative_prompt(
                context_positives=context_positives,
                round_num=current_round,
                need=need
            )
            
            try:
                resp = call_gpt(
                    prompt, 
                    temperature=self.temperature, 
                    max_tokens=self.max_tokens
                )
                content = extract_content(resp)
                parsed = self._parse_negative_response(content)
                
                # 按代码内容去重
                for item in parsed:
                    code_key = item["code"].replace(" ", "").replace("\n", "").lower()
                    if code_key not in seen_codes:
                        out.append(item)
                        seen_codes.add(code_key)
                    if len(out) >= target_n:
                        break
                        
            except Exception as e:
                self.logger.error(
                    f"[negative] Round {current_round} attempt {attempt} failed: {e}"
                )
        
        # ========== Fallback 补齐 ========== #
        if len(out) < target_n:
            shortage = target_n - len(out)
            self.logger.info(f"[negative] Round {current_round} fallback: 补 {shortage} 个")
            
            for fb in FALLBACK_NEGATIVES:
                if len(out) >= target_n:
                    break
                code_key = fb["code"].replace(" ", "").replace("\n", "").lower()
                if code_key not in seen_codes:
                    out.append(fb)
                    seen_codes.add(code_key)
        
        return out[:target_n]
    
    # ========== Prompt 构建 ========== #
    
    def _build_negative_prompt(
        self,
        context_positives: List[Dict[str, Any]],
        round_num: int,
        need: int,
    ) -> str:
        """构造生成坏因子的 Prompt（一步法：code + reason）"""
        
        # 正样本上下文（最多 3 个）
        pos_examples = []
        for i, rec in enumerate(context_positives[:3], 1):
            code = str(rec.get("code", ""))[:150]
            report = str(rec.get("factor_report", ""))[:100]
            pos_examples.append(f"GOOD #{i}: {code}\n  Why it works: {report}...")
        
        pos_block = "\n".join(pos_examples) if pos_examples else "N/A"
        
        prompt = f"""
You generate DELIBERATELY BAD single-line factor formulas for Round {round_num}.
These factors must be syntactically valid but expected to perform poorly on validation.

CONTEXT — these are GOOD factors (do the OPPOSITE):
{pos_block}

YOUR TASK:
Generate {need} BAD factors. Each must have:
- "code": a valid single-line formula (data['factor_score'] = ...)
- "reason": 2-3 sentences explaining the core flaw and why it will fail

Common ways to make bad factors:
- Economically nonsensical relationships (mixing unrelated items)
- Unnormalized sums dominated by firm size
- Excessive noise amplification (squaring differences, cubing ratios)
- Near-constant expressions with no cross-sectional variation
- Unstable denominators causing extreme outliers
- Redundant transformations that cancel out signal

CONSTRAINTS:
- Use ONLY fields from: {_FIELDS_FOR_PROMPT}
- Access via: data.get('field', 0)
- Format: data['factor_score'] = <expression>
- NO imports, multiple statements, or placeholders
- Each factor must be DIFFERENT from the others

Return JSON list ONLY (no markdown, no explanation):
[
  {{"code": "data['factor_score'] = ...", "reason": "This factor fails because ..."}},
  ...
]
""".strip()
        
        return prompt
    
    # ========== 解析 ========== #
    
    def _parse_negative_response(self, text: str) -> List[Dict[str, Any]]:
        """
        解析 GPT 响应为 [{"code": ..., "reason": ...}]
        """
        results = []
        
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            
            # 找 JSON 数组边界
            l = cleaned.find("[")
            r = cleaned.rfind("]")
            if l != -1 and r != -1 and r > l:
                cleaned = cleaned[l:r+1]
            
            data = json.loads(cleaned)
            
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict) or "code" not in item:
                        continue
                    
                    code_raw = str(item["code"]).strip()
                    code_single = to_single_line(code_raw)
                    
                    if code_single and is_valid_single_line(code_single):
                        reason = str(item.get("reason", "Deliberately bad factor.")).strip()
                        # 截断过长的 reason
                        if len(reason) > self.report_max_chars:
                            reason = reason[:self.report_max_chars] + "..."
                        
                        results.append({
                            "code": code_single,
                            "reason": reason
                        })
        
        except Exception as e:
            self.logger.debug(f"[negative] JSON parse failed: {e}")
        
        return results


# ========================================================================
# Module-level API
# ========================================================================

def generate_negative_factors_with_reports(
    current_round: int,
    target_n: int,
    context_positives: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    模块级便捷接口
    
    Returns:
        [{"code": ..., "factor_report": ..., "negative_id": ...}, ...]
    """
    agent = NegativeAgent()
    return agent.generate_negative_factors_with_reports(
        current_round=current_round,
        target_n=target_n,
        context_positives=context_positives
    )
