# -*- coding: utf-8 -*-
# experiments/iterative_negative_memory_with_reports_v2/negative_agents_reports.py
"""
Negative Agents WITH REPORTS (精简版)

核心流程（两步法）:
1. Step 1: 生成坏因子代码 + anti_pattern 标签
2. Step 2: 为每个代码生成简短分析报告（150-200 words）

改动 vs iterative_negative_memory:
- 不仅生成代码，还生成 factor_report
- 报告简洁（聚焦失败原因）
- 确保反模式类型多样化（distinct）
"""

from typing import List, Dict, Any, Optional, Tuple, Set
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
# 常量配置
# ========================================================================

# 反模式类别（规范化映射）
CANONICAL_ANTI_PATTERNS = {
    "financial_logic": {"financial_logic", "finance_logic", "logic", "financial"},
    "statistical": {"statistical", "stats", "stat"},
    "technical": {"technical", "tech"},
    "overfitting": {"overfit", "overfitting"},
    "data_quality": {"data_quality", "data", "quality"},
}

# Fallback 负样本模板
FALLBACK_NEGATIVE_TEMPLATES = [
    ("data['factor_score'] = (data.get('saleq',0) - data.get('saleq',0).shift(1))**2", "statistical"),
    ("data['factor_score'] = data.get('niq',0) / (data.get('ibq',0) + 1e-9)", "technical"),
    ("data['factor_score'] = data.get('atq',0) + data.get('saleq',0) + data.get('ltq',0)", "financial_logic"),
    ("data['factor_score'] = (data.get('actq',0) - data.get('lctq',0)).rank(pct=True) - (data.get('actq',0)-data.get('lctq',0)).rank(pct=True).shift(1)", "overfitting"),
    ("data['factor_score'] = (data.get('rectq',0) + data.get('invtq',0) - data.get('ppentq',0)) / (1 + np.abs(data.get('saleq',0)))", "data_quality"),
]

# Fallback 报告模板
FALLBACK_REPORT_TEMPLATE = """=== Negative Factor Report ===

Anti-Pattern Type:
Auto-generated fallback for item #{factor_num}.

Why This Will Fail:
Likely to carry little signal due to flawed construction; economic meaning is weak and unstable.

Expected Problems:
Noise sensitivity; poor generalization; fragile behavior under realistic data issues.

Learning Purpose:
Highlight why coherent economic logic and proper normalization are necessary.

Improvement Direction:
Use well-founded ratios/transformations; handle missing/outliers carefully; avoid ad-hoc conditions."""


# ========================================================================
# 工具函数
# ========================================================================

def _normalize_antipattern(s: str) -> str:
    """规范化反模式名称"""
    s = (s or "").strip().lower()
    for canonical, aliases in CANONICAL_ANTI_PATTERNS.items():
        if s in aliases or any(alias in s for alias in aliases):
            return canonical
    return s or "deliberate_negative"


# ========================================================================
# 核心类：NegativeAgent
# ========================================================================

class NegativeAgent:
    """
    负向代理（两步法 WITH REPORTS）
    
    主要方法:
    - generate_negative_factors_with_reports() 主入口
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 统一读取配置
        neg_cfg = CONFIG.get("NEGATIVE_AGENT_CONFIG", NEGATIVE_AGENT_CONFIG)
        
        # 代码生成参数
        self.code_temperature = float(neg_cfg.get("code_temperature", 0.65))
        self.code_max_tokens = int(neg_cfg.get("code_max_tokens", 800))
        
        # 报告生成参数
        self.report_temperature = float(neg_cfg.get("report_temperature", 0.80))
        self.report_max_tokens = int(neg_cfg.get("report_max_tokens", 500))
        self.report_max_chars = int(neg_cfg.get("report_max_chars", 200))
        
        # 质量控制
        self.enforce_distinct = bool(neg_cfg.get("enforce_distinct", True))
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
        两步法生成负样本（代码 + 报告）
        
        Args:
            current_round: 当前轮次
            target_n: 目标负样本数
            context_positives: 正样本上下文
                格式: [{"code": "...", "factor_report": "..."}, ...]
        
        Returns:
            [{"code": ..., "factor_report": ..., "anti_pattern": ..., "negative_id": ...}, ...]
        """
        context_positives = context_positives or []
        
        self.logger.info(
            f"[negative] Round {current_round}: 生成 {target_n} 个负样本 (WITH REPORTS)"
        )
        
        # ========== Step 1: 生成代码 + anti_pattern ========== #
        codes, patterns = self._step1_generate_codes_with_patterns(
            current_round=current_round,
            target_n=target_n,
            context_positives=context_positives
        )
        
        if not codes:
            self.logger.warning(f"[negative] Round {current_round}: 代码生成失败")
            return []
        
        # ========== Step 2: 生成报告 ========== #
        reports = self._step2_generate_short_reports(
            codes=codes,
            patterns=patterns,
            round_num=current_round
        )
        
        # ========== 组装返回 ========== #
        results = []
        for i, (code, report, pattern) in enumerate(zip(codes, reports, patterns), 1):
            results.append({
                "code": code,
                "factor_report": report,
                "anti_pattern": pattern,
                "negative_id": f"neg_{current_round:02d}_{i:02d}",
            })
        
        self.logger.info(
            f"[negative] Round {current_round}: 最终生成 {len(results)}/{target_n} 负样本 "
            f"(patterns: {sorted(set(patterns))})"
        )
        
        return results
    
    # ========== Step 1: 生成代码 + anti_pattern ========== #
    
    def _step1_generate_codes_with_patterns(
        self,
        current_round: int,
        target_n: int,
        context_positives: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        生成坏因子代码并附带 anti_pattern 标签
        
        Returns:
            (codes, patterns)
        """
        seen_patterns: Set[str] = set()
        out_codes: List[str] = []
        out_patterns: List[str] = []
        
        for attempt in range(1, self.max_calls_per_round + 1):
            need = target_n - len(out_codes)
            if need <= 0:
                break
            
            # 构建 Prompt
            prompt = self._build_negative_code_prompt(
                context_positives=context_positives,
                round_num=current_round,
                need=need,
                avoid_patterns=sorted(seen_patterns)
            )
            
            try:
                # 调用 GPT
                resp = call_gpt(
                    prompt, 
                    temperature=self.code_temperature, 
                    max_tokens=self.code_max_tokens
                )
                content = extract_content(resp)
                
                # 解析
                new_codes, new_patterns = self._parse_negative_code_response(content)
                
                # 过滤并收集
                for code, pattern in zip(new_codes, new_patterns):
                    norm_pattern = _normalize_antipattern(pattern)
                    
                    # 如果要求 distinct，跳过已有的反模式
                    if self.enforce_distinct and norm_pattern in seen_patterns:
                        continue
                    
                    out_codes.append(code)
                    out_patterns.append(norm_pattern)
                    seen_patterns.add(norm_pattern)
                    
                    if len(out_codes) >= target_n:
                        break
                
            except Exception as e:
                self.logger.error(
                    f"[negative] Round {current_round} attempt {attempt} failed: {e}"
                )
        
        # ========== 补齐（如果不足）========== #
        if len(out_codes) < target_n:
            shortage = target_n - len(out_codes)
            self.logger.info(f"[negative] Round {current_round} fallback: 需要 {shortage} 个")
            
            fallback_codes, fallback_patterns = self._fallback_negative_codes(
                shortage, 
                seen_patterns
            )
            out_codes.extend(fallback_codes)
            out_patterns.extend(fallback_patterns)
        
        return out_codes[:target_n], out_patterns[:target_n]
    
    def _build_negative_code_prompt(
        self,
        context_positives: List[Dict[str, Any]],
        round_num: int,
        need: int,
        avoid_patterns: List[str]
    ) -> str:
        """构造生成坏因子代码的 Prompt"""
        
        # 正样本示例（最多 3 个）
        pos_examples = []
        for i, rec in enumerate(context_positives[:3], 1):
            code = str(rec.get("code", ""))[:150]
            report = str(rec.get("factor_report", ""))[:100]
            pos_examples.append(f"POSITIVE #{i}: {code}\n  Report: {report}...")
        
        pos_block = "\n".join(pos_examples) if pos_examples else "N/A"
        
        # 已避免的反模式
        avoid_txt = ", ".join(avoid_patterns) if avoid_patterns else "None"
        
        # 可用的反模式类别
        allowable = [k for k in CANONICAL_ANTI_PATTERNS.keys() if k not in avoid_patterns]
        allowable_txt = ", ".join(allowable) if allowable else "Any"
        
        prompt = f"""
You are tasked to generate **DELIBERATELY BAD single-line factor formulas** for Round {round_num}.
These factors are expected to perform poorly on validation, while remaining syntactically valid.

CONTEXT (positive examples for contrast):
{pos_block}

YOUR TASK:
- Generate {need} BAD factors, each with a specific anti-pattern type.
- Anti-pattern categories to use: {allowable_txt}
- Already used (AVOID): {avoid_txt}

FAILURE MODES to exhibit:
• financial_logic: economically nonsensical relationships, mixing unrelated items
• statistical: unnormalized sums/diffs, excessive noise amplification
• technical: unstable denominators, improper handling of NA/Inf
• overfitting: overly specific conditions, fragile filters
• data_quality: ignoring outliers, using raw extremes improperly

CONSTRAINTS:
- Use ONLY fields from: {_FIELDS_FOR_PROMPT}
- Access via: data.get('field', 0)
- Format: data['factor_score'] = <expression>
- NO imports, multiple statements, or placeholders

Return JSON list ONLY:
[
  {{"code": "data['factor_score'] = ...", "anti_pattern": "financial_logic"}},
  {{"code": "data['factor_score'] = ...", "anti_pattern": "statistical"}}
]
""".strip()
        
        return prompt
    
    def _parse_negative_code_response(self, text: str) -> Tuple[List[str], List[str]]:
        """
        解析负样本代码响应（简化版）
        
        Returns:
            (codes, anti_patterns)
        """
        codes = []
        patterns = []
        
        try:
            # 清理可能的 markdown 包裹
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            
            # 解析 JSON
            data = json.loads(cleaned)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "code" in item:
                        # 清洗代码
                        code_raw = str(item["code"]).strip()
                        code_single = to_single_line(code_raw)
                        
                        if code_single and is_valid_single_line(code_single):
                            codes.append(code_single)
                            patterns.append(str(item.get("anti_pattern", "deliberate_negative")))
        
        except Exception as e:
            self.logger.debug(f"[negative] JSON parse failed: {e}")
        
        return codes, patterns
    
    # ========== Step 2: 生成简短报告 ========== #
    
    def _step2_generate_short_reports(
        self,
        codes: List[str],
        patterns: List[str],
        round_num: int
    ) -> List[str]:
        """
        为每个坏因子生成简短分析报告
        
        Args:
            codes: 坏因子代码列表
            patterns: 对应的反模式类型
            round_num: 轮次号
        
        Returns:
            报告列表
        """
        reports = []
        
        for i, (code, pattern) in enumerate(zip(codes, patterns), 1):
            try:
                report = self._generate_single_negative_report(
                    code=code,
                    pattern=pattern,
                    index=i,
                    round_num=round_num
                )
                reports.append(report)
            except Exception as e:
                self.logger.warning(f"[negative] Report generation failed for #{i}: {e}")
                # 使用 fallback
                reports.append(FALLBACK_REPORT_TEMPLATE.format(factor_num=i))
        
        return reports
    
    def _generate_single_negative_report(
        self,
        code: str,
        pattern: str,
        index: int,
        round_num: int
    ) -> str:
        """生成单个负样本的简短报告"""
        
        pattern_norm = _normalize_antipattern(pattern)
        
        prompt = f"""
You are analyzing a DELIBERATELY BAD factor for Round {round_num}, item #{index}.
The factor CODE is:

{code}

CONTEXT:
- Anti-pattern category: {pattern_norm}
- This factor is intentionally designed to fail on validation

TASK:
Write a SHORT analysis report (150-200 words) explaining WHY this factor will fail.

Required sections (concise):
1. Anti-Pattern Type: {pattern_norm}
2. Why This Will Fail: (1-2 sentences on the core flaw)
3. Expected Problems: (2-3 specific issues)
4. Learning Purpose: (what to avoid)
5. Improvement Direction: (1-2 concrete fixes)

Focus on the single most critical flaw and its concrete consequence.
Keep total length within {self.report_max_chars} characters.
Output plain text only (no markdown, no code fences).
""".strip()
        
        try:
            resp = call_gpt(
                prompt, 
                temperature=self.report_temperature, 
                max_tokens=self.report_max_tokens
            )
            
            report = extract_content(resp).strip()
            
            # 长度限制
            if len(report) > self.report_max_chars * 1.5:
                report = report[:self.report_max_chars] + "..."
            
            return report
            
        except Exception as e:
            self.logger.warning(f"[negative] GPT report generation failed: {e}")
            return FALLBACK_REPORT_TEMPLATE.format(factor_num=index)
    
    # ========== Fallback 生成 ========== #
    
    def _fallback_negative_codes(
        self, 
        n: int, 
        seen_patterns: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """生成 fallback 坏因子"""
        codes = []
        patterns = []
        
        for code, pattern in FALLBACK_NEGATIVE_TEMPLATES:
            if len(codes) >= n:
                break
            
            norm_pattern = _normalize_antipattern(pattern)
            
            # 如果要求 distinct 且已有此类型，跳过
            if self.enforce_distinct and norm_pattern in seen_patterns:
                continue
            
            codes.append(code)
            patterns.append(norm_pattern)
            seen_patterns.add(norm_pattern)
        
        return codes[:n], patterns[:n]


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
        [{"code": ..., "factor_report": ..., "anti_pattern": ...}, ...]
    """
    agent = NegativeAgent()
    return agent.generate_negative_factors_with_reports(
        current_round=current_round,
        target_n=target_n,
        context_positives=context_positives
    )


# ========================================================================
# 测试入口
# ========================================================================

if __name__ == "__main__":
    # 简单测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    agent = NegativeAgent()
    
    mock_positives = [
        {
            "code": "data['x'] = (data['revtq'] / data['atq']).fillna(0)", 
            "factor_report": "Strong asset turnover factor with broad coverage...",
        }
    ]
    
    try:
        results = agent.generate_negative_factors_with_reports(
            current_round=1,
            target_n=2,
            context_positives=mock_positives
        )
        
        print(f"\n✓ Generated {len(results)} negative samples:")
        for r in results:
            print(f"\nPattern: {r['anti_pattern']}")
            print(f"Code: {r['code'][:80]}...")
            print(f"Report: {r['factor_report'][:150]}...")
    except Exception as e:
        print(f"✗ Test failed: {e}")