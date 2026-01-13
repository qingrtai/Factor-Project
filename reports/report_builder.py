# reports/report_builder.py (IMPROVED - 适配版)

"""
改进策略：
1. 保持所有现有函数接口不变
2. 只改进 report_template.txt 的内容
3. 在 build_report_prompt 中增强 prompt 构建
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from common.gpt_runner import call_gpt

# 默认值
REPORT_TEMPLATE_FILE = Path(__file__).parent / "report_template.txt"
REPORT_MAX_TOKENS = 600  # 保持不变
TEMPERATURE = 0.7
LOGS_DIR = Path(__file__).resolve().parents[1] / "results" / "logs"


@dataclass
class FactorMetrics:
    """Container for metrics to feed into the report."""
    sharpe: Optional[float] = None
    ann_ret: Optional[float] = None
    max_dd: Optional[float] = None
    coverage: Optional[float] = None
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    style_hint: Optional[str] = None
    rank: Optional[int] = None
    total_factors: Optional[int] = None


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.strip()


def _append_json_item(path: Path, item: Dict) -> None:
    """Append item to a JSON list file (create if missing)."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(item)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _round_or_skip(x: Optional[float], ndigits: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except Exception:
        return None


def _summarize_code(code: str, max_chars: int = 800) -> str:
    """Make a short summary of the code."""
    code = code.strip()
    if len(code) <= max_chars:
        return code
    head = code[: max_chars - 50].rstrip()
    return head + "\n... [truncated]"


def _format_metrics_table(m: FactorMetrics) -> str:
    """Return a compact text table with metrics."""
    rows: List[str] = []
    
    # 排名信息
    if m.rank is not None and m.total_factors is not None:
        rank_pct = (m.rank / m.total_factors) * 100
        rank_tier = "TOP" if rank_pct <= 30 else "MIDDLE" if rank_pct <= 70 else "BOTTOM"
        rows.append(f"- Rank: {m.rank}/{m.total_factors} ({rank_tier} tier)")
    
    if m.style_hint:
        rows.append(f"- Style Hint: {m.style_hint}")
    
    s = _round_or_skip(m.sharpe, 3)
    a = _round_or_skip(m.ann_ret, 3)
    d = _round_or_skip(m.max_dd, 3)
    c = _round_or_skip(m.coverage, 4)
    tr = _round_or_skip(m.train_score, 4)
    
    if tr is not None:
        rows.append(f"- Train Score: {tr}")
    if s is not None:
        quality = "excellent" if s > 1.5 else "good" if s > 1.0 else "moderate" if s > 0.5 else "poor"
        rows.append(f"- Sharpe: {s} ({quality})")
    if a is not None:
        rows.append(f"- Annual Return: {a}")
    if d is not None:
        risk = "low" if abs(d) < 0.2 else "medium" if abs(d) < 0.4 else "high"
        rows.append(f"- Max Drawdown: {d} ({risk} risk)")
    if c is not None:
        adequacy = "adequate" if c > 0.35 else "insufficient"
        rows.append(f"- Coverage: {c} ({adequacy})")
    
    if not rows:
        return "(no metrics provided)"
    return "\n".join(rows)


def _infer_style_from_code(code: str) -> str:
    """Simple heuristic to infer factor style from code."""
    code_lower = code.lower()
    
    if any(k in code_lower for k in ['niq', 'profit', 'earnings', 'ebit']):
        return "profitability"
    elif any(k in code_lower for k in ['revt', 'sale', 'revenue', 'growth']):
        return "growth"
    elif any(k in code_lower for k in ['cogs', 'opex', 'efficiency', 'margin']):
        return "efficiency"
    elif any(k in code_lower for k in ['debt', 'leverage', 'equity', 'solvency']):
        return "quality"
    elif any(k in code_lower for k in ['momentum', 'return', 'price', 'trend']):
        return "momentum"
    elif any(k in code_lower for k in ['value', 'book', 'pe', 'pb']):
        return "value"
    elif any(k in code_lower for k in ['volatility', 'std', 'var']):
        return "volatility"
    else:
        return "mixed"


# ========== 关键改进：增强版 Prompt 构建 ========== #

def build_report_prompt(
    code: str,
    metrics: FactorMetrics,
    template_text: str,
    *,
    max_code_chars: int = 800,
    is_detailed: bool = True,
) -> str:
    """
    Assemble a prompt for GPT to produce a factor report.
    
    ========== 改进点 ========== 
    1. 结构化模板（5 部分正样本，4 部分负样本）
    2. 明确要求（具体字数、格式）
    3. 强调可操作性
    
    Args:
        is_detailed: If False, generate a brief negative-example report
    """
    code_snippet = _summarize_code(code, max_chars=max_code_chars)
    metrics_table = _format_metrics_table(metrics)
    
    # 自动推断 style
    if not metrics.style_hint:
        metrics.style_hint = _infer_style_from_code(code)
    
    if is_detailed:
        # ========== 改进版：详细报告（正样本）========== #
        parts = [
            "=== TASK: Generate a STRUCTURED factor analysis report ===",
            "",
            "CODE:",
            code_snippet,
            "",
            "METRICS:",
            metrics_table,
            "",
            "=== OUTPUT REQUIREMENTS ===",
            "",
            "Write a 350-400 word analysis with EXACTLY these 5 sections:",
            "",
            "**1. ECONOMIC LOGIC**",
            "What financial relationship does this factor capture?",
            "Why should it predict stock returns?",
            "Example: \"Captures profitability efficiency by combining operating income with asset utilization\"",
            "",
            "**2. TECHNICAL IMPLEMENTATION**",
            "How is it calculated? What transformations are used?",
            "Why this specific formula design?",
            "Example: \"Uses cross-sectional ranking to normalize, applies pct_change for momentum signal\"",
            "",
            "**3. PERFORMANCE ANALYSIS**",
            f"Interpret the metrics: Sharpe {metrics.sharpe if metrics.sharpe else 'N/A'}, ",
            f"Return/Drawdown tradeoff, Coverage {metrics.coverage if metrics.coverage else 'N/A'}",
            "Example: \"Strong Sharpe 2.0+ indicates stable risk-adjusted returns with moderate 30% drawdown\"",
            "",
            "**4. SUCCESS FACTORS**",
            "List 2-3 SPECIFIC reasons why this factor works.",
            "What are its technical or economic advantages?",
            "Example: \"1) Robust ratio handles outliers 2) Combines growth + quality 3) Adequate 70% coverage\"",
            "",
            "**5. IMPROVEMENT DIRECTIONS**",
            "Suggest SPECIFIC variations to try:",
            "- Different field ratios or combinations",
            "- Alternative time windows or smoothing",
            "- Additional normalizations",
            "Example: \"Try longer 8-quarter rolling windows; add debt/equity ratio; test geometric vs arithmetic mean\"",
            "",
            "=== FORMAT REQUIREMENTS ===",
            "- Use section headers EXACTLY as shown above",
            "- Each section 2-4 sentences (not a single long paragraph)",
            "- Be SPECIFIC and CONCRETE (avoid generic statements)",
            "- Focus on ACTIONABLE insights that help generate better factors",
            "- Total 350-400 words",
            "- Plain text only (no markdown formatting)",
            "",
            "=== CRITICAL NOTES ===",
            "Your report will be used to teach the next round of factor generation.",
            "Extract insights that can be APPLIED, not just observed.",
        ]
    else:
        # ========== 改进版：简要报告（负样本）========== #
        parts = [
            "=== TASK: Generate a STRUCTURED negative-example report ===",
            "",
            "CODE:",
            code_snippet,
            "",
            "METRICS:",
            metrics_table,
            "",
            "This is a POOR-performing factor. Write a 250-300 word analysis with EXACTLY these 4 sections:",
            "",
            "**1. CORE FLAW**",
            "What is fundamentally wrong with this factor's design? (1-2 sentences)",
            "",
            "**2. SPECIFIC PROBLEMS**",
            "List 3-4 concrete issues:",
            "- Economic logic problem (if any)",
            "- Statistical or technical flaw",
            "- Data quality concern",
            "- Why it fails on validation",
            "",
            "**3. EXPECTED FAILURES**",
            "Explain WHY (not just THAT) it will fail:",
            f"- Low Sharpe {metrics.sharpe if metrics.sharpe else 'N/A'}: why?",
            f"- High drawdown {metrics.max_dd if metrics.max_dd else 'N/A'}: why?",
            f"- Poor coverage {metrics.coverage if metrics.coverage else 'N/A'}: why?",
            "",
            "**4. CORRECT APPROACH**",
            "How should this concept be implemented properly? Give SPECIFIC fixes.",
            "Example: \"Should use rank() for normalization, add np.where() for zero handling, combine with quality screen\"",
            "",
            "=== FORMAT REQUIREMENTS ===",
            "- Use section headers EXACTLY as shown",
            "- Be SPECIFIC about WHY it fails (root cause analysis)",
            "- Total 250-300 words",
            "- Plain text only",
        ]
    
    return "\n".join(parts)


# ========== 保持原函数不变 ========== #

def generate_factor_report(
    code: str,
    metrics: FactorMetrics,
    *,
    template_path: Optional[Path] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = REPORT_MAX_TOKENS,
    round_id: Optional[int] = None,
    factor_id: Optional[str] = None,
    is_detailed: bool = True,
) -> str:
    """Generate a factor report using GPT.
    
    Args:
        is_detailed: If True, generate detailed report; if False, brief negative-example report
    """
    # ========== 改进点：不再使用外部模板文件 ========== #
    # 直接使用增强版的 build_report_prompt
    
    prompt = build_report_prompt(
        code=code, 
        metrics=metrics, 
        template_text="",  # 不再需要模板
        is_detailed=is_detailed
    )
    
    raw_resp = ""
    try:
        # ========== 改进点：调整温度和 tokens ========== #
        # 详细报告用稍低温度（提高准确性），更多 tokens
        if is_detailed:
            actual_temp = 0.60  # 从 0.70 降低到 0.60
            actual_max_tokens = 1000  # 从 600 增加到 1000
        else:
            actual_temp = 0.70
            actual_max_tokens = 700  # 从 300 增加到 700
        
        raw_resp = call_gpt(prompt, temperature=actual_temp, max_tokens=actual_max_tokens)
        report = (raw_resp or "").strip()
    except Exception as e:
        if is_detailed:
            report = f"(fallback) Unable to generate detailed report. Reason: {e}"
        else:
            report = f"(fallback) Poor performance - avoid this pattern."
    
    # 后处理
    if len(report) > 2500:
        report = report[:2500].rstrip() + "\n[truncated]"
    
    # Logging
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        rid = f"{round_id}" if round_id is not None else "unknown"
        log_path = LOGS_DIR / f"gpt_reports_round_{rid}.json"
        _append_json_item(
            log_path,
            {
                "factor_id": factor_id or "",
                "round_id": round_id,
                "is_detailed": is_detailed,
                "rank": metrics.rank,
                "train_score": metrics.train_score,
                "prompt_len": len(prompt),
                "report_len": len(report),
                "report": report,
            },
        )
    except Exception:
        pass
    
    if not report:
        report = "(fallback) Empty report."
    
    return report


def generate_comparative_summary(
    top_factors: List[Tuple[str, FactorMetrics, str]],
    bottom_factors: List[Tuple[str, FactorMetrics, str]],
) -> str:
    """Generate a brief comparative summary of what worked vs what didn't.
    
    Args:
        top_factors: List of (code, metrics, report) for top performers
        bottom_factors: List of (code, metrics, report) for bottom performers
    
    Returns:
        A concise summary string
    """
    if not top_factors:
        return ""
    
    avg_top_sharpe = sum(m.sharpe for _, m, _ in top_factors if m.sharpe) / len(top_factors)
    avg_bottom_sharpe = sum(m.sharpe for _, m, _ in bottom_factors if m.sharpe) / len(bottom_factors) if bottom_factors else 0
    
    summary = f"""
=== ROUND SUMMARY ===
Top {len(top_factors)} factors avg Sharpe: {avg_top_sharpe:.3f}
Bottom {len(bottom_factors)} factors avg Sharpe: {avg_bottom_sharpe:.3f}

Key patterns in successful factors:
{_extract_common_patterns([c for c, _, _ in top_factors])}

Patterns to avoid (from failed factors):
{_extract_common_patterns([c for c, _, _ in bottom_factors])}
"""
    return summary.strip()


def _extract_common_patterns(codes: List[str]) -> str:
    """Extract common patterns from a list of factor codes."""
    if not codes:
        return "N/A"
    
    keywords = ['pct_change', 'rolling', 'rank', 'fillna', 'shift', 'diff', 
                'revtq', 'saleq', 'cogsq', 'niq', 'atq']
    
    counts = {k: sum(1 for c in codes if k in c.lower()) for k in keywords}
    common = [k for k, v in sorted(counts.items(), key=lambda x: -x[1])[:3] if v > 0]
    
    if not common:
        return "No clear patterns detected"
    
    return ", ".join(common)


if __name__ == "__main__":
    # Test
    dummy_code = "factor = (df['revtq'].pct_change(4) / df['atq']).rank(pct=True)"
    fm = FactorMetrics(
        sharpe=1.23,
        ann_ret=0.18,
        max_dd=-0.35,
        coverage=0.42,
        train_score=0.27,
        rank=2,
        total_factors=10,
    )
    out = generate_factor_report(
        code=dummy_code,
        metrics=fm,
        round_id=1,
        factor_id="test_001",
        is_detailed=True
    )
    print(out)
