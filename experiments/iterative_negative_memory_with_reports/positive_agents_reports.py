# experiments/iterative_negative_memory_with_reports/positive_agents_reports.py
"""
正向因子生成代理（带报告版本）

核心特点：
1. 基于报告学习（而非数值分数）
2. 输入：memory_records = [{"code": "...", "factor_report": "...", "memory_type": "positive"}, ...]
3. Prompt：自然语言展示历史因子分析（Top-3 优点 + Bottom-3 问题）
4. 健壮的去重、安全校验、补货逻辑

改动 vs iterative_baseline_with_reports:
- 类名：PositiveAgents（复数，匹配 iterator）
- 接口：generate_factors() 适配 iterator 调用
- 内部：保留 generate_optimized_factors() 核心逻辑
"""

import json
import re
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# 路径：把仓库根目录加入 sys.path
_CUR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# GPT 调用
from common.gpt_runner import call_gpt

# 配置：兼容 dict CONFIG 和模块常量
try:
    from .config import CONFIG as _CONFIG
except Exception:
    _CONFIG = None
try:
    from . import config as _cfg_module
except Exception:
    _cfg_module = None


def _cfg_get(key: str, default=None):
    """兼容配置读取"""
    if isinstance(_CONFIG, dict) and key in _CONFIG:
        return _CONFIG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default


# 列名白名单
try:
    from common.column_desc import COLUMN_DESC
    FIELD_WHITELIST = sorted(list(COLUMN_DESC.keys()))
except Exception:
    FIELD_WHITELIST = []


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


# ========================================================================
# 主类：PositiveAgents（注意复数）
# ========================================================================

class PositiveAgents:  # ← 类名改为复数
    """
    正向因子生成代理（带报告版本）
    
    对外接口：
    - generate_factors(current_round, target_n, id_prefix, memory_records)
      → 返回 [{"factor_id": "r2_01", "code": "..."}, ...]
    
    内部实现：
    - generate_optimized_factors(previous_factors, round_num, ...)
      → 返回 List[str]（代码列表）
    """

    # ================= 初始化 =================
    def __init__(self):
        self.logger = logging.getLogger('positive_agents_reports')
        
        # 读取配置
        pos_cfg = _cfg_get('POSITIVE_AGENT_CONFIG', {})
        self.temperature = float(pos_cfg.get('temp_evolve', 0.70))
        self.max_tokens = int(_cfg_get('FACTOR_GENERATION_MAX_TOKENS', 900))
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', 10))
        self.logs_dir = Path(_cfg_get('LOGS_DIR', _CUR / "logs"))
        _ensure_dir(self.logs_dir)

        # 列名白名单
        self.field_whitelist: List[str] = [str(c) for c in FIELD_WHITELIST]
        self.field_set_lower = {c.lower(): c for c in self.field_whitelist}
        self.logger.info(f"[positive] 字段白名单载入：{len(self.field_whitelist)} 列")

        # 安全限制
        self.max_code_len = int(_cfg_get('MAX_CODE_LEN', 1500))
        self.forbidden_tokens = ['import', 'def ', 'class ', 'for ', 'while ', 'eval(', 'exec(']
        self.lookahead_tokens = ['shift(-', 'lead(', 'future']
        
        # 生成参数
        self.refill_max_attempts = 3
        self.batch_size = 5

    # ================= 对外接口（iterator 调用）=================
    def generate_factors(
        self,
        current_round: int,
        target_n: int,
        id_prefix: str,
        memory_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成因子（适配 iterator 调用）
        
        Args:
            current_round: 当前轮次
            target_n: 目标生成数量
            id_prefix: 因子 ID 前缀（如 "r2_"）
            memory_records: 记忆列表
                格式: [{"code": "...", "factor_report": "...", "memory_type": "positive"}, ...]
        
        Returns:
            [{"factor_id": "r2_01", "code": "..."}, ...]
        """
        self.logger.info(
            f"[positive] Round {current_round}: 生成 {target_n} 个因子 "
            f"(记忆数: {len(memory_records)})"
        )
        
        # ========== Step 1: 转换格式 ========== #
        # memory_records → previous_factors
        previous_factors = []
        for rec in memory_records:
            code = str(rec.get('code', '')).strip()
            if not code:
                continue
            
            previous_factors.append({
                'code': code,
                'report': rec.get('factor_report', ''),  # ← factor_report → report
                'train_score': rec.get('train_score', 0),  # 用于排序
                'memory_type': rec.get('memory_type', 'positive')
            })
        
        self.logger.info(f"[positive]   - 有效记忆: {len(previous_factors)}")
        
        # ========== Step 2: 调用内部生成逻辑 ========== #
        codes = self.generate_optimized_factors(
            previous_factors=previous_factors,
            round_num=current_round,
            n_override=target_n,
            save_response=True,
            existing_codes=None
        )
        
        # ========== Step 3: 转换返回格式 ========== #
        results = []
        for i, code in enumerate(codes, 1):
            results.append({
                "factor_id": f"{id_prefix}{i:02d}",
                "code": code
            })
        
        self.logger.info(f"[positive]   - 最终生成: {len(results)}/{target_n}")
        
        return results

    # ================= 内部核心逻辑 =================
    def generate_optimized_factors(
        self,
        previous_factors: List[Dict],  # ← [{'code': '...', 'report': '...', 'train_score': ...}, ...]
        round_num: int,
        save_response: bool = True,
        n_override: Optional[int] = None,
        existing_codes: Optional[List[str]] = None
    ) -> List[str]:
        """
        基于报告生成新因子（内部方法）
        
        Args:
            previous_factors: 上一轮因子列表
                格式: [{'code': '...', 'report': '...', 'train_score': ...}, ...]
            round_num: 当前轮次
            save_response: 是否保存 GPT 响应
            n_override: 覆盖默认的因子数量
            existing_codes: 已存在的代码（用于去重）
        
        Returns:
            生成的因子代码列表: List[str]
        """
        target_n = int(n_override) if (n_override and n_override > 0) else self.factors_per_round

        # ========== 按 train_score 排序和分组 ========== #
        prev_pairs_with_score = []
        for f in previous_factors or []:
            code = str(f.get('code', '')).strip()
            if not code:
                continue
            report = str(f.get('report', '')).strip()
            train_score = f.get('train_score', -999)
            memory_type = f.get('memory_type', 'positive')
            
            prev_pairs_with_score.append({
                'code': code, 
                'report': report, 
                'train_score': train_score,
                'memory_type': memory_type
            })

        # 按 train_score 排序（降序）
        prev_pairs_with_score.sort(key=lambda x: x.get('train_score', -999), reverse=True)

        # 分离 Top 和 Bottom（只从正样本中选）
        positives = [p for p in prev_pairs_with_score if p.get('memory_type') == 'positive']
        negatives = [p for p in prev_pairs_with_score if p.get('memory_type') == 'negative']
        
        top_k = min(3, len(positives))
        bottom_k = min(2, len(positives))
        
        top_factors = positives[:top_k] if positives else []
        bottom_factors = positives[-bottom_k:] if len(positives) > bottom_k else []
        
        # 负样本单独处理（作为反面教材）
        negative_factors = negatives[:3] if negatives else []

        prev_pairs = prev_pairs_with_score

        if not prev_pairs:
            self.logger.warning("[positive] 记忆为空，零上下文生成")

        # ========== 分批生成 ========== #
        codes: List[str] = []
        need = target_n
        batch = max(1, min(self.batch_size, need))
        batch_idx = 0
        
        while need > 0:
            batch_idx += 1
            
            # 构建 Prompt
            prompt = self._build_prompt(
                prev_pairs=prev_pairs,
                target_n=batch,
                top_factors=top_factors,
                bottom_factors=bottom_factors,
                negative_factors=negative_factors  # ← 传入负样本
            )
            
            self.logger.info(f"[positive] Round {round_num} Batch {batch_idx}: 请求 GPT 生成 {batch} 个")
            
            # 调用 GPT
            resp = call_gpt(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
            if not isinstance(resp, str):
                resp = str(resp or '')
            
            if save_response:
                self._save_gpt_response(round_num, prompt, resp, suffix=f"_batch_{batch_idx}_{batch}")
            
            self.logger.debug(f"[positive] GPT 响应前 200 字：{resp[:200].replace(os.linesep, ' ')}")

            # 解析
            got = self._parse_and_transform(resp, batch)

            # 去重并收集
            seen = set(self._norm_key(c) for c in codes)
            for c in got:
                k = self._norm_key(c)
                if k not in seen and len(codes) < target_n:
                    codes.append(c)
                    seen.add(k)

            need = target_n - len(codes)
            batch = max(1, min(self.batch_size, need))

        # 去重：排除已尝试/已评估
        if existing_codes:
            exist_keys = set(self._norm_key(c) for c in existing_codes if isinstance(c, str))
            codes = [c for c in codes if self._norm_key(c) not in exist_keys]

        # ========== 补货 ========== #
        attempts = 0
        consecutive_no_progress = 0
        last_count = 0
        
        while len(codes) < target_n and attempts < self.refill_max_attempts:
            attempts += 1
            last_count = len(codes)
            missing = target_n - len(codes)
            
            self.logger.info(
                f"[positive] Round {round_num} 补货 {attempts}/{self.refill_max_attempts}: "
                f"需要 {missing} 个（已有 {len(codes)}）"
            )

            reasons = getattr(self, "_last_reject_reasons", [])
            refill_prompt = self._build_refill_prompt(missing, reasons, existing_codes=codes)
            
            resp2 = call_gpt(
                refill_prompt, 
                temperature=min(self.temperature + 0.08, 0.85),
                max_tokens=max(600, self.max_tokens)
            )
            if not isinstance(resp2, str):
                resp2 = str(resp2 or '')
            
            if save_response:
                self._save_gpt_response(round_num, refill_prompt, resp2, suffix=f"_refill_{attempts}")
            
            more = self._parse_and_transform(resp2, missing)
            
            if existing_codes:
                exist_keys = set(self._norm_key(c) for c in existing_codes if isinstance(c, str))
                more = [c for c in more if self._norm_key(c) not in exist_keys]

            seen = set(self._norm_key(c) for c in codes)
            for c in more:
                k = self._norm_key(c)
                if k not in seen and len(codes) < target_n:
                    codes.append(c)
                    seen.add(k)

            if len(codes) == last_count:
                consecutive_no_progress += 1
                if consecutive_no_progress >= 2:
                    self.logger.warning(f"[positive] 连续 {consecutive_no_progress} 次补货无效，停止")
                    break
            else:
                consecutive_no_progress = 0

        return codes[:target_n]

    

    # ================= Prompt 构造 =================
    def _whitelist_block(self) -> str:
        if not self.field_whitelist:
            return ""
        cols = "\n".join(self.field_whitelist[:200])  # 限制长度
        return (
            "ALLOWED COLUMNS (USE ONLY THESE):\n"
            f"{cols}\n"
            f"... ({len(self.field_whitelist)} total)\n\n"
        )

    
    # ========== 方法 1: 深度提取经济逻辑 ========== #
    
    def _extract_economic_logic_deep(self, report: str) -> str:
        """
        深度提取经济逻辑（按结构提取）
        
        优先级：
        1. 按 "ECONOMIC LOGIC" 部分标题提取
        2. Fallback 到关键词匹配
        3. 最后返回报告前 200 字符
        """
        if not report:
            return ""
        
        lines = report.split('\n')
        
        # 方法 1: 按部分标题提取
        logic_lines = []
        in_section = False
        
        for line in lines:
            upper = line.upper()
            
            # 进入 ECONOMIC LOGIC 部分
            if 'ECONOMIC LOGIC' in upper:
                in_section = True
                continue
            
            # 离开该部分（遇到其他部分标题）
            if in_section and any(h in upper for h in [
                'TECHNICAL', 'PERFORMANCE', 'SUCCESS', 'IMPROVEMENT'
            ]):
                break
            
            # 收集内容行
            if in_section and line.strip() and len(line.strip()) > 10:
                # 排除纯标题行（以 : 结尾）
                if not line.strip().endswith(':'):
                    logic_lines.append(line.strip())
        
        if logic_lines:
            result = " ".join(logic_lines)
            return result[:280]
        
        # 方法 2: 关键词搜索（fallback）
        keywords = ['captures', 'measures', 'represents', 'reflects', 
                    'profitability', 'efficiency', 'growth', 'quality', 'momentum']
        
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in keywords) and len(line.strip()) > 20:
                return line.strip()[:280]
        
        # 方法 3: 返回报告前 200 字符（最后 fallback）
        return report[:200].strip()
    
    
    # ========== 方法 2: 深度提取成功要素 ========== #
    
    def _extract_success_factors_deep(self, report: str) -> str:
        """
        深度提取成功要素（按结构提取）
        
        目标：提取 "SUCCESS FACTORS" 或 "WHY IT WORKS" 部分
        """
        if not report:
            return ""
        
        lines = report.split('\n')
        
        # 按部分标题提取
        success_lines = []
        in_section = False
        
        for line in lines:
            upper = line.upper()
            
            # 进入 SUCCESS 部分（多种可能的标题）
            if any(header in upper for header in [
                'SUCCESS FACTOR', 'KEY SUCCESS', 'WHY IT WORKS', 'WHY THIS WORKS'
            ]):
                in_section = True
                continue
            
            # 离开该部分
            if in_section and any(h in upper for h in [
                'IMPROVEMENT', 'PERFORMANCE', 'TECHNICAL', 'ECONOMIC'
            ]):
                break
            
            # 收集内容行
            if in_section and line.strip() and len(line.strip()) > 10:
                if not line.strip().endswith(':') and not line.strip().startswith('#'):
                    success_lines.append(line.strip())
        
        if success_lines:
            result = " ".join(success_lines)
            return result[:250]
        
        # Fallback: 关键词搜索
        keywords = ['because', 'advantage', 'strength', 'effective', 
                    'robust', 'works by', 'key reason', 'success']
        
        relevant = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in keywords) and len(line.strip()) > 15:
                relevant.append(line.strip())
                if len(relevant) >= 2:
                    break
        
        if relevant:
            return " ".join(relevant)[:250]
        
        return ""
    
    
    # ========== 方法 3: 深度提取改进建议 ========== #
    
    def _extract_improvement_ideas_deep(self, report: str) -> str:
        """
        深度提取改进建议（按结构提取）
        
        目标：提取 "IMPROVEMENT DIRECTIONS" 部分
        """
        if not report:
            return ""
        
        lines = report.split('\n')
        
        # 按部分标题提取
        improve_lines = []
        in_section = False
        
        for line in lines:
            upper = line.upper()
            
            # 进入 IMPROVEMENT 部分
            if 'IMPROVEMENT' in upper or 'ENHANCE' in upper or 'VARIATION' in upper:
                in_section = True
                continue
            
            # 离开该部分
            if in_section and any(h in upper for h in [
                'ECONOMIC', 'TECHNICAL', 'PERFORMANCE', 'SUCCESS'
            ]):
                break
            
            # 收集内容行
            if in_section and line.strip() and len(line.strip()) > 10:
                if not line.strip().endswith(':'):
                    improve_lines.append(line.strip())
        
        if improve_lines:
            result = " ".join(improve_lines)
            return result[:220]
        
        # Fallback: 关键词搜索
        keywords = ['try', 'could', 'consider', 'suggest', 'alternative',
                    'variation', 'enhance', 'improve', 'test', 'add']
        
        relevant = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in keywords) and len(line.strip()) > 15:
                relevant.append(line.strip())
                if len(relevant) >= 2:
                    break
        
        if relevant:
            return " ".join(relevant)[:220]
        
        return ""
    
    
    # ========== 方法 4: 深度提取失败分析 ========== #
    
    def _extract_failure_analysis_deep(self, report: str) -> str:
        """
        深度提取失败分析（按结构提取）
        
        目标：提取 "CORE FLAW" 或 "SPECIFIC PROBLEMS" 部分
        """
        if not report:
            return ""
        
        lines = report.split('\n')
        
        # 按部分标题提取
        flaw_lines = []
        in_section = False
        
        for line in lines:
            upper = line.upper()
            
            # 进入 FLAW 部分
            if any(header in upper for header in [
                'CORE FLAW', 'SPECIFIC PROBLEM', 'WHY IT FAIL'
            ]):
                in_section = True
                continue
            
            # 离开该部分
            if in_section and any(h in upper for h in [
                'EXPECTED FAILURE', 'CORRECT APPROACH', 'IMPROVEMENT'
            ]):
                break
            
            # 收集内容行
            if in_section and line.strip() and len(line.strip()) > 10:
                if not line.strip().endswith(':'):
                    flaw_lines.append(line.strip())
        
        if flaw_lines:
            result = " ".join(flaw_lines)
            return result[:250]
        
        # Fallback: 关键词搜索
        keywords = ['flaw', 'problem', 'issue', 'fails because', 'wrong',
                    'unstable', 'poor', 'error', 'weakness']
        
        relevant = []
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in keywords) and len(line.strip()) > 15:
                relevant.append(line.strip())
                if len(relevant) >= 2:
                    break
        
        if relevant:
            return " ".join(relevant)[:250]
        
        # 最后 fallback
        return report[:180].strip()
    
    
    # ========== 方法 5: 改进版 Prompt 构建 ========== #
    
    def _build_prompt(
        self, 
        prev_pairs: List[Dict],
        target_n: int,
        top_factors: Optional[List[Dict]] = None,
        bottom_factors: Optional[List[Dict]] = None,
        negative_factors: Optional[List[Dict]] = None
    ) -> str:
        """
        改进版 Prompt - 三层学习框架
        
        关键改进：
        1. 使用深度信息提取函数
        2. 结构化展示学习材料
        3. 三层学习框架（Why → How → Avoid）
        4. 提供具体设计模式
        """
        
        rules = (
            "You are a quantitative finance expert. Generate profitable factors.\n\n"
            f"**OUTPUT FORMAT (STRICT):**\n"
            f"{{\"factors\": [{{\"id\":\"r01_01\",\"code\":\"...\"}}, ...]}} - EXACTLY {target_n} factors\n"
            "- Format: data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)\n"
            "- Use ONLY allowed columns, NO loops/imports\n\n"
        )
        
        # ========== 构建学习材料 ========== #
        learning_block = ""
        
        use_top = top_factors or []
        use_bottom = bottom_factors or []
        use_negative = negative_factors or []
        
        # ========== PART 1: Success Patterns ========== #
        if use_top:
            learning_block += "=" * 75 + "\n"
            learning_block += "SUCCESS PATTERNS - Learn WHY and HOW they work\n"
            learning_block += "=" * 75 + "\n\n"
            
            for i, p in enumerate(use_top, 1):
                code = p.get('code', '').strip()[:200]
                report = p.get('report', '').strip()
                score = p.get('train_score', 0)
                
                # ========== 使用深度提取函数 ========== #
                logic = self._extract_economic_logic_deep(report)
                success = self._extract_success_factors_deep(report)
                ideas = self._extract_improvement_ideas_deep(report)
                
                learning_block += f"✅ SUCCESS #{i} (Train Score: {score:.3f})\n\n"
                learning_block += f"Code:\n{code}\n\n"
                
                if logic:
                    learning_block += f"📊 Economic Logic:\n   {logic}\n\n"
                if success:
                    learning_block += f"🎯 Why It Works:\n   {success}\n\n"
                if ideas:
                    learning_block += f"💡 Variations to Try:\n   {ideas}\n\n"
                
                learning_block += "-" * 75 + "\n\n"
        
        # ========== PART 2: Failure Patterns ========== #
        if use_bottom:
            learning_block += "=" * 75 + "\n"
            learning_block += "FAILURE PATTERNS - Learn what to AVOID\n"
            learning_block += "=" * 75 + "\n\n"
            
            for i, p in enumerate(use_bottom, 1):
                code = p.get('code', '').strip()[:180]
                report = p.get('report', '').strip()
                
                # ========== 使用深度提取函数 ========== #
                failure = self._extract_failure_analysis_deep(report)
                
                learning_block += f"❌ FAILURE #{i}\n\n"
                learning_block += f"Code:\n{code}\n\n"
                
                if failure:
                    learning_block += f"⚠️  Why It Failed:\n   {failure}\n\n"
                
                learning_block += "-" * 75 + "\n\n"
        
        # ========== PART 3: Anti-Patterns ========== #
        if use_negative:
            learning_block += "=" * 75 + "\n"
            learning_block += "ANTI-PATTERNS - NEVER repeat these mistakes\n"
            learning_block += "=" * 75 + "\n\n"
            
            for i, p in enumerate(use_negative, 1):
                code = p.get('code', '').strip()[:150]
                report = p.get('report', '').strip()[:150]
                
                learning_block += f"🚫 Anti-Pattern #{i}\n"
                learning_block += f"   Code: {code}\n"
                learning_block += f"   Problem: {report}\n\n"
        
        if not learning_block:
            learning_block = "(No historical factors available)\n\n"
        
        # ========== 任务指导（三层学习框架）========== #
        task = f"""
    {"=" * 75}
    YOUR TASK: Generate {target_n} factors that OUTPERFORM previous attempts
    {"=" * 75}
    
    **THREE-LEVEL LEARNING FRAMEWORK:**
    
    ┌──────────────────────────────────────────────────────────┐
    │ LEVEL 1: UNDERSTAND WHY (Economic Logic)                 │
    └──────────────────────────────────────────────────────────┘
    
    From SUCCESS PATTERNS above:
    → What financial relationships do they capture?
    → Key combinations: Profitability × Efficiency, Growth × Quality, Momentum × Fundamentals
    
    Your NEW factors MUST have clear economic interpretation.
    
    ┌──────────────────────────────────────────────────────────┐
    │ LEVEL 2: LEARN HOW (Technical Implementation)            │
    └──────────────────────────────────────────────────────────┘
    
    From SUCCESS PATTERNS above:
    → Robust transforms: rank(), pct_change(), rolling().mean(), quantile()
    → Proper normalization: division by scale factors, cross-sectional ranking
    → Zero handling: np.where(denom != 0, num/denom, 0)
    
    Your NEW factors MUST be technically sound.
    
    ┌──────────────────────────────────────────────────────────┐
    │ LEVEL 3: AVOID MISTAKES (What NOT to do)                 │
    └──────────────────────────────────────────────────────────┘
    
    From FAILURE PATTERNS:
    → Understand their core flaws
    → Don't repeat similar mistakes
    
    From ANTI-PATTERNS:
    → NEVER use those fundamental errors
    → Ensure opposite characteristics
    
    **CONCRETE DESIGN PATTERNS (Use these as templates):**
    
    Pattern A - Profitability Ratio:
      Formula: (Operating Income / Total Assets).rank()
      Why: Captures efficiency, cross-sectionally normalized
      
    Pattern B - Growth Quality:
      Formula: (Revenue.pct_change() × (Revenue / Assets)).rank()
      Why: Combines growth momentum + operational efficiency
    
    Pattern C - Cash Quality:
      Formula: (Operating Cash Flow / Net Income).rolling(4).mean()
      Why: Earnings quality, time-smoothed
    
    Pattern D - Efficiency Momentum:
      Formula: ((Revenue / Assets) / (Revenue / Assets).shift(4)).rank()
      Why: Improving efficiency trend
    
    **FIELD COMBINATIONS THAT WORK:**
    
    Strong Predictors:
    - revtq (revenue) / atq (assets) → efficiency
    - ibq (income) / atq (assets) → profitability
    - oancfq (cash flow) / saleq (sales) → quality
    
    Growth Signals:
    - revtq.pct_change() → revenue growth
    - niq.pct_change() → earnings growth
    
    Quality Indicators:
    - oancfq / niq → cash quality
    - (actq - lctq) / atq → working capital ratio
    
    **FINAL CHECKLIST (Verify EACH factor before including):**
    
    ✓ Has clear economic interpretation (can explain WHY it predicts returns)
    ✓ Uses 2-4 relevant fields (not too simple, not too complex)
    ✓ Applies robust transformation (rank/pct_change/rolling/quantile)
    ✓ Handles zeros properly: np.where(denom != 0, ...)
    ✓ Is DIFFERENT from failed factors
    ✓ Is DIFFERENT from anti-patterns
    ✓ Would plausibly work in real markets
    
    Now generate EXACTLY {target_n} NEW factors following ALL guidelines above.
    Return ONLY JSON with no extra text.
    """
        
        return (
            rules +
            self._whitelist_block() +
            learning_block +
            task
        )
    
    
    def _build_refill_prompt(
        self, 
        missing: int, 
        reasons: List[str], 
        existing_codes: List[str] = None
    ) -> str:
        """构建补货 Prompt"""
        reasons_txt = ""
        if reasons:
            reasons_txt = "Previously rejected: " + "; ".join(set(reasons[:6])) + "\n"
        
        existing_txt = ""
        if existing_codes:
            shown = existing_codes[:10]
            existing_txt = (
                "ALREADY GENERATED (DO NOT REPEAT):\n" +
                "\n".join(f"  - {code[:120]}" for code in shown) +
                "\n\n**GENERATE COMPLETELY DIFFERENT FACTORS!**\n\n"
            )
        
        return (
            "REFILL REQUEST:\n" +
            reasons_txt +
            existing_txt +
            f"Generate EXACTLY {missing} NEW and UNIQUE factors.\n" +
            f"Return ONLY JSON {{\"factors\":[...]}} with EXACTLY {missing} items.\n" +
            "Same STRICT rules: single statement, no semicolons.\n" +
            self._whitelist_block()
        )

    # ================= 工具方法 =================
    _DATA_ASSIGN = re.compile(r"data\[['\"]factor_score['\"]\]\s*=", flags=re.IGNORECASE)
    _COLREF = re.compile(r"data\[['\"]([A-Za-z0-9_]+)['\"]\]")

    def _norm_key(self, code: str) -> str:
        """代码归一化（用于去重）"""
        c = re.sub(r"#.*", "", code or "")
        c = c.replace('"', "'")
        c = re.sub(r"\s+", "", c)
        c = c.replace(";;", ";").strip(";")
        return c.lower()

    def _extract_columns(self, code: str) -> List[str]:
        """提取代码中使用的列名"""
        cols = [m.group(1) for m in self._COLREF.finditer(code or "")]
        return [c for c in cols if c.lower() != "factor_score"]
    
    
    # ================= 代码清洗与校验 =================
    def _force_single_assignment(self, code: str) -> str:
        """强制重写为单语句"""
        if not isinstance(code, str):
            return ""
        s = code.strip()
        s = self._preclean_json_text(s)
        
        m = re.search(r"data\[['\"]factor_score['\"]\]\s*=\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        rhs = m.group(1).strip()
        
        # 处理多行（只取第一条）
        for sep in [";", "\n", "\r"]:
            p = rhs.find(sep)
            if p != -1:
                rhs = rhs[:p].strip()
                break
        
        if rhs.startswith("{"):
            return ""
        
        # 检查是否已经是完整格式
        if (rhs.startswith("pd.Series(") and 
            ".replace([np.inf,-np.inf],np.nan)" in rhs and 
            ".fillna(0)" in rhs):
            return f"data['factor_score']={rhs}"
        
        # 包装
        return (
            "data['factor_score']=pd.Series("
            f"{rhs}"
            ",index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)"
        )

    def _all_columns_allowed(self, code: str, rejected: List[str]) -> bool:
        """检查列名是否在白名单"""
        if not self.field_set_lower:
            return True
        cols = self._extract_columns(code)
        for c in cols:
            if c.lower() not in self.field_set_lower:
                rejected.append(f"illegal_column:{c}")
                return False
        return True

    def _forbidden_scan(self, code: str, rejected: List[str]) -> bool:
        """扫描禁用词"""
        low = code.lower()
        for tok in self.forbidden_tokens:
            if tok in low:
                rejected.append(f"forbidden:{tok.strip()}")
                return False
        for tok in self.lookahead_tokens:
            if tok in low:
                rejected.append(f"lookahead:{tok.strip()}")
                return False
        return True

    def _sanitize_one(self, code: str, rejected: List[str]) -> Optional[str]:
        """清洗并校验单个代码"""
        if not code or len(code) > self.max_code_len:
            rejected.append("too_long_or_empty")
            return None

        single = self._force_single_assignment(code)
        if not single:
            rejected.append("cannot_force_single")
            return None

        c = single.strip()
        if not self._forbidden_scan(c, rejected):
            return None
        if "data['factor_score']" not in c or "=" not in c:
            rejected.append("missing_assignment")
            return None
        if not self._all_columns_allowed(c, rejected):
            return None
        return c

    def _preclean_json_text(self, text: str) -> str:
        """剥掉 ```json / ``` 包裹"""
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
        if raw.lower().startswith("json\n"):
            raw = raw[5:].lstrip()
        l = raw.find("{"); r = raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            raw = raw[l:r+1]
        return raw

    def _try_parse_json_factors(self, text: str) -> Tuple[List[str], List[str]]:
        """解析 {"factors":[...]} JSON"""
        reject: List[str] = []
        raw = self._preclean_json_text(text)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or "factors" not in obj or not isinstance(obj["factors"], list):
                reject.append("json_structure_invalid")
                return [], reject
            items = obj["factors"]
            out: List[str] = []
            for it in items:
                if not isinstance(it, dict):
                    reject.append("item_not_object")
                    continue
                code = str(it.get("code", "")).strip()
                if not code:
                    reject.append("empty_code")
                    continue
                s = self._sanitize_one(code, reject)
                if not s:
                    continue
                out.append(s)
            return out, reject
        except Exception as e:
            reject.append(f"json_parse_error:{type(e).__name__}")
            return [], reject

    def _split_candidates(self, text: str) -> List[str]:
        """从文本中切出候选代码片段"""
        t = text or ""
        out = []
        for m in self._DATA_ASSIGN.finditer(t):
            start = m.start()
            chunk = t[start:start + 1200]
            next_m = self._DATA_ASSIGN.search(t, pos=m.end())
            if next_m and (next_m.start() - start) < 1200:
                chunk = t[start:next_m.start()]
            else:
                end_hint = max(chunk.find("\n\n"), chunk.rfind("}"), chunk.rfind("]"))
                if end_hint != -1 and end_hint > 80:
                    chunk = chunk[:end_hint]
            out.append(chunk.strip())
        return out

    def _parse_and_transform(self, text: str, target_n: int) -> List[str]:
        """解析 GPT 响应"""
        self._last_reject_reasons: List[str] = []

        # 1) 严格 JSON
        codes, reject = self._try_parse_json_factors(text)
        self._last_reject_reasons.extend(reject)
        if len(codes) >= target_n:
            self.logger.debug(f"[positive] reject_reasons={list(set(self._last_reject_reasons))[:12]}")
            return codes[:target_n]

        # 2) 回退：按前缀切片
        cands = self._split_candidates(text)
        cleaned = list(codes)
        seen = set(self._norm_key(c) for c in cleaned)

        for c in cands:
            s = self._sanitize_one(c, self._last_reject_reasons)
            if not s:
                continue
            k = self._norm_key(s)
            if k in seen:
                continue
            cleaned.append(s)
            seen.add(k)
            if len(cleaned) >= target_n:
                break

        self.logger.debug(f"[positive] reject_reasons={list(set(self._last_reject_reasons))[:12]}")
        return cleaned

    # ================= 日志保存 =================
    def _save_gpt_response(
        self, 
        round_num: int, 
        prompt: str, 
        resp: str, 
        suffix: str = ""
    ) -> None:
        """保存 GPT 响应到日志文件"""
        _ensure_dir(self.logs_dir)
        fn = self.logs_dir / f"round_{round_num:02d}_positive_agent{suffix}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({
                "round": round_num,
                "prompt_preview": prompt[:1200],
                "response": resp[:50000]
            }, f, ensure_ascii=False, indent=2)
