# experiments/iterative_baseline_with_reports/positive_agents_reports.py
"""
正向因子生成代理（带报告版本）
核心特点：
1. 基于 iterative_baseline/positive_agents.py 的架构
2. 输入改为 code + report（而非 code + train_score）
3. Prompt 构建：自然语言展示历史因子分析
4. 保持健壮的去重、安全校验、补货逻辑
5. 统一因子分配方案（Scheme A / C），不区分因子数量
"""

import json
import math
import re
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 路径：把仓库根目录加入 sys.path，便于导入 gpt_runner 与 column_desc
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


# ==================== 统一因子分配函数 ==================== #
def allocate_factors(N: int, scheme: str = "A") -> Tuple[int, int, int]:
    """
    统一因子分配（不区分因子数量，固定比例）
    
    Scheme A: top 35% + middle 30% + bottom 35%（全部带报告）
    Scheme C: top 35% + bottom 35%（丢弃中间 30%，无 middle）
    
    两种方案的 top 和 bottom 选取完全一致，便于消融实验对比。
    
    Args:
        N: 总因子数量
        scheme: "A" 或 "C"
    
    Returns:
        (top_k, middle_k, bottom_k)
    """
    top_k = round(N * 0.35)
    bottom_k = round(N * 0.35)
    
    # 确保 top_k 和 bottom_k 至少为 1（N 极小时）
    top_k = max(1, top_k)
    bottom_k = max(1, bottom_k)
    
    # 防止 top + bottom 超过 N
    if top_k + bottom_k > N:
        top_k = math.ceil(N / 2)
        bottom_k = N - top_k
    
    if scheme.upper() == "C":
        middle_k = 0  # 丢弃中间 30%
    else:  # Scheme A
        middle_k = N - top_k - bottom_k
    
    return top_k, middle_k, bottom_k


class PositiveAgent:
    """
    正向因子生成代理（带报告版本）
    
    核心改动：
    - 输入：previous_factors = [{'code': '...', 'report': '...'}, ...]
    - Prompt：展示自然语言分析（而非数值分数）
    - 统一因子分配方案（Scheme A / C）
    """

    # ================= 初始化 =================
    def __init__(self):
        self.logger = logging.getLogger('positive_agents_reports')
        self.temperature = float(_cfg_get('REPORT_TEMPERATURE', _cfg_get('TEMPERATURE', 0.7)))
        self.max_tokens = int(_cfg_get('FACTOR_GENERATION_MAX_TOKENS', _cfg_get('MAX_TOKENS', 900)))
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', _cfg_get('N_FACTORS', 10)))
        self.logs_dir = Path(_cfg_get('LOGS_DIR', _CUR / "logs"))
        _ensure_dir(self.logs_dir)

        # 因子分配方案（新增）
        self.allocation_scheme = str(_cfg_get('ALLOCATION_SCHEME', 'A')).upper()
        self.logger.info(f"因子分配方案: Scheme {self.allocation_scheme}")

        # 列名白名单（严格校验用）
        self.field_whitelist: List[str] = [str(c) for c in FIELD_WHITELIST]
        self.field_set_lower = {c.lower(): c for c in self.field_whitelist}
        self.logger.info(f"字段白名单载入：{len(self.field_whitelist)} 列")

        # 解析与安全限制
        self.max_code_len = int(_cfg_get('MAX_CODE_LEN', 1500))
        self.forbidden_tokens = ['import', 'def ', 'class ', 'for ', 'while ', 'eval(', 'exec(']
        self.lookahead_tokens = ['shift(-', 'lead(', 'future']
        self.blacklist_words = ['price', 'volume', 'high', 'low']

        self.strict_memory = bool(_cfg_get('STRICT_MEMORY', True))
        self.top_k_prev = int(_cfg_get('TOP_K_FACTORS', 5))
        self.refill_max_attempts = int(_cfg_get('POS_AGENT_REFILL_ATTEMPTS', 3))
        self.batch_size = int(_cfg_get('POS_AGENT_BATCH', 5))

    # ================= 对外主函数 =================
    def generate_optimized_factors(
        self,
        previous_factors: List[Dict],  # ← [{'code': '...', 'report': '...'}, ...]
        round_num: int,
        save_response: bool = True,
        n_override: Optional[int] = None,
        existing_codes: Optional[List[str]] = None
    ) -> List[str]:
        """
        基于上一轮的因子分析生成新因子
        
        Args:
            previous_factors: 上一轮因子列表
                格式: [{'code': '...', 'report': '...'}, ...]
            round_num: 当前轮次
            save_response: 是否保存 GPT 响应
            n_override: 覆盖默认的因子数量
            existing_codes: 已存在的代码（用于去重）
        
        Returns:
            生成的因子代码列表
        """
        target_n = int(n_override) if (n_override and n_override > 0) else self.factors_per_round

        # ========== 对因子进行排序和分组 ========== #
        prev_pairs_with_score = []
        for f in previous_factors or []:
            code = str(f.get('code', '')).strip()
            if not code:
                continue
            report = str(f.get('report', '')).strip()
            val_score = f.get('val_score', -999)
            train_score_val = f.get('train_score', -999)
            
            prev_pairs_with_score.append({
                'code': code, 
                'report': report, 
                'train_score': train_score_val,
                'val_score': val_score,
            })

        # 按 val_score 排序（内部排序，不传给 GPT）
        prev_pairs_with_score.sort(key=lambda x: x.get('val_score', -999), reverse=True)

        # ========== 统一比例分配（不区分因子数量）========== #
        N = len(prev_pairs_with_score)
        top_k, middle_k, bottom_k = allocate_factors(N, self.allocation_scheme)
        
        # 安全裁剪
        top_k = min(top_k, N)
        bottom_k = min(bottom_k, max(0, N - top_k))
        middle_k = min(middle_k, max(0, N - top_k - bottom_k))
        
        self.logger.info(
            f"因子分配 (Scheme {self.allocation_scheme}): "
            f"N={N}, top={top_k}, middle={middle_k}, bottom={bottom_k}"
        )
        
        # 分离 top / middle / bottom
        top_factors = prev_pairs_with_score[:top_k] if prev_pairs_with_score else []
        bottom_factors = prev_pairs_with_score[-bottom_k:] if bottom_k > 0 and len(prev_pairs_with_score) > bottom_k else prev_pairs_with_score[-bottom_k:] if bottom_k > 0 else []
        
        # Middle: 从 top 和 bottom 之间均匀采样
        if middle_k > 0:
            mid_start = top_k
            mid_end = max(mid_start, N - bottom_k)
            middle_pool = prev_pairs_with_score[mid_start:mid_end]
            if len(middle_pool) >= middle_k:
                step = max(1, len(middle_pool) // middle_k)
                middle_factors = [middle_pool[i * step] for i in range(middle_k)]
            else:
                middle_factors = middle_pool
        else:
            middle_factors = []

        # 保持兼容性
        prev_pairs = prev_pairs_with_score

        if not prev_pairs:
            self.logger.warning("上一轮有效记忆为空（需要包含 code+report）。将以零上下文请求 GPT。")

        # ========= 分批首呼 ========= #
        codes: List[str] = []
        need = target_n
        batch = max(1, min(self.batch_size, need))
        batch_idx = 0
        while need > 0:
            batch_idx += 1
            prompt = self._build_prompt(
                prev_pairs, 
                batch,
                top_factors=top_factors,
                bottom_factors=bottom_factors,
                middle_factors=middle_factors
            )
            self.logger.info(f"Round {round_num}: 调用 GPT 生成 {batch} 个新因子")
            resp = call_gpt(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
            if not isinstance(resp, str):
                resp = str(resp or '')
            if save_response:
                self._save_gpt_response(round_num, prompt, resp, suffix=f"_batch_{batch_idx}_{batch}")
            self.logger.info(f"Round {round_num}: GPT首呼返回前200字：{resp[:200].replace(os.linesep,' ')}")

            got = self._parse_and_transform(resp, batch)

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

        # 补货
        attempts = 0
        consecutive_no_progress = 0
        last_count = 0
        while len(codes) < target_n and attempts < self.refill_max_attempts:
            attempts += 1
            last_count = len(codes)
            missing = target_n - len(codes)
            self.logger.info(f"Round {round_num}: 解析不足 {len(codes)}/{target_n}，补货 {missing}（attempt {attempts}/{self.refill_max_attempts}）")

            reasons = self._last_reject_reasons if hasattr(self, "_last_reject_reasons") else []
            refill_prompt = self._build_refill_prompt(
                missing, 
                reasons,
                existing_codes=codes
            )
            resp2 = call_gpt(refill_prompt, temperature=min(self.temperature + 0.08, 0.5),
                             max_tokens=max(600, self.max_tokens))
            if not isinstance(resp2, str):
                resp2 = str(resp2 or '')
            if save_response:
                self._save_gpt_response(round_num, refill_prompt, resp2, suffix=f"_refill_{attempts}")
            self.logger.info(f"Round {round_num}: 补货返回前200字：{resp2[:200].replace(os.linesep,' ')}")
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
                    self.logger.warning(f"连续 {consecutive_no_progress} 次补货无效（生成的都是重复因子），停止补货")
                    break
            else:
                consecutive_no_progress = 0

        return codes

    # ================= Prompt 构造 =================
    def _whitelist_block(self) -> str:
        if not self.field_whitelist:
            return ""
        cols = "\n".join(self.field_whitelist)
        return (
            "ALLOWED COLUMNS (USE ONLY THESE; any other names are invalid and will be discarded):\n"
            f"{cols}\n\n"
        )
    
    def _build_prompt(
        self, 
        prev_pairs: List[Dict],
        target_n: int,
        top_factors: Optional[List[Dict]] = None,    
        bottom_factors: Optional[List[Dict]] = None,   
        middle_factors: Optional[List[Dict]] = None,
    ) -> str:
        """
        构建生成 Prompt（带报告版本）
        
        根据 allocation_scheme 自动调整展示策略：
        - Scheme A: Top(strengths) + Middle(neutral完整报告) + Bottom(weaknesses)
        - Scheme C: Top(strengths) + Bottom(weaknesses)，无 middle
        """
        rules = (
            "You generate quarterly factor code on a pandas DataFrame named `data`.\n"
            "RETURN (STRICT): ONLY a JSON object {\"factors\": [{\"id\":\"r01_01\",\"code\":\"...\"}, ...]} with EXACTLY "
            f"{target_n} items.\n"
            "- JSON schema: factors = Array of objects with keys {id: string, code: string}. "
            "Never return an array of strings; such output will be rejected.\n"
            "- ONE SINGLE STATEMENT per item. NO semicolons. NO second assignment. NO prose. NO code fences. DO NOT prepend `json`.\n"
            "- Each `code` must be exactly:\n"
            "  data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)\n"
            "- Use ONLY allowed columns (whitelist below). Handle zero-divisions inside <EXPR> with np.where.\n"
            "- For <EXPR> avoid future/lead/shift(-k); no loops/imports/functions.\n"
            "- Keep code concise (≤ 240 chars).\n\n"
            "**CRITICAL - AVOID NUMPY ARRAY METHODS:**\n"
            "- NEVER write: np.where(...).rank() or np.where(...).rolling() or np.where(...).pct_change()\n"
            "- np.where() returns a numpy array, which does NOT have .rank(), .rolling(), .shift(), .pct_change() methods\n"
            "- If you need these transformations, apply them BEFORE np.where or on pandas Series directly:\n"
            "  ✓ CORRECT: (data['col1']/data['col2']).rolling(4).mean()\n"
            "  ✓ CORRECT: data['col1'].rank() / data['col2']\n"
            "  ✗ WRONG: np.where(data['col']==0, 0, data['col1']/data['col2']).rolling(4)\n"
            "  ✗ WRONG: np.where(data['col']==0, 0, expr).rank()\n\n"
        )
        
        # ========== 对比学习式历史展示 ========== #
        history_block = ""

        use_top = top_factors if top_factors else []
        use_bottom = bottom_factors if bottom_factors else []
        use_middle = middle_factors if middle_factors else []

        # fallback（无分组传入时）
        if not use_top and not use_bottom and prev_pairs:
            n = len(prev_pairs)
            top_n = max(1, min(3, n // 3))
            bottom_n = max(1, min(2, n // 3))
            use_top = prev_pairs[:top_n]
            use_bottom = prev_pairs[-bottom_n:] if n > bottom_n else []

        # === Top 因子：strengths 报告 ===
        if use_top:
            history_block += "=== HIGH-PERFORMING FACTORS (Learn from these patterns) ===\n\n"
            for i, p in enumerate(use_top, 1):
                code = p.get('code', '').strip()
                report = p.get('report', '').strip()
                score = p.get('train_score', 'N/A')
                
                strengths = self._extract_strengths(report)
                
                history_block += f"Top Factor #{i} (Train Score: {score}):\n"
                history_block += f"```python\n{code[:250]}\n```\n"
                if strengths:
                    history_block += f"✓ Key Strengths: {strengths}\n"
                history_block += "\n"

        # === Middle 因子：仅 Scheme A 展示，使用完整 neutral 报告 ===
        if use_middle:
            history_block += "=== MIDDLE-PERFORMING FACTORS (Understand what's average) ===\n\n"
            for i, p in enumerate(use_middle, 1):
                code = p.get('code', '').strip()
                report = p.get('report', '').strip()
                score = p.get('train_score', 'N/A')
                
                history_block += f"Middle Factor #{i} (Train Score: {score}):\n"
                history_block += f"```python\n{code[:250]}\n```\n"
                if report:
                    # Scheme A: 展示完整报告（neutral，既有优点也有不足）
                    history_block += f"Analysis:\n{report[:400]}\n"
                history_block += "\n"

        # === Bottom 因子：weaknesses 报告 ===
        if use_bottom:
            history_block += "=== LOW-PERFORMING FACTORS (Avoid these mistakes) ===\n\n"
            for i, p in enumerate(use_bottom, 1):
                code = p.get('code', '').strip()
                report = p.get('report', '').strip()
                score = p.get('train_score', 'N/A')
                
                weaknesses = self._extract_weaknesses(report)
                
                history_block += f"Failed Factor #{i} (Train Score: {score}):\n"
                history_block += f"```python\n{code[:200]}\n```\n"
                if weaknesses:
                    history_block += f"✗ Main Issues: {weaknesses}\n"
                history_block += "\n"

        if not history_block:
            history_block = "(No previous factors available - generating from scratch)\n\n"
        
        # === 学习策略（根据 scheme 调整）===
        if use_middle:
            # Scheme A: 有 middle
            learning_strategy = (
                "**Learning Strategy:**\n"
                "1. BUILD ON top factors: Use similar calculation patterns, field combinations, and transformations\n"
                "2. LEARN FROM middle factors: Understand what makes them average - they have partial merit but need improvement\n"
                "3. AVOID bottom factors: Don't repeat their mistakes (high drawdown, low coverage, poor diversification)\n"
                "4. INTRODUCE VARIATIONS: Don't copy exactly - try different field ratios, time windows, or normalizations\n\n"
            )
        else:
            # Scheme C: 无 middle，纯正负对比
            learning_strategy = (
                "**Learning Strategy:**\n"
                "1. BUILD ON top factors: Use similar calculation patterns, field combinations, and transformations\n"
                "2. AVOID bottom factors: Don't repeat their mistakes (high drawdown, low coverage, poor diversification)\n"
                "3. INTRODUCE VARIATIONS: Don't copy exactly - try different field ratios, time windows, or normalizations\n\n"
            )
        
        return (
            rules +
            self._whitelist_block() +
            history_block +
            "=== YOUR TASK ===\n" +
            f"Generate EXACTLY {target_n} NEW factors that OUTPERFORM previous attempts.\n\n" +
            learning_strategy +
            "**Code-Level Guidelines:**\n" +
            "- Use robust transformations on pandas Series: (data['col1']/data['col2']).rank(), .rolling().mean()\n" +
            "- Remember: Call pandas methods (.rank, .rolling) on data['col'] or expressions, NOT on np.where()\n" +
            "- Combine different financial dimensions (profitability + efficiency, growth + quality)\n" +
            "- Apply proper normalization to reduce outliers\n" +
            "- Ensure adequate data coverage (avoid excessive NaN)\n\n" 
        )

    # ========== numpy array 方法检测 ========== #
    def _check_numpy_array_methods(self, code: str, rejected: List[str]) -> bool:
        """
        检测是否在 numpy array 上调用 pandas 方法
        """
        import re
        
        if 'np.where' in code:
            for match in re.finditer(r'np\.where', code, re.IGNORECASE):
                start = match.start()
                snippet = code[start:start+200]
                
                method_pattern = r'\)\s*\.\s*(rank|rolling|pct_change|shift|diff|resample|cumsum|cumprod)\s*\('
                if re.search(method_pattern, snippet, re.IGNORECASE):
                    match_obj = re.search(method_pattern, snippet, re.IGNORECASE)
                    if match_obj:
                        rejected.append(f"numpy_array_method:{match_obj.group(0)[:50]}")
                        return False
        
        return True

    # ========== 补货 prompt ========== #
    def _build_refill_prompt(self, missing: int, reasons: List[str], existing_codes: List[str] = None) -> str:
        """构建补货 prompt"""
        reasons_txt = ""
        if reasons:
            reasons_txt = "Previously rejected reasons: " + "; ".join(set(reasons[:6])) + "\n"
            if any('numpy_array_method' in r for r in reasons):
                reasons_txt += "\n**CRITICAL**: Some factors were rejected for calling pandas methods on numpy arrays.\n"
                reasons_txt += "Remember: np.where() returns numpy array, use pandas Series for .rank()/.rolling() etc.\n"
        
        existing_txt = ""
        if existing_codes:
            shown = existing_codes[:10]
            existing_txt = (
                "CRITICAL - ALREADY GENERATED (DO NOT REPEAT):\n" +
                "\n".join(f"  - {code[:120]}" for code in shown) +
                "\n\n**YOU MUST GENERATE COMPLETELY DIFFERENT FACTORS!**\n" +
                "Use different field combinations, different calculations!\n\n"
            )
        
        return (
            "REFILL REQUEST:\n" +
            reasons_txt +
            existing_txt +
            f"Generate EXACTLY {missing} NEW and UNIQUE factors.\n" +
            "Return ONLY JSON {\"factors\":[...]} with EXACTLY "
            f"{missing} items, same STRICT rules (single statement only). "
            "JSON schema: factors = Array of objects with keys {id: string, code: string}. "
            "Do NOT return an array of strings; it will be discarded.\n\n"
            "**REMINDER**: Do NOT call .rank()/.rolling()/.pct_change() on np.where() results!\n"
            "These methods only work on pandas Series, not numpy arrays.\n\n" +
            self._whitelist_block() +
            "Do not use semicolons, no second assignment, no code fences, no `json` prefix."
        )

    # ================= 解析与清洗 =================
    _DATA_ASSIGN = re.compile(r"data\[['\"]factor_score['\"]\]\s*=", flags=re.IGNORECASE)
    _COLREF = re.compile(r"data\[['\"]([A-Za-z0-9_]+)['\"]\]")

    def _norm_key(self, code: str) -> str:
        c = re.sub(r"#.*", "", code or "")
        c = c.replace('"', "'")
        c = re.sub(r"\s+", "", c)
        c = c.replace(";;", ";").strip(";")
        return c.lower()

    def _split_candidates(self, text: str) -> List[str]:
        """从任意位置切出以 data['factor_score']= 起头的片段"""
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

    def _extract_columns(self, code: str) -> List[str]:
        cols = [m.group(1) for m in self._COLREF.finditer(code or "")]
        return [c for c in cols if c.lower() != "factor_score".lower()]
    
    def _extract_strengths(self, report: str, max_chars: int = 200) -> str:
        """从报告中提取成功要素"""
        if not report or not report.strip():
            return ""
        
        success_keywords = [
            'strength', 'good', 'high', 'effective', 'excellent', 
            'consistent', 'robust', 'advantage', 'positive', 'strong'
        ]
        
        lines = report.split('\n')
        relevant = []
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in success_keywords):
                relevant.append(line.strip())
                if len(relevant) >= 2:
                    break
        
        if relevant:
            result = ". ".join(relevant)
            if len(result) > max_chars:
                result = result[:max_chars] + "..."
            return result
        
        return report[:150].strip() + "..." if len(report) > 150 else report.strip()
    
    def _extract_weaknesses(self, report: str, max_chars: int = 200) -> str:
        """从报告中提取失败原因"""
        if not report or not report.strip():
            return ""
        
        failure_keywords = [
            'weakness', 'weak', 'poor', 'low', 'risk', 'problem', 
            'issue', 'drawdown', 'avoid', 'negative', 'bad', 'insufficient'
        ]
        
        lines = report.split('\n')
        relevant = []
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in failure_keywords):
                relevant.append(line.strip())
                if len(relevant) >= 2:
                    break
        
        if relevant:
            result = ". ".join(relevant)
            if len(result) > max_chars:
                result = result[:max_chars] + "..."
            return result
        
        return report[:150].strip() + "..." if len(report) > 150 else report.strip()

    def _force_single_assignment(self, code: str) -> str:
        """强制重写为单语句（改进版 - 处理 JSON 污染）"""
        if not isinstance(code, str):
            return ""
        s = code.strip()
        s = self._preclean_json_text(s)
        
        if '"},{"' in s or '\"},{"' in s:
            json_marker_pos = s.find('"},{"')
            if json_marker_pos == -1:
                json_marker_pos = s.find('\"},{"')
            if json_marker_pos != -1:
                s = s[:json_marker_pos].strip()
        
        m = re.search(r"data\[['\"]factor_score['\"]\]\s*=\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        rhs = m.group(1).strip()
        
        for json_end in ['"}', '"},{', '\"},', '\"},{']:
            if json_end in rhs:
                pos = rhs.find(json_end)
                rhs = rhs[:pos].strip()
        
        for sep in [";", "\n", "\r"]:
            p = rhs.find(sep)
            if p != -1:
                rhs = rhs[:p].strip()
                break
        
        if rhs.startswith("{"):
            return ""
        
        if (rhs.startswith("pd.Series(") and 
            ".replace([np.inf,-np.inf],np.nan)" in rhs and 
            ".fillna(0)" in rhs):
            return f"data['factor_score']={rhs}"
        
        return (
            "data['factor_score']=pd.Series("
            f"{rhs}"
            ",index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)"
        )

    def _all_columns_allowed(self, code: str, rejected: List[str]) -> bool:
        if not self.field_set_lower:
            return True
        cols = self._extract_columns(code)
        for c in cols:
            if c.lower() not in self.field_set_lower:
                rejected.append(f"illegal_column:{c}")
                return False
        return True

    def _forbidden_scan(self, code: str, rejected: List[str]) -> bool:
        low = code.lower()
        for tok in self.forbidden_tokens:
            if tok in low:
                rejected.append(f"forbidden:{tok.strip()}")
                return False
        for tok in self.lookahead_tokens:
            if tok in low:
                rejected.append(f"lookahead:{tok.strip()}")
                return False
        for w in self.blacklist_words:
            if w in low:
                rejected.append(f"keyword:{w}")
        return True

    def _no_bare_column_refs(self, code: str, rejected: List[str]) -> bool:
        """要求所有列访问都写成 data['col']"""
        if not self.field_whitelist:
            return True
        s = code
        for col in self.field_whitelist:
            pattern = r"(?<!data\[['\"])\b" + re.escape(col) + r"\b"
            if re.search(pattern, s):
                rejected.append(f"bare_column:{col}")
                return False
        return True

    def _sanitize_one(self, code: str, rejected: List[str]) -> Optional[str]:
        if not code or len(code) > self.max_code_len:
            rejected.append("too_long_or_empty")
            return None

        single = self._force_single_assignment(code)
        if not single:
            rejected.append("cannot_force_single")
            return None

        c = single.strip()

        if not self._check_numpy_array_methods(c, rejected):
            return None
        
        if not self._forbidden_scan(c, rejected):
            return None
        if not self._no_bare_column_refs(c, rejected):
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
        """尝试把 {"factors":[...]} 解析为代码列表"""
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

    def _parse_and_transform(self, text: str, target_n: int) -> List[str]:
        """解析 GPT 响应"""
        self._last_reject_reasons: List[str] = []

        codes, reject = self._try_parse_json_factors(text)
        self._last_reject_reasons.extend(reject)
        if len(codes) >= target_n:
            self.logger.debug(f"reject_reasons={list(set(self._last_reject_reasons))[:12]}")
            return codes[:target_n]

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

        self.logger.debug(f"reject_reasons={list(set(self._last_reject_reasons))[:12]}")
        return cleaned

    # ================= 日志保存 =================
    def _save_gpt_response(self, round_num: int, prompt: str, resp: str, suffix: str = "") -> None:
        _ensure_dir(self.logs_dir)
        fn = self.logs_dir / f"round_{round_num:02d}_positive_agent{suffix}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({
                "round": round_num,
                "strict_memory": self.strict_memory,
                "allocation_scheme": self.allocation_scheme,
                "prompt_preview": prompt[:1200],
                "response": resp[:50000]
            }, f, ensure_ascii=False, indent=2)
