# experiments/iterative_global_memory/positive_agents.py
"""
全局记忆因子生成代理（累积所有历史轮次）
核心特点：
1. 记忆所有历史轮次而非仅上一轮
2. 支持top-k高分 + bottom-k低分采样（对比学习）
3. 分析字段使用频率，鼓励低频字段探索
4. 支持多种生成策略（evolution/diversification/exploitation/auto_mix）
"""

import json
import re
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 导入共享模块
_CUR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.gpt_runner import call_gpt
from common.column_desc import COLUMN_DESC

# 导入配置
try:
    from .config import CONFIG as _CONFIG
except Exception:
    _CONFIG = None
try:
    from . import config as _cfg_module
except Exception:
    _cfg_module = None


def _cfg_get(key: str, default=None):
    """从CONFIG字典或config模块获取配置"""
    if isinstance(_CONFIG, dict) and key in _CONFIG:
        return _CONFIG.get(key, default)
    if hasattr(_cfg_module, key):
        return getattr(_cfg_module, key)
    return default


# 列名白名单
FIELD_WHITELIST = sorted(list(COLUMN_DESC.keys()))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


class GlobalMemoryAgent:
    """
    全局记忆因子生成代理
    与iterative_baseline的区别：累积所有历史轮次的记忆
    """

    # ================= 初始化 =================
    def __init__(self):
        self.logger = logging.getLogger('global_memory_agent')
        
        # GPT参数
        self.temperature = float(_cfg_get('GPT_TEMPERATURE', 0.7))
        self.max_tokens = int(_cfg_get('GPT_MAX_TOKENS', 900))
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', 10))
        
        # 日志目录
        self.logs_dir = Path(_cfg_get('LOGS_DIR', _CUR / "logs"))
        _ensure_dir(self.logs_dir)
        
        # 全局记忆采样参数（核心区别）
        self.global_top_k = int(_cfg_get('GLOBAL_TOP_K', 8))
        self.global_bottom_k = int(_cfg_get('GLOBAL_BOTTOM_K', 8))
        self.max_memory_items = int(_cfg_get('MAX_MEMORY_ITEMS', 2000))
        self.required_underused_ratio = float(_cfg_get('REQUIRED_UNDERUSED_RATIO', 0.4))
        
        # 列名白名单（严格校验用）
        self.field_whitelist: List[str] = FIELD_WHITELIST
        self.field_set_lower = {c.lower(): c for c in self.field_whitelist}
        self.logger.info(f"字段白名单载入：{len(self.field_whitelist)} 列")
        
        # 解析与安全限制
        self.max_code_len = int(_cfg_get('MAX_CODE_LEN', 1500))
        self.forbidden_tokens = ['import', 'def ', 'class ', 'for ', 'while ', 'eval(', 'exec(']
        self.lookahead_tokens = ['shift(-', 'lead(', 'future']
        
        # 补货参数
        self.refill_max_attempts = int(_cfg_get('REFILL_MAX_ATTEMPTS', 3))
        self.batch_size = int(_cfg_get('BATCH_SIZE', 5))

    # ================= 对外主函数 =================
    def generate_optimized_factors(
        self,
        cumulative_memory: List[Dict],  # ← 核心区别：传入所有历史轮次
        round_num: int,
        generation_strategy: str = "auto_mix",
        save_response: bool = True,
        n_override: Optional[int] = None,
        existing_codes: Optional[List[str]] = None
    ) -> List[str]:
        """
        基于全局记忆生成优化因子
        
        Args:
            cumulative_memory: 所有历史轮次的记忆列表
                格式：[{'round': 0, 'factors': [{'code': ..., 'train_score': ...}]}, ...]
            round_num: 当前轮次
            generation_strategy: 生成策略（evolution/diversification/exploitation/auto_mix）
            save_response: 是否保存GPT响应
            n_override: 覆盖默认的因子数量
            existing_codes: 已存在的因子代码（用于去重）
            
        Returns:
            生成的因子代码列表
        """
        target_n = int(n_override) if (n_override and n_override > 0) else self.factors_per_round
        
        # 分析全局记忆（核心区别）
        all_factors, top_factors, bottom_factors, field_usage = self._analyze_global_memory(
            cumulative_memory
        )
        
        self.logger.info(f"全局记忆统计：总因子数={len(all_factors)}, "
                        f"top-{self.global_top_k}={len(top_factors)}, "
                        f"bottom-{self.global_bottom_k}={len(bottom_factors)}")
        
        # 确定生成策略
        strategy = self._determine_strategy(generation_strategy, round_num)
        self.logger.info(f"Round {round_num}: 生成策略={strategy}")
        
        # ========= 分批首呼 ========= #
        codes: List[str] = []
        need = target_n
        batch = max(1, min(self.batch_size, need))
        batch_idx = 0
        
        while need > 0:
            batch_idx += 1
            prompt = self._build_prompt(
                all_factors=all_factors,      # ← 新增这一行
                top_factors=top_factors,
                bottom_factors=bottom_factors,
                field_usage=field_usage,
                strategy=strategy,
                target_n=batch
            )
            
            self.logger.info(f"Round {round_num}: 调用 GPT 生成 {batch} 个新因子（batch {batch_idx}）")
            resp = call_gpt(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
            if not isinstance(resp, str):
                resp = str(resp or '')
            
            if save_response:
                self._save_gpt_response(round_num, prompt, resp, suffix=f"_batch_{batch_idx}")
            
            self.logger.info(f"Round {round_num}: GPT返回前200字：{resp[:200].replace(os.linesep,' ')}")
            got = self._parse_and_transform(resp, batch)
            
            # 去重并并入
            seen = set(self._norm_key(c) for c in codes)
            for c in got:
                k = self._norm_key(c)
                if k not in seen and len(codes) < target_n:
                    codes.append(c)
                    seen.add(k)
            
            need = target_n - len(codes)
            batch = max(1, min(self.batch_size, need))
        
        # 去重：排除已存在的代码
        if existing_codes:
            exist_keys = set(self._norm_key(c) for c in existing_codes if isinstance(c, str))
            codes = [c for c in codes if self._norm_key(c) not in exist_keys]
        
        # 补货逻辑
        attempts = 0
        consecutive_no_progress = 0
        last_count = 0
        
        while len(codes) < target_n and attempts < self.refill_max_attempts:
            attempts += 1
            last_count = len(codes)
            missing = target_n - len(codes)
            
            self.logger.info(f"Round {round_num}: 补货 {missing} 个因子（attempt {attempts}/{self.refill_max_attempts}）")
            
            reasons = self._last_reject_reasons if hasattr(self, "_last_reject_reasons") else []
            refill_prompt = self._build_refill_prompt(
                missing=missing,
                reasons=reasons,
                existing_codes=codes
            )
            
            resp2 = call_gpt(
                refill_prompt,
                temperature=min(self.temperature + 0.08, 0.5),
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
            
            # 检查进展
            if len(codes) == last_count:
                consecutive_no_progress += 1
                if consecutive_no_progress >= 2:
                    self.logger.warning(f"连续 {consecutive_no_progress} 次补货无效，停止补货")
                    break
            else:
                consecutive_no_progress = 0
        
        return codes

    # ================= 全局记忆分析（核心区别）=================
    def _analyze_global_memory(
        self,
        cumulative_memory: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:
        """
        分析全局记忆，提取：
        1. 所有因子列表（完整）
        2. top-k高分因子
        3. bottom-k低分因子
        4. 字段使用频率统计
        
        Returns:
            (all_factors, top_factors, bottom_factors, field_usage)
        """
        # 收集所有因子
        all_factors = []
        for mem in (cumulative_memory or []):
            if not isinstance(mem, dict):
                continue
            
            round_num = mem.get('round', '?')  # ← 保留轮次信息
            factors = mem.get('factors', [])
            if not isinstance(factors, list):
                continue
            
            for f in factors:
                if not isinstance(f, dict):
                    continue
                code = str(f.get('code', '')).strip()
                if not code:
                    continue
                
                # 优先使用train_score
                score = f.get('train_score')
                if score is None:
                    score = f.get('val_score', 0.0)
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                
                # ← 保存因子（带轮次信息）
                all_factors.append({
                    'code': code,
                    'train_score': score,
                    'round': round_num  # ← 新增：记录来自哪一轮
                })
        
        # 限制记忆条目数量（防止token溢出）
        if len(all_factors) > self.max_memory_items:
            # 先按分数排序，保留高分的
            all_factors.sort(key=lambda x: x['train_score'], reverse=True)
            all_factors = all_factors[:self.max_memory_items]
            self.logger.warning(f"全局记忆过长，截断至{self.max_memory_items}条")
        
        # 排序（按train_score降序）
        all_factors.sort(key=lambda x: x['train_score'], reverse=True)
        
        # 提取top-k和bottom-k
        top_factors = all_factors[:self.global_top_k] if all_factors else []
        bottom_factors = all_factors[-self.global_bottom_k:] if len(all_factors) > self.global_bottom_k else []
        
        # 统计字段使用频率
        field_usage = {c: 0 for c in self.field_whitelist}
        for f in all_factors:
            code = f['code']
            for col in self.field_whitelist:
                if re.search(rf"data\['{re.escape(col)}'\]", code):
                    field_usage[col] += 1
        
        return all_factors, top_factors, bottom_factors, field_usage  # ← 返回all_factors

    def _determine_strategy(self, strategy: str, round_num: int) -> str:
        """确定生成策略（auto_mix会根据轮次自动切换）"""
        if strategy != "auto_mix":
            return strategy
        
        # auto_mix: 早期多样化 → 中期演化 → 后期收敛
        if round_num <= 2:
            return "diversification"
        elif round_num == 3:
            return "evolution"
        else:
            return "exploitation"

    # ================= Prompt 构造 =================
    def _whitelist_block(self) -> str:
        if not self.field_whitelist:
            return ""
        cols = "\n".join(self.field_whitelist)
        return (
            "ALLOWED COLUMNS (USE ONLY THESE; any other names are invalid):\n"
            f"{cols}\n\n"
        )
    
    def _build_prompt(
        self,
        all_factors: List[Dict],       # ← 修改1：新增参数
        top_factors: List[Dict],
        bottom_factors: List[Dict],
        field_usage: Dict[str, int],
        strategy: str,
        target_n: int
    ) -> str:
        """
        构建生成prompt（动态采样版本）
        
        策略：
        - 早期（总因子数≤50）：展示所有历史因子
        - 后期（总因子数>50）：智能采样（top + bottom + 随机中间）
        """
        
        # 基础规则
        rules = (
            "You generate quarterly factor code on a pandas DataFrame named `data`.\n"
            "RETURN (STRICT): ONLY a JSON object {\"factors\": [{\"id\":\"r01_01\",\"code\":\"...\"}, ...]} with EXACTLY "
            f"{target_n} items.\n"
            "- JSON schema: factors = Array of objects with keys {id: string, code: string}.\n"
            "- ONE SINGLE STATEMENT per item. NO semicolons. NO second assignment. NO prose. NO code fences.\n"
            "- Each `code` must be exactly:\n"
            "  data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)\n"
            "- Use ONLY allowed columns (whitelist below). Handle zero-divisions with np.where.\n"
            "- NO future/lead/shift(-k); no loops/imports/functions.\n"
            "- Keep code concise (≤ 240 chars).\n\n"
        )
        
        # ========== 动态采样逻辑（核心改动）========== #
        MAX_FACTORS_TO_SHOW = 50  # 可调整：GPT能处理的最大因子数
        
        # ← 修改2：删除了错误的重建代码，直接使用传入的all_factors参数
        
        if len(all_factors) <= MAX_FACTORS_TO_SHOW:
            # 全量展示（推荐用于前3-5轮）
            factors_to_show = all_factors
            sampling_note = f"ALL {len(all_factors)} historical factors shown (complete evolution)"
        else:
            # 智能采样（后期轮次）
            import random
            
            top_k = min(self.global_top_k, len(top_factors))
            bottom_k = min(self.global_bottom_k, len(bottom_factors))
            remaining_budget = MAX_FACTORS_TO_SHOW - top_k - bottom_k
            
            # 中间部分（排除top和bottom）- 使用代码哈希来比较
            top_codes = {self._norm_key(f['code']) for f in top_factors[:top_k]}
            bottom_codes = {self._norm_key(f['code']) for f in bottom_factors[-bottom_k:]}
            
            middle_factors = [
                f for f in all_factors 
                if self._norm_key(f['code']) not in top_codes 
                and self._norm_key(f['code']) not in bottom_codes
            ]
            
            # 随机采样中间部分
            if middle_factors and remaining_budget > 0:
                sampled_middle = random.sample(
                    middle_factors, 
                    min(remaining_budget, len(middle_factors))
                )
            else:
                sampled_middle = []
            
            factors_to_show = (
                top_factors[:top_k] +       # 最佳
                sampled_middle +             # 中间随机采样
                bottom_factors[-bottom_k:]   # 最差
            )
            sampling_note = (
                f"Sampled {len(factors_to_show)}/{len(all_factors)} factors "
                f"(top {top_k} + bottom {bottom_k} + random {len(sampled_middle)})"
            )
        
        # 构建历史因子展示（带轮次信息）
        history_block = ""
        if factors_to_show:
            items = []
            for i, f in enumerate(factors_to_show, 1):
                # 提取轮次信息（如果有）
                round_info = f.get('round', '?')
                code = f['code']
                score = f['train_score']
                
                # 格式化输出（每个因子一行，包含轮次、分数和代码）
                items.append(f"  {i}. [Round {round_info}] train_score={score:.4f} | {code}")
            
            history_block = (
                f"HISTORICAL FACTORS ({sampling_note}):\n" +
                "\n".join(items) + "\n\n"
            )
        
        # 字段使用统计（鼓励低频字段）
        field_block = ""
        if field_usage:
            sorted_fields = sorted(field_usage.items(), key=lambda x: x[1])
            underused = [f for f, cnt in sorted_fields if cnt < 2][:10]
            if underused:
                field_block = (
                    "UNDERUSED FIELDS (try to explore these):\n" +
                    ", ".join(underused) + "\n\n"
                )
        
        # 策略指导
        strategy_guide = self._get_strategy_guidance(strategy)
        
        # 示例模板
        example = (
            "TEMPLATE OF ONE VALID ITEM:\n"
            "{\"id\":\"r01_01\",\"code\":\"data['factor_score']=pd.Series("
            "np.where(data['atq']==0,0,(data['ibq']-data['txpq'])/data['atq'])"
            ",index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)\"}\n"
        )
        
        return (
            rules +
            self._whitelist_block() +
            history_block +          # ← 改进后的历史因子展示
            field_block +
            strategy_guide +
            example
        )


    def _get_strategy_guidance(self, strategy: str) -> str:
        """根据策略返回指导文本"""
        if strategy == "diversification":
            return (
                "STRATEGY: DIVERSIFICATION\n"
                "- Explore unusual field combinations\n"
                "- Try underused fields\n"
                "- Be creative and unconventional\n\n"
            )
        elif strategy == "evolution":
            return (
                "STRATEGY: EVOLUTION\n"
                "- Evolve from top performers\n"
                "- Modify successful patterns slightly\n"
                "- Keep core logic but vary fields/operations\n\n"
            )
        elif strategy == "exploitation":
            return (
                "STRATEGY: EXPLOITATION\n"
                "- Focus on variations of best performers\n"
                "- Refine successful factor types\n"
                "- Double down on what works\n\n"
            )
        else:
            return ""

    def _build_refill_prompt(
        self,
        missing: int,
        reasons: List[str],
        existing_codes: List[str] = None
    ) -> str:
        """构建补货prompt"""
        reasons_txt = ""
        if reasons:
            reasons_txt = "Previously rejected reasons: " + "; ".join(set(reasons[:6])) + "\n"
        
        existing_txt = ""
        if existing_codes:
            shown = existing_codes[:10]
            existing_txt = (
                "CRITICAL - ALREADY GENERATED (DO NOT REPEAT):\n" +
                "\n".join(f"  - {code[:120]}" for code in shown) +
                "\n\n**GENERATE COMPLETELY DIFFERENT FACTORS!**\n\n"
            )
        
        return (
            "REFILL REQUEST:\n" +
            reasons_txt +
            existing_txt +
            f"Generate EXACTLY {missing} NEW and UNIQUE factors.\n" +
            "Return ONLY JSON {\"factors\":[...]} with EXACTLY "
            f"{missing} items, same STRICT rules (single statement only).\n" +
            self._whitelist_block() +
            "No semicolons, no second assignment, no code fences, no `json` prefix."
        )

    # ================= 解析与清洗（复用iterative_baseline的逻辑）=================
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
        
        for sep in [";", "\n", "\r"]:
            p = rhs.find(sep)
            if p != -1:
                rhs = rhs[:p].strip()
                break
        
        if rhs.startswith("{"):
            return ""
        
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
        """解析GPT响应"""
        self._last_reject_reasons: List[str] = []
        
        # 1) 严格 JSON
        codes, reject = self._try_parse_json_factors(text)
        self._last_reject_reasons.extend(reject)
        if len(codes) >= target_n:
            self.logger.debug(f"reject_reasons={list(set(self._last_reject_reasons))[:12]}")
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
        
        self.logger.debug(f"reject_reasons={list(set(self._last_reject_reasons))[:12]}")
        return cleaned

    # ================= 日志保存 =================
    def _save_gpt_response(self, round_num: int, prompt: str, resp: str, suffix: str = "") -> None:
        _ensure_dir(self.logs_dir)
        fn = self.logs_dir / f"round_{round_num:02d}_global_agent{suffix}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({
                "round": round_num,
                "global_top_k": self.global_top_k,
                "global_bottom_k": self.global_bottom_k,
                "prompt_preview": prompt[:1200],
                "response": resp[:50000]
            }, f, ensure_ascii=False, indent=2)