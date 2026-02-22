# iterative_baseline/positive_agents.py
# Strict JSON prompting + whitelist enforcement + robust parsing/refill
import json
import re
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 路径：把仓库根目录加入 sys.path，便于导入 gpt_runner 与 column_desc
_CUR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# GPT 调用
from common.gpt_runner import call_gpt  # 项目根目录下

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


class PositiveAgent:
    """
    正向因子生成代理（只记忆上一轮）
    输出必须是“单语句”赋值：
      data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)
    """

    # ================= 初始化 =================
    def __init__(self):
        self.logger = logging.getLogger('positive_agents')
        self.temperature = float(_cfg_get('TEMPERATURE', 0.7))  # 稍降随机性
        self.max_tokens = int(_cfg_get('MAX_TOKENS', 900))
        self.factors_per_round = int(_cfg_get('FACTORS_PER_ROUND', _cfg_get('N_FACTORS', 10)))
        self.logs_dir = Path(_cfg_get('GPT_RESPONSE_DIR', _CUR / "logs" / "gpt_responses"))
        _ensure_dir(self.logs_dir)

        # 列名白名单（严格校验用）
        self.field_whitelist: List[str] = [str(c) for c in FIELD_WHITELIST]
        self.field_set_lower = {c.lower(): c for c in self.field_whitelist}
        self.logger.info(f"字段白名单载入：{len(self.field_whitelist)} 列")

        # === 可选：自动增强白名单（仅当配置开启时） ===
        try:
            from config import CONFIG as _C
            auto_aug = bool(_C.get("AUTO_AUGMENT_WHITELIST", False))
            data_file = _C.get("DATA_FILE")
            if auto_aug and data_file and len(self.field_whitelist) < 40:
                import pandas as pd
                cols = pd.read_csv(data_file, nrows=1).columns.tolist()
                # 仅纳入“季度变量风格”的列名，避免把无关字段放进来
                qcols = [c for c in cols if isinstance(c, str) and c.endswith("q")]
                merged = sorted(set(self.field_whitelist) | set(qcols))
                self.field_whitelist = merged
                self.field_set_lower = {c.lower(): c for c in self.field_whitelist}
                self.logger.info(f"[whitelist-augment] enabled -> {len(self.field_whitelist)} columns")
        except Exception as _e:
            self.logger.warning(f"[whitelist-augment] skip ({type(_e).__name__})")

        # 解析与安全限制
        self.max_code_len = int(_cfg_get('MAX_CODE_LEN', 1500))
        self.forbidden_tokens = ['import', 'def ', 'class ', 'for ', 'while ', 'eval(', 'exec(']
        self.lookahead_tokens = ['shift(-', 'lead(', 'future']
        self.blacklist_words = ['price', 'volume', 'high', 'low']

        self.strict_memory = bool(_cfg_get('STRICT_MEMORY', True))
        self.top_k_prev = int(_cfg_get('TOP_K_FACTORS', 5))
        self.refill_max_attempts = int(_cfg_get('POS_AGENT_REFILL_ATTEMPTS', 3))  # 建议 3
        self.batch_size = int(_cfg_get('POS_AGENT_BATCH', 5))  # 分批首呼，降低截断概率

    # ================= 对外主函数 =================
    def generate_optimized_factors(
        self,
        previous_factors: List[Dict],
        round_num: int,
        save_response: bool = True,
        n_override: Optional[int] = None,
        existing_codes: Optional[List[str]] = None
    ) -> List[str]:
        target_n = int(n_override) if (n_override and n_override > 0) else self.factors_per_round

        # —— 仅收集 code + train_score（严格忽略 val）——
        prev_pairs = []
        for f in previous_factors or []:
            code = str(f.get('code', '')).strip()
            if not code:
                continue
            if 'train_score' in f and f.get('train_score') is not None:
                try:
                    ts = float(f.get('train_score'))
                except Exception:
                    continue
                prev_pairs.append({'code': code, 'train_score': ts})

        if not prev_pairs:
            self.logger.warning("上一轮有效记忆为空（需要包含 code+train_score）。将以零上下文请求 GPT。")

        # ========= 分批首呼：显著降低长 JSON 被截断的概率 ========= #
        codes: List[str] = []
        need = target_n
        batch = max(1, min(self.batch_size, need))
        batch_idx = 0
        while need > 0:
            batch_idx += 1
            prompt = self._build_prompt(prev_pairs, batch)
            self.logger.info(f"Round {round_num}: 调用 GPT 生成 {batch} 个新因子")
            resp = call_gpt(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
            if not isinstance(resp, str):
                resp = str(resp or '')
            if save_response:
                self._save_gpt_response(round_num, prompt, resp, suffix=f"_batch_{batch_idx}_{batch}")
            self.logger.info(f"Round {round_num}: GPT首呼返回前200字：{resp[:200].replace(os.linesep,' ')}")

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

        # 去重：排除已尝试/已评估
        if existing_codes:
            exist_keys = set(self._norm_key(c) for c in existing_codes if isinstance(c, str))
            codes = [c for c in codes if self._norm_key(c) not in exist_keys]

        # 补货（仍然保留，以防极端情况下首呼/分批仍有损耗）
        attempts = 0
        consecutive_no_progress = 0  # ← 新增
        last_count = 0  # ← 新增
        while len(codes) < target_n and attempts < self.refill_max_attempts:
            attempts += 1
            last_count = len(codes)  # ← 记录补货前数量
            missing = target_n - len(codes)
            self.logger.info(f"Round {round_num}: 解析不足 {len(codes)}/{target_n}，补货 {missing}（attempt {attempts}/{self.refill_max_attempts}）")

            reasons = self._last_reject_reasons if hasattr(self, "_last_reject_reasons") else []
            refill_prompt = self._build_refill_prompt(
                missing, 
                reasons,
                existing_codes=codes  # ← 传入当前已生成的因子
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

            # ← 移到这里：在添加完 more 之后检查是否有进展
            if len(codes) == last_count:  # ← 现在位置正确了
                consecutive_no_progress += 1
                if consecutive_no_progress >= 2:  # 连续 2 次无进展
                    self.logger.warning(f"连续 {consecutive_no_progress} 次补货无效（生成的都是重复因子），停止补货")
                    break
            else:
                consecutive_no_progress = 0

        # 兜底：仍为 0 时再要一次“强格式化”输出
        if len(codes) == 0:
            self.logger.info(f"Round {round_num}: 触发强格式兜底请求")
            strict_prompt = self._build_strict_prompt(target_n, prev_pairs)
            resp3 = call_gpt(strict_prompt, temperature=0.28, max_tokens=self.max_tokens)
            if not isinstance(resp3, str):
                resp3 = str(resp3 or '')
            if save_response:
                self._save_gpt_response(round_num, strict_prompt, resp3, suffix="_strict")
            self.logger.info(f"Round {round_num}: 兜底返回前200字：{resp3[:200].replace(os.linesep,' ')}")
            codes = self._parse_and_transform(resp3, target_n)

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

    def _examples_json(self, prev_pairs: List[Dict]) -> str:
        items = []
        for i, p in enumerate(prev_pairs[: self.top_k_prev], 1):
            items.append({"id": f"ex_{i:02d}", "train_score": float(p["train_score"]), "code": p["code"]})
        return json.dumps({"previous": items}, ensure_ascii=False)

    def _build_prompt(self, prev_pairs: List[Dict], target_n: int) -> str:
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
            "- Keep code concise (≤ 240 chars).\n"
        )
        # 示例尽量用白名单里常见字段
        example_item = (
            "{\"id\":\"r01_01\",\"code\":\"data['factor_score']=pd.Series("
            "np.where(data['atq']==0,0,(data['ibq']-data['txpq'])/data['atq'])"
            ",index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)\"}"
        )
        prev_json = self._examples_json(prev_pairs)
        return (
            rules + self._whitelist_block() +
            "TRAINING FEEDBACK (code + train_score only):\n" +
            prev_json + "\n\n" +
            "TEMPLATE OF ONE VALID ITEM:\n" + example_item + "\n"
        )

    def _build_refill_prompt(self, missing: int, reasons: List[str], existing_codes: List[str] = None) -> str:
        reasons_txt = ""
        if reasons:
            reasons_txt = "Previously rejected reasons: " + "; ".join(set(reasons[:6])) + "\n"
        
        existing_txt = ""
        if existing_codes:
            shown = existing_codes[:10]
            existing_txt = (
                "ALREADY GENERATED (DO NOT REPEAT):\n" +
                "\n".join(f"  - {code[:120]}" for code in shown) +
                "\n\n**GENERATE COMPLETELY DIFFERENT FACTORS!**\n\n"
            )
        
        # ← 新增：明确的格式模板
        example = (
            "data['factor_score']=pd.Series(np.where(data['atq']==0,0,"
            "data['ibq']/data['atq']),index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)"
        )
        
        return (
            "REFILL REQUEST:\n" +
            reasons_txt +
            existing_txt +
            # ← 关键：重申格式要求
            "CRITICAL FORMAT RULES (violations will be discarded):\n"
            "1. Each code MUST start with: data['factor_score'] = pd.Series(...)\n"
            "2. NO bare column names like: (revtq - cogsq) / saleq  ← WRONG\n"
            "3. ALL columns MUST use: data['colname']  ← CORRECT\n"
            "4. ONE single statement only, no semicolons\n\n"
            f"VALID EXAMPLE:\n{example}\n\n" +
            f"Generate EXACTLY {missing} NEW factors.\n"
            f"Return ONLY JSON {{\"factors\":[...]}} with EXACTLY {missing} items.\n" +
            self._whitelist_block()
        )



    def _build_strict_prompt(self, target_n: int, prev_pairs: List[Dict]) -> str:
        return (
            "STRICT FORMAT REQUEST:\n"
            "Return ONLY a JSON object with key `factors` containing EXACTLY "
            f"{target_n} items. Each item must include keys \"id\" and \"code\".\n"
            "The `code` must be a SINGLE assignment line in the exact pattern:\n"
            "data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)\n"
            "Use ONLY allowed columns. No look-ahead, no loops/imports, no prose, no code fences, no `json` prefix.\n\n" +
            self._whitelist_block()
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
        """从任意位置切出以 data['factor_score']= 起头的片段；尽量截到该语句块结束。"""
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

    # —— 单语句强制重写器 —— #
    def _force_single_assignment(self, code: str) -> str:
        """
        把任意“多语句/二次赋值”重写为 单语句：
        取第一次 data['factor_score']= 的 RHS 表达式 <EXPR>，生成：
        data['factor_score'] = pd.Series(<EXPR>, index=data.index).replace([np.inf,-np.inf], np.nan).fillna(0)
        """
        if not isinstance(code, str):
            return ""
        s = code.strip()

        # 1) 把围栏/前缀去掉
        s = self._preclean_json_text(s)

        # 2) 找第一次赋值
        m = re.search(r"data\[['\"]factor_score['\"]\]\s*=\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        rhs = m.group(1).strip()

        # 3) 遇到分号/换行就截断（仅保留第一次赋值的 RHS）
        for sep in [";", "\n", "\r"]:
            p = rhs.find(sep)
            if p != -1:
                rhs = rhs[:p].strip()
                break

        # 4) 如果 RHS 是 JSON 对象，直接判空
        if rhs.startswith("{"):
            return ""

        # 5) 构建单语句
        return (
            "data['factor_score']=pd.Series("
            f"{rhs}"
            ",index=data.index).replace([np.inf,-np.inf],np.nan).fillna(0)"
        )

    def _all_columns_allowed(self, code: str, rejected: List[str]) -> bool:
        if not self.field_set_lower:  # 没有白名单时不做拦截
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
        """
        要求所有列访问都写成 data['col']；出现裸列名（如 revtq）则拒绝。
        避免 evaluator 出现 name 'revtq' is not defined 之类错误。
        """
        if not self.field_whitelist:
            return True
        s = code
        for col in self.field_whitelist:
            # 前面不是 data[' 或 data[" 的“裸”单词命中
            pattern = r"(?<!data\[['\'])\b" + re.escape(col) + r"\b"
            if re.search(pattern, s):
                rejected.append(f"bare_column:{col}")
                return False
        return True

    def _sanitize_one(self, code: str, rejected: List[str]) -> Optional[str]:
        if not code or len(code) > self.max_code_len:
            rejected.append("too_long_or_empty")
            return None

        # —— 强制单语句重写 —— #
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
        """剥掉 ```json / ``` 包裹与 'json\\n' 前缀，并截取最外层 {...}"""
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
        """
        尝试把 {"factors":[{"id":"...","code":"..."}]} 解析为代码列表
        返回：codes, reject_reasons
        """
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
                if isinstance(it, str):
                    code = it.strip()
                elif isinstance(it, dict):
                    code = str(it.get("code", "")).strip()
                else:
                    reject.append("item_not_object")
                    continue
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
        """
        解析优先级：
        1) 尝试严格 JSON 解析 factors
        2) 失败再回退到基于前缀的切片解析（同样强制单语句）
        """
        self._last_reject_reasons: List[str] = []

        # 1) 严格 JSON
        codes, reject = self._try_parse_json_factors(text)
        self._last_reject_reasons.extend(reject)
        if len(codes) >= target_n:
            self.logger.debug(f"reject_reasons={list(set(self._last_reject_reasons))[:12]}")
            return codes[:target_n]

        # 2) 回退：按 “data['factor_score'] =” 从任意位置切片
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
                "prompt_preview": prompt[:1200],
                "response": resp[:50000]
            }, f, ensure_ascii=False, indent=2)
