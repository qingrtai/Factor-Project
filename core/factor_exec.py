# core/factor_exec.py  —— 安全执行 + 稳定提取版
from __future__ import annotations
import re
from typing import Dict, Any
import numpy as np
import pandas as pd
from core.utils import FactorExecutionError

# 1) 禁止的关键词/模块/明显 I/O 与网络调用（尽量在源头拦）
_BANNED_PATTERNS = (
    r"\bimport\b",
    r"\bfrom\s+\w+\s+import\b",
    r"\b__import__\b",
    r"\beval\s*\(", 
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\bos\.", r"\bsys\.", r"\bsubprocess\.", r"\bsocket\.", r"\bpathlib\.",
    r"\brequests?\.", r"\burllib\.", r"\bhttpx?\.", r"\bboto3\b", r"\bsqlalchemy\b",
    r"\bsklearn\b", r"\btorch\b", r"\btensorflow\b", r"\bdask\b", r"\bjoblib\b", r"\bpickle\b",
    # 典型文件 I/O 接口（即便 import 被挡，pandas 内部也可能访问内建 open，这里先从代码文本层面阻断）
    r"\bread_csv\s*\(", r"\bread_parquet\s*\(", r"\bread_excel\s*\(",
    r"\bto_csv\s*\(", r"\bto_parquet\s*\(", r"\bto_excel\s*\(", r"\bto_pickle\s*\(", r"\bto_json\s*\(",
)

# 2) 清理 markdown 代码围栏
_CODE_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# 3) 允许的最大代码行数（给 GPT 留一点冗余，但仍限制体量）
_MAX_LINES = 100


def _sanitize_code(code_str: str) -> str:
    code = _CODE_FENCE_RE.sub("", str(code_str)).strip()

    # 常见补丁：显式 fill_method=None，避免版本差异导致隐式前向填充
    code = code.replace(".pct_change()", ".pct_change(fill_method=None)")

    # 拦截高危标记
    for pat in _BANNED_PATTERNS:
        if re.search(pat, code):
            raise FactorExecutionError("forbidden token/module or IO/network call found in code")

    # 长度限制（避免超长脚本）
    if len(code.splitlines()) > _MAX_LINES:
        raise FactorExecutionError(f"code too long (> {_MAX_LINES} lines); keep it concise")

    return code


def safe_execute(code_str: str, data: pd.DataFrame) -> pd.Series:
    """
    在受限环境中执行因子代码。要求被执行代码最终写入：
        data['factor_score'] = <Series-like>
    仅暴露 'data'（副本）、'np'、'pd'；屏蔽 __builtins__ 以降低风险。
    返回：与输入 data.index 对齐的 float Series（NaN 允许，Inf 会被置 NaN）。
    质量门槛：至少存在 2 个非 NaN 的不同取值，否则判定为无效（常数/准常数因子）。
    """
    if not isinstance(data, pd.DataFrame):
        raise FactorExecutionError("input 'data' must be a pandas DataFrame")

    code = _sanitize_code(code_str)

    # 受限执行环境
    local_env: Dict[str, Any] = {"data": data.copy(), "np": np, "pd": pd}
    try:
        # 注意：globals 设置空 builtins 只是降低风险，不能完全隔离第三方内部 I/O。
        # 上面对源码的正则拦截是第一道强约束。
        exec(code, {"__builtins__": {}}, local_env)
    except Exception as e:
        raise FactorExecutionError(f"execution error: {e}")

    out_df = local_env.get("data", None)
    if not isinstance(out_df, pd.DataFrame):
        raise FactorExecutionError("code must operate on the provided 'data' DataFrame")

    if "factor_score" not in out_df.columns:
        raise FactorExecutionError("missing output column: data['factor_score']")

    s = out_df["factor_score"]

    # 数值化 + 清理异常值
    s = pd.to_numeric(s, errors="coerce")
    s[~np.isfinite(s)] = np.nan

    # 与输入索引对齐（防止用户在代码里 drop/reindex）
    if not s.index.equals(data.index):
        s = s.reindex(data.index)

    # 质量检查：不能是常数或近似常数（至少需要 2 个不同的非 NaN 值）
    non_null = s.dropna()
    if non_null.nunique() <= 1:
        raise FactorExecutionError("factor is constant or invalid (<=1 unique non-NaN)")

    # 不做任何数值“正则化/截断”——这些由 evaluator/score 模块负责
    return s
