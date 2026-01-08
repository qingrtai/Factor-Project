# shared/code_utils.py
"""
代码工具函数（agents 共享）
"""

import re
import difflib
from typing import Optional, Any
import json


# ========== 代码规范化 ========== #

def normalize_code(code: str) -> str:
    """
    规范化代码：去回车、去多余空白、去注释行
    
    用于：
    - 代码相似度比较
    - 去重判断
    """
    c = str(code or "")
    c = c.replace("\r\n", "\n")
    # 去除注释行
    c = "\n".join([ln for ln in c.split("\n") if not ln.strip().startswith("#")])
    c = c.strip()
    # 压缩空白
    c = re.sub(r"\s+", " ", c)
    return c


def code_similarity(a: str, b: str) -> float:
    """
    计算两段代码的文本相似度（0-1）
    
    用于去重判断
    """
    return difflib.SequenceMatcher(
        a=normalize_code(a), 
        b=normalize_code(b)
    ).ratio()


# ========== 代码校验和转换 ========== #

def to_single_line(code: str) -> Optional[str]:
    """
    将代码转为单行赋值：data['factor_score'] = <expr>
    
    过滤：
    - dict-like RHS
    - 纯标点（".", "...", "..."）
    - 括号不平衡
    """
    if not code:
        return None
    
    # 优先抓包含 factor_score 的行
    if "\n" in code:
        for ln in code.split("\n"):
            if "factor_score" in ln:
                code = ln
                break
    
    c = normalize_code(code)
    
    # 补左值
    if "data['factor_score']" not in c and 'data["factor_score"]' not in c:
        c = f"data['factor_score'] = {c}"
    
    # 拆 RHS
    rhs = c.split("=", 1)[1].strip() if "=" in c else c
    if not rhs:
        return None
    
    # 拒绝纯点号/省略号/纯标点
    if rhs in {".", "..", "..."}:
        return None
    if not re.search(r"[A-Za-z0-9_]", rhs):
        return None
    
    # 过滤明显 dict/json-like
    if ("{" in rhs and "}" in rhs and ":") or rhs.startswith(("{", "[")):
        return None
    
    # 括号/方括号数目平衡
    if rhs.count("(") != rhs.count(")") or rhs.count("[") != rhs.count("]"):
        return None
    
    return f"data['factor_score'] = {rhs}"


def is_valid_single_line(code: str) -> bool:
    """
    基础合法性 + 语法编译校验
    
    检查：
    - 是否包含 data['factor_score'] = ...
    - 禁止 import/多语句/反引号
    - 括号平衡
    - 可编译
    """
    if not isinstance(code, str):
        return False
    
    s = code.strip()
    if not s:
        return False
    
    if "data['factor_score']" not in s and 'data["factor_score"]' not in s:
        return False
    
    rhs = s.split("=", 1)[1].strip() if "=" in s else s
    if not rhs:
        return False
    
    # 禁止 import/多语句/反引号
    if "import " in s or ";" in s or "```" in s:
        return False
    
    # 括号平衡
    if s.count("(") != s.count(")"):
        return False
    if s.count("[") != s.count("]"):
        return False
    
    # 拒绝 "." / "..." / 纯标点
    if rhs in {".", "..", "..."}:
        return False
    if not re.search(r"[A-Za-z0-9_]", rhs):
        return False
    
    # 语法层编译（只查语法，不执行）
    try:
        compile(s, "<code_check>", "exec")
    except Exception:
        return False
    
    return True


def validate_and_fix_code(code: str) -> Optional[str]:
    """
    规范成单行赋值：data['factor_score'] = <expr>
    
    并过滤明显不合法的 RHS（占位/字典/前视等）
    
    用于 positive_agents
    """
    c = normalize_code(code)
    if not c:
        return None
    
    # 统一为单行赋值
    if "data['factor_score']" not in c and 'data["factor_score"]' not in c:
        c = f"data['factor_score'] = {c}"
    
    # 基础拦截：禁止 import / 反引号 / 多语句
    if "import " in c or "```" in c or ";" in c:
        return None
    
    # 拆 RHS
    if "=" in c:
        _, rhs = c.split("=", 1)
        rhs = rhs.strip()
    else:
        rhs = c
    
    # 占位与纯标点过滤
    if rhs in {".", "..", "..."}:
        return None
    if not re.search(r"[A-Za-z0-9_]", rhs):
        return None
    
    # 拒绝 dict/json-like RHS
    if ("{" in rhs and "}" in rhs and ":" in rhs) or rhs.startswith(("{", "[")):
        return None
    
    # 括号/方括号匹配
    if rhs.count("(") != rhs.count(")") or rhs.count("[") != rhs.count("]"):
        return None
    
    # 语法层编译检查（不执行）
    try:
        compile(f"data['factor_score'] = {rhs}", "<code_validation>", "exec")
    except Exception:
        return None
    
    return f"data['factor_score'] = {rhs}"


# ========== GPT 响应处理 ========== #

def extract_content(resp: Any) -> str:
    """
    兼容常见返回结构，提取文本
    
    支持：
    - 字符串
    - OpenAI 格式 dict
    - 其他格式（转 JSON）
    """
    if isinstance(resp, str):
        return resp
    
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass
        return json.dumps(resp, ensure_ascii=False)
    
    return str(resp)