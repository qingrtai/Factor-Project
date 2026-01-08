# core/utils.py
from __future__ import annotations
import logging
import sys
import os
import random
import warnings
from typing import Optional
import numpy as np


__all__ = [
    "FactorExecutionError",
    "DataValidationError",
    "TimeoutError",           # 保留原名以兼容旧代码
    "FactorTimeoutError",     # 推荐使用的新别名，避免与内置 TimeoutError 混淆
    "setup_logger",
    "set_global_seed",
]


class FactorExecutionError(Exception):
    """Raised when executing factor code fails (syntax/runtime/invalid output)."""
    pass


class DataValidationError(Exception):
    """Raised when input data schema/splits are invalid or unavailable."""
    pass


class FactorTimeoutError(Exception):
    """Raised when a factor operation exceeds the allowed time budget."""
    pass


# 兼容：保留旧名，避免与 Python 内置 TimeoutError 产生语义混淆
TimeoutError = FactorTimeoutError


def setup_logger(
    name: str,
    level: Optional[int | str] = None,
    to_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create or fetch a configured logger.
    - level: 可传 logging.INFO / "INFO"；若不传则读取环境变量 LOG_LEVEL，默认 INFO
    - to_file: 若提供文件路径，则同时写入该文件（和 stdout）
    - 多次调用不会重复添加 handler；但会根据新的 level 动态调整日志级别
    """
    lg = logging.getLogger(name)

    # 解析日志级别
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # 如果已存在 handler，仍支持调整级别后直接返回
    if lg.handlers:
        lg.setLevel(level)
        return lg

    lg.setLevel(level)
    lg.propagate = False  # 防止父 logger 重复输出

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    lg.addHandler(ch)

    # Optional file handler
    if to_file:
        try:
            fh = logging.FileHandler(to_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            lg.addHandler(fh)
        except Exception as e:
            # 不影响主流程
            warnings.warn(f"setup_logger: failed to create file handler: {e}", RuntimeWarning)

    return lg


def set_global_seed(
    seed: int = 42,
    *,
    set_hash: bool = True,
    try_torch: bool = True,
) -> np.random.Generator:
    """
    Set global RNG seeds for reproducibility.
    - seed: 基础随机种子
    - set_hash: 是否设置 PYTHONHASHSEED（建议 True，提升可重复性）
    - try_torch: 若系统已安装 torch，则同步设置 torch 的随机种子（不强制依赖）

    返回：一个 numpy.random.Generator，可在上层需要确定性采样时复用。
    """
    if set_hash:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if try_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # 尽量保证确定性（用户未装 torch 时不会报错）
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            # 不强依赖 torch，忽略所有导入/设定失败
            pass

    # 返回一个生成器，便于需要本地 rng 的模块使用
    return np.random.default_rng(seed)
