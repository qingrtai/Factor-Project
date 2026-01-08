# shared/paths.py
from pathlib import Path

def results_dir(experiment_name: str) -> Path:
    """
    返回实验结果目录并自动创建
    
    Args:
        experiment_name: 实验名称（如 "factor_baseline"）
    
    Returns:
        Path 对象指向 results/{experiment_name}/
    """
    path = Path("results") / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path