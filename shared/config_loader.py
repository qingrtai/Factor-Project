# shared/config_loader.py
import yaml
from pathlib import Path

def load_global_config() -> dict:
    """加载 configs/global.yaml"""
    config_path = Path("configs/global.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Global config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_experiment_config(experiment_name: str) -> dict:
    """从 configs/experiments.yaml 中加载指定实验的配置"""
    config_path = Path("configs/experiments.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Experiments config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        all_experiments = yaml.safe_load(f)
    
    if experiment_name not in all_experiments:
        raise KeyError(f"Experiment '{experiment_name}' not found in {config_path}")
    
    return all_experiments[experiment_name]