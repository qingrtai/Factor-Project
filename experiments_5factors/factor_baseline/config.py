from shared.config_loader import load_experiment_config
class BaselineConfig:
    def __init__(self):
        cfg = load_experiment_config("factor_baseline")
        self.n_factors = int(cfg["n_factors"])
        self.min_coverage = float(cfg.get("min_coverage", 0.3))
