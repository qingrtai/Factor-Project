# experiments/factor_baseline/run_baseline.py
from shared.config_loader import load_global_config
from shared.paths import results_dir
from core.data_loader import load_splits
from core.factor_evaluator import batch_evaluate
from experiments.factor_baseline.config import BaselineConfig
from common.generate_factors import generate_factors
from core.utils import set_global_seed  # 新增

def run():
    set_global_seed(42, try_torch=True)  # 新增：可复现
    g = load_global_config()
    splits = load_splits(
        g["data"]["raw_file"], g["schema"]["date_col"], g["years"],
        id_col=g["schema"]["id_col"], ret_col=g["schema"]["ret_col"]
    )
    cfg = BaselineConfig()
    facts = generate_factors(n=cfg.n_factors)

    ppy = int(g.get("freq_per_year", 4))
    res = batch_evaluate(
        facts, splits,
        ret_col=g["schema"]["ret_col"],
        date_col=g["schema"]["date_col"],
        periods_per_year=ppy
    )
    outdir = results_dir("factor_baseline")
    # 先保存“未过滤”版本，便于排查
    raw_path = outdir / "baseline_factor_metrics_raw.csv"
    res.to_csv(raw_path, index=False)

    # 覆盖度过滤（仍按你的配置阈值）
    kept = res[res["val_coverage"] >= cfg.min_coverage].copy()
    out = outdir / "baseline_factor_metrics.csv"
    kept.to_csv(out, index=False)

    print(f"saved raw -> {raw_path}")
    print(f"saved -> {out} (kept {len(kept)}/{len(res)} by coverage≥{cfg.min_coverage:.2f})")
    if kept.empty:
        print("WARNING: all factors were filtered out by coverage. "
              "Consider lowering min_coverage, or inspect *_raw.csv for diagnostics.")

if __name__ == "__main__":
    run()
