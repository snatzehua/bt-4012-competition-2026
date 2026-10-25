"""Weighted rank blends of LightGBM, CatBoost and GraphSAGE (last 15 steps), plus
the LightGBM + GraphSAGE probability average.

Reuses the predictions saved by exp_architectures.py, so nothing is retrained.
The plain lgb+sage blend scored 0.95584 on the public LB while validation ranked it
below lgb alone, so validation numbers here are for reference, not for choosing.

Usage: .venv/bin/python src/exp_weighted_blend.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS, setup  # noqa: E402

# (lgb, catboost, sage) weights
WEIGHTS = {
    "lgb_sage_1_1": (1, 0, 1),  # LB 0.95584, the reference
    "lgb_cat_sage_1_1_1": (1, 1, 1),
    "lgb_cat_sage_1_1_2": (1, 1, 2),
    "lgb_cat_sage_2_1_2": (2, 1, 2),
    "lgb_cat_sage_1_0.5_1.5": (1, 0.5, 1.5),
    "lgb_sage_1_2": (1, 0, 2),
    "lgb_sage_1_3": (1, 0, 3),
}


def weighted_rank(ps, ws):
    return sum(w * rankdata(p) / len(p) for p, w in zip(ps, ws)) / sum(ws)


def main():
    _, mv, tasks = setup()
    y = mv.label.values
    models = ("lgb", "catboost", "sage")
    P = {(m, name): np.load(PREDS / f"{m}_{name}.npy") for m in models for name, _, _ in tasks}
    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    ref = weighted_rank([P[(m, "test")] for m in models], WEIGHTS["lgb_sage_1_1"])

    results = {}
    for label, ws in WEIGHTS.items():
        folds = [metrics(y[pr], weighted_rank([P[(m, name)] for m in models], ws))
                 for name, _, pr in tasks if name.startswith("fold")]
        r = pd.DataFrame(folds).mean().to_dict()
        p = weighted_rank([P[(m, "test")] for m in models], ws)
        r["test_spearman_vs_lgb_sage"] = spearmanr(p, ref).correlation
        results[label] = r
        print(f"{label:24s} val AUC {r['auc']:.4f}  AP {r['ap']:.4f}  "
              f"test rank corr vs lgb+sage {r['test_spearman_vs_lgb_sage']:.4f}")
        pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
            OUT / f"submission_wblend_{label}_w15.csv", index=False)
    (OUT / "results_weighted_blend.json").write_text(json.dumps(results, indent=2))

    # LightGBM + GraphSAGE 1:1 with probabilities averaged instead of ranks (LB 0.95507)
    p = (P[("lgb", "test")] + P[("sage", "test")]) / 2
    pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
        OUT / "submission_pblend_lgb_sage_w15.csv", index=False)


if __name__ == "__main__":
    main()
