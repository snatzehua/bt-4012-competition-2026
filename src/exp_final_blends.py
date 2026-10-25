"""Stability variants of the best blend (LightGBM : GraphSAGE 10 seeds : GAT at 2:1:1,
public LB 0.95627): LightGBM averaged over 5 seeds, and adding the 3-layer GraphSAGE.

Usage: .venv/bin/python src/exp_final_blends.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS  # noqa: E402
from exp_smooth_norm import setup  # noqa: E402
from models import fit_predict_lgb  # noqa: E402

TASKS = ("fold1", "c17", "test")
LGB_SEEDS = (1, 2, 3, 4)  # seed 0 is the existing lgb_{task}.npy


def main():
    Xv, mv, tasks = setup()
    y = mv.label.values
    for name, tr, pr in tasks:
        if name in TASKS and not (PREDS / f"lgb_s1to4_{name}.npy").exists():
            p = np.mean([fit_predict_lgb(Xv[tr], y[tr], Xv[pr], seed=s)[0] for s in LGB_SEEDS], axis=0)
            np.save(PREDS / f"lgb_s1to4_{name}.npy", p)

    r = lambda p: rankdata(p) / len(p)
    P = lambda m, n: np.load(PREDS / f"{m}_{n}.npy")
    base = lambda n: dict(
        lgb=P("lgb", n), lgb5=(P("lgb", n) + 4 * P("lgb_s1to4", n)) / 5,
        sage10=(P("sage", n) + P("sage_s5to9", n)) / 2, gat=P("gat", n), sage3=P("sage3", n))
    blends = {
        "best: lgb:sage10:gat 2:1:1": {"lgb": 2, "sage10": 1, "gat": 1},
        "lgb5:sage10:gat 2:1:1": {"lgb5": 2, "sage10": 1, "gat": 1},
        "lgb5:sage10:gat:sage3 3:1:1:1": {"lgb5": 3, "sage10": 1, "gat": 1, "sage3": 1},
    }
    rows, test_preds = [], {}
    for label, w in blends.items():
        row = {"variant": label}
        for name, _, pr in tasks:
            if name not in TASKS:
                continue
            B = base(name)
            p = sum(k * r(B[m]) for m, k in w.items()) / sum(w.values())
            if name == "test":
                test_preds[label] = p
            else:
                row[name] = metrics(y[pr], p)["auc"]
        rows.append(row)
    ref = rankdata(test_preds["best: lgb:sage10:gat 2:1:1"])
    for row in rows:
        row["test_corr_vs_best"] = np.corrcoef(rankdata(test_preds[row["variant"]]), ref)[0, 1]
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    for label, p in list(test_preds.items())[1:]:
        fname = label.replace(":", "_").replace(" ", "_")
        pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
            OUT / f"submission_final_{fname}_w15.csv", index=False)


if __name__ == "__main__":
    main()
