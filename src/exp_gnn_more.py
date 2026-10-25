"""More GraphSAGE variants for blending with LightGBM (best so far: LightGBM :
GraphSAGE 10 seeds : GAT at 2:1:1, public LB 0.95627). Each changes one thing:

- sage_all, sage_w20, sage_w10: training window (trees preferred 15 steps; a neural
  model may want more data, and a different window gives a different view)
- sage_local: only the 93 local features; the network does all the neighbour
  aggregation itself instead of also reading the dataset's precomputed summaries
- sage_teststd: features of the prediction period standardised with that period's
  own mean and spread (simple test-time adaptation to drift)
- sage_dann: domain-adversarial GraphSAGE (nn_models.fit_predict_dann)

Validation is a bug check only; it can't rank graph models (exp_resample_val.py).

Usage:
    .venv/bin/python src/exp_gnn_more.py nn [model ...]
    .venv/bin/python src/exp_gnn_more.py score
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS  # noqa: E402
from exp_architectures import setup as arch_setup  # noqa: E402
from features import load  # noqa: E402

SEEDS = range(5)
# (cutoff, eval steps) per task; "test" predicts the test nodes
TASKS = {"fold1": (29, range(30, 36)), "c17": (17, range(18, 36)), "test": (35, None)}
VARIANTS = {  # name: (window, feature set, standardisation, trainer)
    "sage_all": (None, "all", "train", "sage"),
    "sage_w20": (20, "all", "train", "sage"),
    "sage_w10": (10, "all", "train", "sage"),
    "sage_local": (15, "local", "train", "sage"),
    "sage_teststd": (15, "all", "split", "sage"),
    "sage_dann": (15, "all", "train", "dann"),
}


def masks(mv, task, window):
    c, steps = TASKS[task]
    lab, ts = mv.label.notna().values, mv.time_step.values
    tr = lab & (ts <= c) & (ts > c - (window or 99))
    pr = mv.is_test.values if steps is None else lab & np.isin(ts, steps)
    return tr, pr


def run_nn(only=()):
    from nn_models import fit_predict_dann, fit_predict_nn, neighbour_operators, standardise

    Xv, mv, _ = arch_setup()
    _, _, edges = load()
    ops = neighbour_operators(mv.txId.values, edges)
    y = mv.label.values
    local = [c for c in Xv.columns if int(c.split("_")[1]) <= 93]
    for m, (window, feats, std, trainer) in VARIANTS.items():
        if only and m not in only:
            continue
        X = (Xv[local] if feats == "local" else Xv).values
        for task in TASKS:
            t0 = time.time()
            tr, pr = masks(mv, task, window)
            ti, pi = np.flatnonzero(tr), np.flatnonzero(pr)
            Xs = standardise(X, ti)
            if std == "split":
                Xs[pi] = standardise(X, pi)[pi]
            if trainer == "dann":
                p = np.mean([fit_predict_dann(Xs, y, ti, pi, ops, seed=s) for s in SEEDS], axis=0)
            else:
                p = np.mean([fit_predict_nn("sage", Xs, y, ti, pi, ops=ops, seed=s) for s in SEEDS], axis=0)
            np.save(PREDS / f"{m}_{task}.npy", p)
            print(f"{m:13s} {task:6s} {time.time() - t0:.0f}s", flush=True)


def run_score():
    _, mv, _ = arch_setup()
    y = mv.label.values
    r = lambda p: rankdata(p) / len(p)
    P = lambda m, n: np.load(PREDS / f"{m}_{n}.npy")
    lgb5 = lambda n: (P("lgb", n) + 4 * P("lgb_s1to4", n)) / 5
    sage10 = lambda n: (P("sage", n) + P("sage_s5to9", n)) / 2

    def best(n, lgb=lambda n: P("lgb", n)):
        return (2 * r(lgb(n)) + r(sage10(n)) + r(P("gat", n))) / 4

    variants = {"best: lgb:sage10:gat 2:1:1": best, "lgb5:sage10:gat 2:1:1": lambda n: best(n, lgb5)}
    for m in VARIANTS:
        variants[f"{m} alone"] = lambda n, m=m: P(m, n)
        variants[f"lgb5+{m} 1:1"] = lambda n, m=m: (r(lgb5(n)) + r(P(m, n))) / 2
        variants[f"lgb5:sage10:gat:{m} 3:1:1:1"] = lambda n, m=m: (
            3 * r(lgb5(n)) + r(sage10(n)) + r(P("gat", n)) + r(P(m, n))) / 6

    rows, test_preds = [], {}
    for label, f in variants.items():
        row = {"variant": label}
        for task in TASKS:
            p = f(task)
            if task == "test":
                test_preds[label] = p
            else:
                row[task] = metrics(y[masks(mv, task, None)[1]], p)["auc"]
        rows.append(row)
    ref = rankdata(test_preds["best: lgb:sage10:gat 2:1:1"])
    sage_ref = rankdata(sage10("test"))
    for row in rows:
        t = rankdata(test_preds[row["variant"]])
        row["test_corr_vs_best"] = np.corrcoef(t, ref)[0, 1]
        row["test_corr_vs_sage10"] = np.corrcoef(t, sage_ref)[0, 1]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gnn_more_results.csv", index=False)
    print(res.round(4).to_string(index=False))

    te = mv.is_test.values
    idx = mv["index"][te].astype(int).values
    for label, p in test_preds.items():
        if "alone" in label or label.startswith("best"):
            continue
        fname = label.replace(":", "_").replace("+", "_").replace(" ", "_")
        pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
            OUT / f"submission_more_{fname}_w15.csv", index=False)


if __name__ == "__main__":
    {"nn": lambda: run_nn(sys.argv[2:]), "score": run_score}[sys.argv[1]]()
