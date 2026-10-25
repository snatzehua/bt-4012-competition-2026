"""Two drift ideas on top of the LightGBM + GraphSAGE blend (public LB 0.95584).

1. Prediction smoothing over the graph: fraud clusters along edges (an illicit node's
   neighbour is illicit 57% of the time, a licit node's 2.8%), so each node's score
   is pulled toward its neighbours' scores. Pure post-processing, label-propagation
   style: s <- (1 - alpha) * s0 + alpha * mean_neighbours(s), repeated `iters` times.
2. Per-time-step feature normalisation: every feature becomes its percentile rank
   among nodes in the same time step, so a value is judged against its contemporaries.

All models use the last 15 steps. Tasks: the two original folds, the "train <= 17"
scheme from exp_resample_val.py, and test.

Usage:
    .venv/bin/python src/exp_smooth_norm.py trees
    .venv/bin/python src/exp_smooth_norm.py nn
    .venv/bin/python src/exp_smooth_norm.py score
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS, SEEDS, WINDOW  # noqa: E402
from exp_architectures import setup as arch_setup  # noqa: E402
from features import load  # noqa: E402

ALPHAS = (0.2, 0.3, 0.5, 0.7)
ITERS = (1, 3, 10)


def setup():
    Xv, mv, tasks = arch_setup()
    lab, ts = mv.label.notna().values, mv.time_step.values
    tasks.insert(2, ("c17", lab & (ts <= 17) & (ts > 17 - WINDOW), lab & (ts > 17)))
    return Xv, mv, tasks


def step_normalise(Xv, mv):
    return Xv.groupby(mv.time_step.values).rank(pct=True).astype(np.float32)


def run_trees():
    from models import fit_predict_lgb

    Xv, mv, tasks = setup()
    Xn = step_normalise(Xv, mv)
    Xrn = pd.concat([Xv, Xn.add_suffix("_steprank")], axis=1)
    y = mv.label.values
    for m, X in (("lgb_norm", Xn), ("lgb_rawnorm", Xrn), ("lgb", Xv)):
        for name, tr, pr in tasks:
            if m == "lgb" and name != "c17":
                continue  # already saved by exp_architectures.py
            t0 = time.time()
            np.save(PREDS / f"{m}_{name}.npy", fit_predict_lgb(X[tr], y[tr], X[pr])[0])
            print(f"{m:12s} {name:6s} {time.time() - t0:.0f}s", flush=True)


def run_nn():
    from nn_models import fit_predict_nn, neighbour_operators, standardise

    Xv, mv, tasks = setup()
    _, _, edges = load()
    ops = neighbour_operators(mv.txId.values, edges)
    Xn = step_normalise(Xv, mv).values
    y = mv.label.values
    for m, X in (("sage_norm", Xn), ("sage", Xv.values)):
        for name, tr, pr in tasks:
            if m == "sage" and name != "c17":
                continue
            t0 = time.time()
            ti, pi = np.flatnonzero(tr), np.flatnonzero(pr)
            Xs = standardise(X, ti)
            p = np.mean([fit_predict_nn("sage", Xs, y, ti, pi, ops=ops, seed=s) for s in SEEDS], axis=0)
            np.save(PREDS / f"{m}_{name}.npy", p)
            print(f"{m:12s} {name:6s} {time.time() - t0:.0f}s", flush=True)


def neighbour_mean(txids, edges):
    """Row-normalised undirected adjacency over the predicted nodes only."""
    pos = pd.Series(np.arange(len(txids)), index=txids)
    e = edges[edges.iloc[:, 0].isin(pos.index) & edges.iloc[:, 1].isin(pos.index)]
    a, b = pos[e.iloc[:, 0]].values, pos[e.iloc[:, 1]].values
    A = sp.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(txids),) * 2).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1), 0)) @ A, deg > 0


def smooth(s0, M, has_nbr, alpha, iters):
    s = s0.copy()
    for _ in range(iters):
        s = np.where(has_nbr, (1 - alpha) * s0 + alpha * (M @ s), s0)
    return s


def run_score():
    from scipy.stats import rankdata

    _, mv, tasks = setup()
    _, _, edges = load()
    y = mv.label.values
    r = lambda p: rankdata(p) / len(p)
    P = lambda m, name: np.load(PREDS / f"{m}_{name}.npy")
    base = {
        "lgb": lambda n: P("lgb", n),
        "lgb+sage (LB 0.95584)": lambda n: (r(P("lgb", n)) + r(P("sage", n))) / 2,
        "lgb_norm": lambda n: P("lgb_norm", n),
        "lgb_rawnorm": lambda n: P("lgb_rawnorm", n),
        "sage_norm": lambda n: P("sage_norm", n),
        "lgb_norm+sage_norm": lambda n: (r(P("lgb_norm", n)) + r(P("sage_norm", n))) / 2,
        "lgb_norm+sage": lambda n: (r(P("lgb_norm", n)) + r(P("sage", n))) / 2,
        "lgb+sage_norm": lambda n: (r(P("lgb", n)) + r(P("sage_norm", n))) / 2,
        "lgb_rawnorm+sage": lambda n: (r(P("lgb_rawnorm", n)) + r(P("sage", n))) / 2,
    }
    graph = {name: neighbour_mean(mv.txId.values[pr], edges) for name, _, pr in tasks}

    variants = {}
    for label, f in base.items():
        variants[label] = (f, None)
    for label in ("lgb", "lgb+sage (LB 0.95584)"):
        for a in ALPHAS:
            for k in ITERS:
                variants[f"{label} smoothed a={a} k={k}"] = (base[label], (a, k))

    rows, test_preds = [], {}
    for label, (f, sm) in variants.items():
        row = {"variant": label}
        for name, _, pr in tasks:
            s = r(f(name))
            if sm:
                s = smooth(s, *graph[name], *sm)
            if name == "test":
                test_preds[label] = s
            else:
                row[name] = metrics(y[pr], s)["auc"]
        row["orig_mean"] = (row["fold0"] + row["fold1"]) / 2
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "smooth_norm_results.csv", index=False)
    print(res.round(4).to_string(index=False))

    ref = test_preds["lgb+sage (LB 0.95584)"]
    corr = {k: float(np.corrcoef(rankdata(v), rankdata(ref))[0, 1]) for k, v in test_preds.items()}
    print("\nTest rank correlation with the 0.95584 blend:")
    print(pd.Series(corr).round(4).to_string())
    (OUT / "results_smooth_norm.json").write_text(json.dumps(
        dict(results=res.to_dict("records"), test_corr_vs_best=corr), indent=2))

    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    for label, p in test_preds.items():
        fname = (label.replace(" (LB 0.95584)", "").replace("+", "_").replace(" smoothed ", "_smooth_")
                 .replace("=", "").replace(" ", "_"))
        pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
            OUT / f"submission_sn_{fname}_w15.csv", index=False)


if __name__ == "__main__":
    {"trees": run_trees, "nn": run_nn, "score": run_score}[sys.argv[1]]()
