"""Stronger and alternative graph models, each blended 1:1 with the 15-step LightGBM
(LightGBM + GraphSAGE scored 0.95584 on the public LB).

- sage10: the existing 5-seed GraphSAGE plus 5 more seeds
- sage3:  GraphSAGE with a third layer (3-hop neighbourhoods)
- gat:    graph attention, learns how much each neighbour matters
- appnp:  MLP logits propagated 10 hops over the graph

Validation is known to be blind to the GraphSAGE blend's gain (exp_resample_val.py),
so the scores here are a bug check, not a ranking.

Usage:
    .venv/bin/python src/exp_gnn.py nn [model ...]   # e.g. "nn appnp" re-runs one model
    .venv/bin/python src/exp_gnn.py score
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS  # noqa: E402
from exp_smooth_norm import setup  # noqa: E402  (adds the "train <= 17" task)
from features import load  # noqa: E402

TASKS = ("fold1", "c17", "test")
NEW_SEEDS = (5, 6, 7, 8, 9)
KINDS = {"sage_s5to9": "sage", "sage3": "sage3", "gat": "gat", "appnp": "appnp"}


def run_nn(only=()):
    from nn_models import fit_predict_nn, neighbour_operators, standardise

    Xv, mv, tasks = setup()
    _, _, edges = load()
    ops = neighbour_operators(mv.txId.values, edges)
    y, Xarr = mv.label.values, Xv.values
    for m, kind in KINDS.items():
        if only and m not in only:
            continue
        seeds = NEW_SEEDS if m == "sage_s5to9" else range(5)
        for name, tr, pr in tasks:
            if name not in TASKS:
                continue
            t0 = time.time()
            ti, pi = np.flatnonzero(tr), np.flatnonzero(pr)
            Xs = standardise(Xarr, ti)
            p = np.mean([fit_predict_nn(kind, Xs, y, ti, pi, ops=ops, seed=s) for s in seeds], axis=0)
            np.save(PREDS / f"{m}_{name}.npy", p)
            print(f"{m:11s} {name:6s} {time.time() - t0:.0f}s", flush=True)


def run_score():
    from scipy.stats import rankdata

    _, mv, tasks = setup()
    y = mv.label.values
    r = lambda p: rankdata(p) / len(p)
    P = lambda m, n: np.load(PREDS / f"{m}_{n}.npy")
    gnn = {
        "sage (5 seeds)": lambda n: P("sage", n),
        "sage10": lambda n: (P("sage", n) + P("sage_s5to9", n)) / 2,
        "sage3": lambda n: P("sage3", n),
        "gat": lambda n: P("gat", n),
        "appnp": lambda n: P("appnp", n),
    }
    variants = {"lgb": lambda n: P("lgb", n)}
    for g, f in gnn.items():
        variants[g] = f
        variants[f"lgb+{g}"] = lambda n, f=f: (r(P("lgb", n)) + r(f(n))) / 2
    # LightGBM keeps half the weight; the graph half is shared between two models.
    variants["lgb+sage10+gat 2:1:1"] = lambda n: (2 * r(P("lgb", n)) + r(gnn["sage10"](n)) + r(P("gat", n))) / 4
    variants["lgb+sage10+sage3 2:1:1"] = lambda n: (2 * r(P("lgb", n)) + r(gnn["sage10"](n)) + r(P("sage3", n))) / 4

    rows, test_preds = [], {}
    for label, f in variants.items():
        row = {"variant": label}
        for name, _, pr in tasks:
            if name not in TASKS:
                continue
            p = f(name)
            if name == "test":
                test_preds[label] = p
            else:
                row[name] = metrics(y[pr], p)["auc"]
        rows.append(row)
    ref = rankdata(test_preds["lgb+sage (5 seeds)"])
    for row in rows:
        row["test_corr_vs_best"] = float(np.corrcoef(rankdata(test_preds[row["variant"]]), ref)[0, 1])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gnn_results.csv", index=False)
    print(res.round(4).to_string(index=False))
    (OUT / "results_gnn.json").write_text(json.dumps(res.to_dict("records"), indent=2))

    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    for label, p in test_preds.items():
        if label.startswith("lgb+") and label != "lgb+sage (5 seeds)":
            fname = label.replace("+", "_").replace(" ", "_").replace(":", "")
            pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
                OUT / f"submission_gnn_{fname}_w15.csv", index=False)


if __name__ == "__main__":
    {"nn": lambda: run_nn(sys.argv[2:]), "score": run_score}[sys.argv[1]]()
