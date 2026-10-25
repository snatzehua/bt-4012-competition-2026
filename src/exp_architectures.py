"""New architectures (CatBoost, MLP, GraphSAGE), each trained on the last 15 time steps,
alone and blended with the 15-step LightGBM (public LB 0.94663).

Tree models and PyTorch models run in separate processes (see common.py for why);
each step saves its predictions to outputs/preds/ and the blend step combines them.

Usage:
    .venv/bin/python src/exp_architectures.py trees
    .venv/bin/python src/exp_architectures.py nn
    .venv/bin/python src/exp_architectures.py blend
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FOLDS, OUT, metrics  # noqa: E402
from features import build, load  # noqa: E402

WINDOW = 15
SEEDS = (0, 1, 2, 3, 4)
PREDS = OUT / "preds"
PREDS.mkdir(exist_ok=True)
MODEL_NAMES = ["lgb", "catboost", "mlp", "sage"]


def setup():
    """Visible graph only (labelled train + test nodes) and the train/predict masks."""
    X, meta = build("raw")
    vis = (meta.label.notna() | meta.is_test).values
    Xv, mv = X[vis].reset_index(drop=True), meta[vis].reset_index(drop=True)
    lab = mv.label.notna().values

    def recent(tr):
        return tr & (mv.time_step > mv.time_step[tr].max() - WINDOW).values

    tasks = []  # (name, train mask, predict mask)
    for f, (tr_ts, va_ts) in enumerate(FOLDS):
        tasks.append((f"fold{f}", recent(lab & mv.time_step.isin(tr_ts).values),
                      lab & mv.time_step.isin(va_ts).values))
    tasks.append(("test", recent(lab), mv.is_test.values))
    return Xv, mv, tasks


def run_trees():
    from models import fit_predict_catboost, fit_predict_lgb

    Xv, mv, tasks = setup()
    y = mv.label.values
    for m, fn in (("lgb", fit_predict_lgb), ("catboost", fit_predict_catboost)):
        for name, tr, pr in tasks:
            t0 = time.time()
            np.save(PREDS / f"{m}_{name}.npy", fn(Xv[tr], y[tr], Xv[pr])[0])
            print(f"{m:9s} {name:6s} {time.time() - t0:.0f}s", flush=True)


def run_nn():
    from nn_models import fit_predict_nn, neighbour_operators, standardise

    Xv, mv, tasks = setup()
    _, _, edges = load()
    ops = neighbour_operators(mv.txId.values, edges)
    y, Xarr = mv.label.values, Xv.values
    for m in ("mlp", "sage"):
        for name, tr, pr in tasks:
            t0 = time.time()
            ti, pi = np.flatnonzero(tr), np.flatnonzero(pr)
            Xs = standardise(Xarr, ti)
            p = np.mean([fit_predict_nn(m, Xs, y, ti, pi, ops=ops, seed=s) for s in SEEDS], axis=0)
            np.save(PREDS / f"{m}_{name}.npy", p)
            print(f"{m:9s} {name:6s} {time.time() - t0:.0f}s", flush=True)


def run_blend():
    from scipy.stats import rankdata

    def rank_mean(ps):
        return np.mean([rankdata(p) / len(p) for p in ps], axis=0)

    _, mv, tasks = setup()
    y = mv.label.values
    P = {(m, name): np.load(PREDS / f"{m}_{name}.npy") for m in MODEL_NAMES for name, _, _ in tasks}
    folds = [(name, y[pr]) for name, _, pr in tasks if name.startswith("fold")]

    results = {}

    def summarise(label, get):
        df = pd.DataFrame([metrics(yf, get(name)) for name, yf in folds])
        r = df.mean().to_dict() | {f"auc_fold{i}": a for i, a in enumerate(df.auc)}
        results[label] = r
        print(f"{label:22s} AUC {r['auc']:.4f}  AP {r['ap']:.4f}  "
              f"(folds AUC {r['auc_fold0']:.4f}/{r['auc_fold1']:.4f})")

    combos = {m: [m] for m in MODEL_NAMES}
    combos |= {f"blend lgb+{m}": ["lgb", m] for m in MODEL_NAMES[1:]}
    combos["blend all four"] = MODEL_NAMES
    for label, ms in combos.items():
        summarise(label, lambda name, ms=ms: P[(ms[0], name)] if len(ms) == 1
                  else rank_mean([P[(m, name)] for m in ms]))
    (OUT / "results_architectures.json").write_text(json.dumps(results, indent=2))

    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    for label, ms in combos.items():
        if ms == ["lgb"]:
            continue
        p = P[(ms[0], "test")] if len(ms) == 1 else rank_mean([P[(m, "test")] for m in ms])
        fname = "arch_" + label.replace("blend ", "blend_").replace("+", "_").replace(" ", "_") + "_w15"
        pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
            OUT / f"submission_{fname}.csv", index=False)

    ranks = pd.DataFrame({m: rankdata(P[(m, "test")]) for m in MODEL_NAMES})
    print("\nRank correlation between models' test predictions:")
    print(ranks.corr().round(3).to_string())
    print("\nPredicted fraud share on test (prob >= 0.5):",
          {m: f"{(P[(m, 'test')] >= 0.5).mean():.1%}" for m in MODEL_NAMES})


if __name__ == "__main__":
    {"trees": run_trees, "nn": run_nn, "blend": run_blend}[sys.argv[1]]()
