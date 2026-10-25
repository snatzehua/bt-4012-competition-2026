"""Ideas 3-5 for robustness to the test-period shift: cautious models, recency, blending.

Usage: .venv/bin/python src/exp_robust.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build  # noqa: E402
from models import MODELS, rank_mean  # noqa: E402
from run import FOLDS, OUT, metrics  # noqa: E402

TOP_FEATURES = ["feat_53", "feat_55", "feat_90", "feat_5", "feat_89"]


def recency_weights(ts, half_life):
    return 0.5 ** ((ts.max() - ts) / half_life)


# name -> (models, how to pick/weight the training rows)
EXPERIMENTS = {
    "baseline_lgb": (["lgb"], {}),
    "3a_lgb_regularised": (["lgb_reg"], {}),
    "3b_random_forest": (["rf"], {}),
    "4a_recent_15_steps": (["lgb"], {"window": 15}),
    "4b_decay_halflife_10": (["lgb"], {"half_life": 10}),
    "4c_decay_halflife_5": (["lgb"], {"half_life": 5}),
    "5_blend_lgb_reg_rf": (["lgb", "lgb_reg", "rf"], {}),
}


def fit_predict(X, meta, tr, te, models, window=None, half_life=None):
    ts = meta.time_step[tr]
    if window:
        tr = tr & (meta.time_step > ts.max() - window)
        ts = meta.time_step[tr]
    w = recency_weights(ts, half_life).values if half_life else None
    preds = [MODELS[m](X[tr], meta.label[tr], X[te], sample_weight=w)[0] for m in models]
    return preds[0] if len(preds) == 1 else rank_mean(preds)


def separation_over_time(X, meta):
    """Single-feature AUC of the top fraud signals at each labelled time step."""
    lab = meta.label.notna()
    rows = {}
    for t, g in meta[lab].groupby("time_step"):
        y = g.label.values
        if y.min() == y.max():
            continue
        rows[t] = {f: max(a, 1 - a) for f in TOP_FEATURES
                   for a in [roc_auc_score(y, X.loc[g.index, f])]}
    return pd.DataFrame(rows).T


def main():
    X, meta = build("raw")
    lab = meta.label.notna()

    sep = separation_over_time(X, meta)
    print("Single-feature AUC of the top signals, averaged over blocks of time steps:")
    blocks = pd.cut(sep.index, [0, 7, 14, 21, 28, 35], labels=["1-7", "8-14", "15-21", "22-28", "29-35"])
    print(sep.groupby(blocks, observed=True).mean().round(3).to_string(), "\n")

    results = {}
    for name, (models, kw) in EXPERIMENTS.items():
        fold = []
        for tr_ts, va_ts in FOLDS:
            tr = lab & meta.time_step.isin(tr_ts)
            va = lab & meta.time_step.isin(va_ts)
            fold.append(metrics(meta.label[va].values, fit_predict(X, meta, tr, va, models, **kw)))
        df = pd.DataFrame(fold)
        r = df.mean().to_dict() | {f"auc_fold{i}": a for i, a in enumerate(df.auc)}
        results[name] = r
        print(f"{name:24s} AUC {r['auc']:.4f}  AP {r['ap']:.4f}  F1 {r['f1']:.4f}  "
              f"(folds AUC {r['auc_fold0']:.4f}/{r['auc_fold1']:.4f})", flush=True)

        p = fit_predict(X, meta, lab, meta.is_test, models, **kw)
        te = meta.is_test
        sub = pd.DataFrame({"index": meta["index"][te].astype(int).values, "target": p}).sort_values("index")
        sub.to_csv(OUT / f"submission_{name}.csv", index=False)

    (OUT / "results_robust.json").write_text(json.dumps(results, indent=2))

    # How differently do the candidates rank the test set?
    subs = {n: pd.read_csv(OUT / f"submission_{n}.csv").target.rank() for n in EXPERIMENTS}
    print("\nRank correlation of test predictions with the baseline:")
    print(pd.Series({n: np.corrcoef(subs["baseline_lgb"], s)[0, 1] for n, s in subs.items()}).round(3).to_string())


if __name__ == "__main__":
    main()
