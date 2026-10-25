"""Shorter training windows, and self-training on the unlabelled target period.

Self-training: fit on the recent window, predict the target rows, add the ones the
model is most confident about back in as pseudo-labelled training rows (at lower
weight), refit, repeat. In validation the "target" is the held-out window with its
labels hidden, which simulates exactly what happens on test.

Usage: .venv/bin/python src/exp_window_selftrain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build  # noqa: E402
from models import fit_predict_lgb  # noqa: E402
from run import FOLDS, OUT, metrics  # noqa: E402


def recent(meta, tr, window):
    return tr & (meta.time_step > meta.time_step[tr].max() - window)


def self_train(X, meta, tr, target, rounds=2, hi=0.9, lo=0.02, weight=0.5, log=None):
    """Returns predictions for `target` after `rounds` of pseudo-labelling."""
    p, _ = fit_predict_lgb(X[tr], meta.label[tr], X[target])
    for r in range(rounds):
        conf = (p >= hi) | (p <= lo)
        Xp = X[target][conf]
        yp = (p[conf] >= hi).astype(float)
        Xa = pd.concat([X[tr], Xp])
        ya = np.r_[meta.label[tr].values, yp]
        w = np.r_[np.ones(tr.sum()), np.full(len(yp), weight)]
        if log is not None:
            log.append(dict(round=r + 1, pseudo=int(conf.sum()), share=float(conf.mean()), fraud=int(yp.sum())))
        p, _ = fit_predict_lgb(Xa, ya, X[target], sample_weight=w)
    return p


def run_cv(X, meta, predict):
    lab = meta.label.notna()
    fold = []
    for tr_ts, va_ts in FOLDS:
        tr = lab & meta.time_step.isin(tr_ts)
        va = lab & meta.time_step.isin(va_ts)
        fold.append(metrics(meta.label[va].values, predict(tr, va)))
    df = pd.DataFrame(fold)
    return df.mean().to_dict() | {f"auc_fold{i}": a for i, a in enumerate(df.auc)}


def write_sub(meta, p, name):
    te = meta.is_test
    sub = pd.DataFrame({"index": meta["index"][te].astype(int).values, "target": p}).sort_values("index")
    sub.to_csv(OUT / f"submission_{name}.csv", index=False)


def main():
    X, meta = build("raw")
    lab, te = meta.label.notna(), meta.is_test
    results, preds = {}, {}

    def report(name, r):
        results[name] = r
        print(f"{name:26s} AUC {r['auc']:.4f}  AP {r['ap']:.4f}  F1 {r['f1']:.4f}  "
              f"(folds AUC {r['auc_fold0']:.4f}/{r['auc_fold1']:.4f})", flush=True)

    for w in (5, 8, 10, 12, 15, 20):
        name = f"window_{w}"
        report(name, run_cv(X, meta, lambda tr, va: fit_predict_lgb(
            X[recent(meta, tr, w)], meta.label[recent(meta, tr, w)], X[va])[0]))
        preds[name], _ = fit_predict_lgb(X[recent(meta, lab, w)], meta.label[recent(meta, lab, w)], X[te])
        write_sub(meta, preds[name], name)

    for w in (10, 15):
        for hi, lo in ((0.9, 0.02), (0.97, 0.005)):
            name = f"selftrain_w{w}_hi{hi}_lo{lo}"
            report(name, run_cv(X, meta, lambda tr, va: self_train(X, meta, recent(meta, tr, w), va, hi=hi, lo=lo)))
            log = []
            preds[name] = self_train(X, meta, recent(meta, lab, w), te, hi=hi, lo=lo, log=log)
            print("   on test:", ", ".join(f"round {d['round']}: {d['pseudo']:,} rows "
                                           f"({d['share']:.0%}), {d['fraud']} fraud" for d in log))
            write_sub(meta, preds[name], name)

    (OUT / "results_window_selftrain.json").write_text(json.dumps(results, indent=2))
    base = pd.read_csv(OUT / "submission_4a_recent_15_steps.csv").target.rank()
    print("\nRank correlation with the submitted window-15 model (0.94663):")
    print(pd.Series({n: np.corrcoef(base, pd.Series(p).rank())[0, 1] for n, p in preds.items()}).round(3).to_string())


if __name__ == "__main__":
    main()
