"""Run the experiment ladder with time-based validation and write submissions.

Usage: .venv/bin/python src/run.py
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FOLDS, OUT, metrics  # noqa: E402,F401  (re-exported for older scripts)
from features import build  # noqa: E402
from models import MODELS, fit_predict_lgb, rank_mean  # noqa: E402


def adversarial_drop(X, meta, k):
    """Rank features by how well they separate train-period from test-period rows."""
    vis = meta.label.notna() | meta.is_test
    Xa, ya = X[vis], meta.is_test[vis].astype(int)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.5, verbose=-1, n_jobs=-1)
    m.fit(Xa, ya)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return list(imp.index[:k]), imp


def evaluate(X, meta, models=("lgb",), seeds=(0,)):
    res = []
    for tr_ts, va_ts in FOLDS:
        lab = meta.label.notna()
        tr = lab & meta.time_step.isin(tr_ts)
        va = lab & meta.time_step.isin(va_ts)
        preds = [MODELS[m](X[tr], meta.label[tr], X[va], seed=s)[0] for m in models for s in seeds]
        p = preds[0] if len(preds) == 1 else rank_mean(preds)
        res.append(metrics(meta.label[va].values, p))
    df = pd.DataFrame(res)
    out = df.mean().to_dict()
    out.update({f"{k}_fold{i}": v for i, row in df.iterrows() for k, v in row.items()})
    return out


def fit_submit(X, meta, name, models=("lgb",), seeds=(0,)):
    tr = meta.label.notna()
    te = meta.is_test
    preds = [MODELS[m](X[tr], meta.label[tr], X[te], seed=s)[0] for m in models for s in seeds]
    p = preds[0] if len(preds) == 1 else rank_mean(preds)
    sub = pd.DataFrame({"index": meta["index"][te].astype(int).values, "target": p}).sort_values("index")
    path = OUT / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    return path


def main():
    results = {}

    def log(name, r, t0):
        results[name] = r
        print(f"{name:28s} AUC {r['auc']:.4f}  AP {r['ap']:.4f}  F1 {r['f1']:.4f}  "
              f"(folds AUC {r['auc_fold0']:.4f}/{r['auc_fold1']:.4f})  {time.time()-t0:.0f}s", flush=True)

    Xr, meta = build("raw")
    Xg, _ = build("graph")
    print("raw", Xr.shape, "graph", Xg.shape, flush=True)

    t0 = time.time(); log("1_baseline_lgb_raw", evaluate(Xr, meta), t0)
    t0 = time.time(); log("1b_rf_raw", evaluate(Xr, meta, models=("rf",)), t0)
    t0 = time.time(); log("2_lgb_graph", evaluate(Xg, meta), t0)

    drop, imp = adversarial_drop(Xg, meta, k=30)
    imp.head(30).to_csv(OUT / "adversarial_top_features.csv")
    Xd = Xg.drop(columns=drop)
    t0 = time.time(); log("3_lgb_graph_dropdrift30", evaluate(Xd, meta), t0)

    best_X = Xd if results["3_lgb_graph_dropdrift30"]["auc"] >= results["2_lgb_graph"]["auc"] else Xg
    best_name = "dropdrift" if best_X is Xd else "graph"
    ens = dict(models=("lgb", "xgb", "rf"), seeds=(0, 1, 2))
    t0 = time.time(); log(f"4_ensemble_{best_name}", evaluate(best_X, meta, **ens), t0)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    print("writing submissions", flush=True)
    print(fit_submit(Xr, meta, "baseline"))
    print(fit_submit(best_X, meta, f"ensemble_{best_name}", **ens))


if __name__ == "__main__":
    main()
