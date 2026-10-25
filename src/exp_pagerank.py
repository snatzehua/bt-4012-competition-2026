"""Compare the baseline with PageRank features and with the time-proxy features removed.

Usage: .venv/bin/python src/exp_pagerank.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import TIME_PROXIES, build  # noqa: E402
from models import fit_predict_lgb  # noqa: E402
from run import OUT, evaluate, fit_submit  # noqa: E402

VARIANTS = {
    "baseline": dict(),
    "pagerank": dict(pagerank=True),
    "no_time_proxies": dict(drop=TIME_PROXIES),
    "no_time_proxies+pagerank": dict(pagerank=True, drop=TIME_PROXIES),
}


def main():
    results = {}
    for name, kw in VARIANTS.items():
        X, meta = build("raw", **kw)
        r = evaluate(X, meta)
        results[name] = r
        print(f"{name:26s} AUC {r['auc']:.4f}  AP {r['ap']:.4f}  F1 {r['f1']:.4f}  "
              f"(folds AUC {r['auc_fold0']:.4f}/{r['auc_fold1']:.4f})", flush=True)
        fit_submit(X, meta, name.replace("+", "_"))

    # How informative is PageRank on its own, and where does the model rank it?
    X, meta = build("raw", pagerank=True)
    lab = meta.label.notna()
    print("\nsingle-feature AUC on labelled train rows:")
    for c in [c for c in X if c.startswith("pr_")]:
        print(f"  {c:12s} {roc_auc_score(meta.label[lab], X.loc[lab, c]):.4f}")
    _, m = fit_predict_lgb(X[lab], meta.label[lab], X[lab].iloc[:1])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).rank(ascending=False)
    print("gain-importance rank (of %d):" % X.shape[1], imp.filter(like="pr_").astype(int).to_dict())

    (OUT / "results_pagerank.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
