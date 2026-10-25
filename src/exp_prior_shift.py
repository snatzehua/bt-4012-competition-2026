"""Per-time-step fraud-rate (prior-shift) correction for the best blend
(LightGBM : GraphSAGE 10 seeds : GAT at 2:1:1, public LB 0.95627).

AUC is pooled over all test steps, but the models' predictions say fraud almost
vanishes after step 43, while those steps' nodes sit mid-ranking in the rank blend.
Here each step's fraud rate is estimated from the models' own probabilities
(Saerens et al. 2002 EM), each model's probabilities are re-weighted from the
training fraud rate to the step's estimated rate, and the results are rank-blended
as before. `strength` interpolates the target rate between the training rate (0,
no change) and the step estimate (1). No labels are used for test.

An "oracle" variant using the true per-step fraud rate shows the ceiling on validation.

Usage: .venv/bin/python src/exp_prior_shift.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, metrics  # noqa: E402
from exp_architectures import PREDS  # noqa: E402
from exp_smooth_norm import setup  # noqa: E402

TASKS = ("fold1", "c17", "test")
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
MODELS = {"lgb": 2, "sage10": 1, "gat": 1}  # rank-blend weights


def load_preds(name):
    P = lambda m: np.load(PREDS / f"{m}_{name}.npy")
    return {"lgb": P("lgb"), "sage10": (P("sage") + P("sage_s5to9")) / 2, "gat": P("gat")}


def reprior(p, pi_from, pi_to):
    """Bayes re-weighting of P(fraud | x) from one class prior to another."""
    a = p * pi_to / pi_from
    b = (1 - p) * (1 - pi_to) / (1 - pi_from)
    return a / (a + b)


def em_prior(p, pi_train, iters=100, tol=1e-8):
    pi = pi_train
    for _ in range(iters):
        new = reprior(p, pi_train, pi).mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def adjust(p, steps, pi_train, strength, oracle=None):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    out = np.empty_like(p)
    for s in np.unique(steps):
        m = steps == s
        pi_s = oracle[s] if oracle is not None else em_prior(p[m], pi_train)
        target = np.exp((1 - strength) * np.log(pi_train) + strength * np.log(max(pi_s, 1e-5)))
        out[m] = reprior(p[m], pi_train, target)
    return out


def blend(P):
    return sum(w * rankdata(P[m]) / len(P[m]) for m, w in MODELS.items()) / sum(MODELS.values())


def main():
    _, mv, tasks = setup()
    y, ts = mv.label.values, mv.time_step.values
    rows, test_preds, step_rates = [], {}, {}
    for name, tr, pr in tasks:
        if name not in TASKS:
            continue
        P, steps = load_preds(name), ts[pr]
        pi_train = float(np.nanmean(y[tr]))
        oracle = None if name == "test" else pd.Series(y[pr]).groupby(steps).mean().to_dict()
        variants = {"best blend (no adjustment)": blend(P)}
        for k in STRENGTHS:
            variants[f"adjusted, strength {k}"] = blend({m: adjust(p, steps, pi_train, k) for m, p in P.items()})
        if oracle is not None:
            variants["oracle (true step rates)"] = blend({m: adjust(p, steps, pi_train, 1.0, oracle) for m, p in P.items()})
        # Estimated vs true step rates, from the averaged model probabilities.
        pm = np.mean([np.clip(p, 1e-7, 1 - 1e-7) for p in P.values()], axis=0)
        step_rates[name] = {int(s): dict(est=em_prior(pm[steps == s], pi_train),
                                         true=None if oracle is None else oracle[s]) for s in np.unique(steps)}
        for label, p in variants.items():
            if name == "test":
                test_preds[label] = p
            else:
                rows.append(dict(task=name, variant=label, auc=metrics(y[pr], p)["auc"]))
    res = pd.DataFrame(rows).pivot(index="variant", columns="task", values="auc")
    ref = rankdata(test_preds["best blend (no adjustment)"])
    res["test_corr_vs_best"] = pd.Series({k: np.corrcoef(rankdata(v), ref)[0, 1] for k, v in test_preds.items()})
    print(res.round(4).to_string())
    for name in ("c17", "test"):
        print(f"\nstep fraud rates, {name} (EM estimate from averaged probabilities"
              f"{'' if name == 'test' else ' vs truth'}):")
        print(pd.DataFrame(step_rates[name]).T.astype(float).round(3).T.to_string())
    res.to_csv(OUT / "prior_shift_results.csv")
    (OUT / "results_prior_shift.json").write_text(json.dumps(
        dict(results=res.reset_index().to_dict("records"), step_rates=step_rates), indent=2, default=float))

    te = [pr for name, _, pr in tasks if name == "test"][0]
    idx = mv["index"][te].astype(int).values
    for label, p in test_preds.items():
        if label.startswith("adjusted"):
            k = label.split()[-1]
            pd.DataFrame({"index": idx, "target": p}).sort_values("index").to_csv(
                OUT / f"submission_prior_s{k}_lgb_sage10_gat_211_w15.csv", index=False)


if __name__ == "__main__":
    main()
