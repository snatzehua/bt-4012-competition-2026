"""Search for a validation scheme whose scores rank our submitted models the way the
public leaderboard does.

Each of the 9 submitted models is re-run from several training cutoffs c (train on
steps <= c, with the model's own recency window), predicting every labelled node
after c. Validation schemes are then different ways of scoring those predictions:
which steps to score, how far ahead of the cutoff, and how to weight nodes
(adversarial importance weights, test-like subsets, per-step averages). A scheme is
judged by the rank correlation between its scores and the 9 public LB scores.

Tree models and PyTorch run in separate processes (see common.py).

Usage:
    .venv/bin/python src/exp_resample_val.py trees
    .venv/bin/python src/exp_resample_val.py nn
    .venv/bin/python src/exp_resample_val.py score
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from features import TIME_PROXIES, build, load  # noqa: E402

CUTOFFS = (17, 19, 21, 23, 25, 27, 29, 31)
SEEDS = (0, 1, 2, 3, 4)
PREDS = OUT / "preds_cv"
PREDS.mkdir(exist_ok=True)
HORIZON = 14  # test covers 14 steps (36-49) after the last labelled step

# Submitted models and their public LB scores (docs/submissions.csv).
LB = {
    "baseline_all": 0.94067, "rf_all": 0.92120, "lgb_w15": 0.94663, "selftrain_w15": 0.93033,
    "lgb_w10": 0.93204, "lgb_w20": 0.94296, "cat_w15": 0.94350,
    "blend_lgb_cat_w15": 0.94736, "blend_lgb_sage_w15": 0.95584,
}


def setup():
    """Visible graph only (labelled train + test nodes), as in exp_architectures."""
    X, meta = build("raw")
    vis = (meta.label.notna() | meta.is_test).values
    return X[vis].reset_index(drop=True), meta[vis].reset_index(drop=True)


def masks(mv, c, window=None):
    lab = mv.label.notna().values
    ts = mv.time_step.values
    tr = lab & (ts <= c)
    if window is not None:
        tr &= ts > c - window
    return tr, lab & (ts > c)


def run_trees():
    from exp_window_selftrain import self_train
    from models import fit_predict_catboost, fit_predict_lgb, fit_predict_rf

    Xv, mv = setup()
    y = mv.label
    jobs = {
        "lgb_all": (fit_predict_lgb, None), "lgb_w10": (fit_predict_lgb, 10),
        "lgb_w15": (fit_predict_lgb, 15), "lgb_w20": (fit_predict_lgb, 20),
        "rf_all": (fit_predict_rf, None), "cat_w15": (fit_predict_catboost, 15),
    }
    for c in CUTOFFS:
        for name, (fn, w) in jobs.items():
            t0 = time.time()
            tr, ev = masks(mv, c, w)
            np.save(PREDS / f"{name}_c{c}.npy", fn(Xv[tr], y[tr], Xv[ev])[0])
            print(f"c={c} {name:9s} {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        tr, ev = masks(mv, c, 15)
        # Self-training's target is the whole period after the cutoff, labels hidden.
        np.save(PREDS / f"selftrain_w15_c{c}.npy",
                self_train(Xv, mv, pd.Series(tr), pd.Series(ev), hi=0.9, lo=0.02))
        print(f"c={c} selftrain {time.time() - t0:.0f}s", flush=True)


def run_nn():
    from nn_models import fit_predict_nn, neighbour_operators, standardise

    Xv, mv = setup()
    _, _, edges = load()
    ops = neighbour_operators(mv.txId.values, edges)
    y, Xarr = mv.label.values, Xv.values
    for c in CUTOFFS:
        t0 = time.time()
        tr, ev = masks(mv, c, 15)
        ti, pi = np.flatnonzero(tr), np.flatnonzero(ev)
        Xs = standardise(Xarr, ti)
        p = np.mean([fit_predict_nn("sage", Xs, y, ti, pi, ops=ops, seed=s) for s in SEEDS], axis=0)
        np.save(PREDS / f"sage_w15_c{c}.npy", p)
        print(f"c={c} sage {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- scoring

def adversarial_weights(Xv, mv, drop_time=False, target_auc=None):
    """Out-of-fold P(test | x) for labelled nodes, turned into importance weights p/(1-p).

    With all features the two periods separate perfectly (AUC 1.0) and every weight
    hits the floor. With target_auc set, the most period-revealing features are
    dropped 10 at a time until the classifier only partly separates them.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cols = [c for c in Xv.columns if not (drop_time and c in TIME_PROXIES)]
    rows = (mv.label.notna() | mv.is_test).values
    ya = mv.is_test.values[rows].astype(int)
    while True:
        Xa = Xv.loc[rows, cols].values
        oof, gain = np.zeros(len(ya)), np.zeros(len(cols))
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya)):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   colsample_bytree=0.5, verbose=-1, random_state=k)
            m.fit(Xa[a], ya[a])
            oof[b] = m.predict_proba(Xa[b])[:, 1]
            gain += m.booster_.feature_importance("gain")
        a_auc = roc_auc_score(ya, oof)
        if target_auc is None or a_auc <= target_auc or len(cols) <= 20:
            break
        cols = [c for _, c in sorted(zip(gain, cols), reverse=True)[10:]]
    print(f"adversarial AUC {a_auc:.4f} with {len(cols)} features "
          f"(drop_time={drop_time}, target={target_auc})")
    p = np.full(len(mv), np.nan)
    p[rows] = np.clip(oof, 1e-4, 1 - 1e-4)
    return p / (1 - p)


def run_score():
    from scipy.stats import kendalltau, rankdata, spearmanr
    from sklearn.metrics import roc_auc_score

    Xv, mv = setup()
    y = mv.label.values
    ts = mv.time_step.values
    r = lambda p: rankdata(p) / len(p)

    BASE = ("lgb_all", "rf_all", "lgb_w15", "selftrain_w15", "lgb_w10", "lgb_w20", "cat_w15", "sage_w15")
    MODEL = {
        "baseline_all": ["lgb_all"], "rf_all": ["rf_all"], "lgb_w15": ["lgb_w15"],
        "selftrain_w15": ["selftrain_w15"], "lgb_w10": ["lgb_w10"], "lgb_w20": ["lgb_w20"],
        "cat_w15": ["cat_w15"], "blend_lgb_cat_w15": ["lgb_w15", "cat_w15"],
        "blend_lgb_sage_w15": ["lgb_w15", "sage_w15"],
    }

    PC = {c: {n: np.load(PREDS / f"{n}_c{c}.npy") for n in BASE} for c in CUTOFFS}
    W = {"adv_all": adversarial_weights(Xv, mv), "adv_notime": adversarial_weights(Xv, mv, drop_time=True),
         "adv_reduced90": adversarial_weights(Xv, mv, target_auc=0.90),
         "adv_reduced80": adversarial_weights(Xv, mv, target_auc=0.80)}
    names = list(LB)
    lb = np.array([LB[n] for n in names])

    def auc(c, steps, model, weight=None, per_step=False, subset=None):
        _, ev = masks(mv, c)
        sel = np.isin(ts[ev], steps)
        if subset is not None:
            sel &= subset[ev]
        # Blends are re-ranked inside the scored subset, as they would be on test.
        parts = [PC[c][m][sel] for m in MODEL[model]]
        p = parts[0] if len(parts) == 1 else np.mean([r(q) for q in parts], axis=0)
        yy, tt = y[ev][sel], ts[ev][sel]
        w = None if weight is None else weight[ev][sel]
        if per_step:
            vals = [roc_auc_score(yy[tt == s], p[tt == s]) for s in np.unique(tt)
                    if 0 < yy[tt == s].sum() < (tt == s).sum()]
            return float(np.mean(vals))
        return roc_auc_score(yy, p, sample_weight=w)

    def scheme(spec):
        """spec: list of (cutoff, steps); scores averaged over the list."""
        return np.array([np.mean([auc(c, s, n, **spec.get("kw", {})) for c, s in spec["folds"]])
                         for n in names])

    schemes = {
        "original (23->24-29, 29->30-35)": dict(folds=[(23, range(24, 30)), (29, range(30, 36))]),
    }
    for c in CUTOFFS:
        schemes[f"c={c} -> all after"] = dict(folds=[(c, range(c + 1, 36))])
    schemes["horizon 14: 21->22-35"] = dict(folds=[(21, range(22, 36))])
    for gap in (4, 6, 8, 10):
        f = [(c, range(c + gap, 36)) for c in CUTOFFS if c + gap <= 33]
        schemes[f"far only: steps >= c+{gap}, all cutoffs"] = dict(folds=f)
    for c in (17, 19, 21):
        schemes[f"far only: {c}->{c + 8}-35"] = dict(folds=[(c, range(c + 8, 36))])
    for k in W:
        schemes[f"{k} weighted: 21->22-35"] = dict(folds=[(21, range(22, 36))], kw=dict(weight=W[k]))
        schemes[f"{k} weighted: all cutoffs"] = dict(folds=[(c, range(c + 1, 36)) for c in CUTOFFS],
                                                    kw=dict(weight=W[k]))
        for q in (0.5, 0.75, 0.9):
            _, ev = masks(mv, 21)
            top = np.zeros(len(mv), bool)
            top[np.flatnonzero(ev)[rankdata(W[k][ev], method="ordinal") > q * ev.sum()]] = True
            schemes[f"{k} top {1 - q:.0%} test-like: 21->22-35"] = dict(
                folds=[(21, range(22, 36))], kw=dict(subset=top))
    schemes["per-step mean: 21->22-35"] = dict(folds=[(21, range(22, 36))], kw=dict(per_step=True))
    schemes["per-step mean: all cutoffs"] = dict(folds=[(c, range(c + 1, 36)) for c in CUTOFFS],
                                                 kw=dict(per_step=True))
    schemes["low-fraud steps (<10%) only: 21->"] = dict(
        folds=[(21, [s for s in range(22, 36) if np.nanmean(y[ts == s]) < 0.10])])

    # Pairs the LB separates by more than its likely noise (~0.003 AUC).
    pairs = [(i, j) for i, j in combinations(range(len(names)), 2) if abs(lb[i] - lb[j]) > 0.003]

    rows, scores = [], {}
    for label, spec in schemes.items():
        s = scheme(spec)
        scores[label] = dict(zip(names, s.round(5)))
        agree = np.mean([np.sign(s[i] - s[j]) == np.sign(lb[i] - lb[j]) for i, j in pairs])
        # Leave-one-out stability: Spearman with each model dropped in turn.
        loo = [spearmanr(np.delete(s, k), np.delete(lb, k)).correlation for k in range(len(names))]
        rows.append(dict(scheme=label, spearman=spearmanr(s, lb).correlation,
                         kendall=kendalltau(s, lb).correlation, pearson=np.corrcoef(s, lb)[0, 1],
                         pair_agree=agree, loo_min=min(loo), spread=s.max() - s.min()))
        print(f"{label:48s} rho {rows[-1]['spearman']:+.3f}  pairs {agree:.0%}", flush=True)

    res = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    # Selection-adjusted null: with the LB scores shuffled, how high does the best
    # scheme's Spearman get by chance?
    S = np.array([[scores[l][n] for n in names] for l in schemes])
    rng = np.random.default_rng(0)
    null = [max(spearmanr(row, perm).correlation for row in S)
            for perm in (rng.permutation(lb) for _ in range(2000))]
    best = res.spearman.iloc[0]
    print(f"\nbest-of-{len(schemes)} Spearman under shuffled LB: median {np.median(null):.3f}, "
          f"95th pct {np.quantile(null, 0.95):.3f}; P(null >= {best:.3f}) = {np.mean(np.array(null) >= best):.3f}")
    res.to_csv(OUT / "resample_val_schemes.csv", index=False)
    pd.DataFrame(scores).T.to_csv(OUT / "resample_val_scores.csv")
    print(f"\n{len(pairs)} model pairs separated by > 0.003 on the LB\n")
    print(res.round(3).to_string(index=False))
    (OUT / "results_resample_val.json").write_text(json.dumps(
        dict(schemes=res.to_dict("records"), scores=scores), indent=2, default=float))


if __name__ == "__main__":
    {"trees": run_trees, "nn": run_nn, "score": run_score}[sys.argv[1]]()
