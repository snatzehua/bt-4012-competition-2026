import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import rankdata

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    n_estimators=700,
    num_leaves=31,
    min_child_samples=30,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
)


# Cautious variant: shallow trees, big leaves and fewer features per tree, so the
# model can't lean on fine-grained patterns that may not survive the drift.
LGB_REG_PARAMS = dict(
    LGB_PARAMS,
    learning_rate=0.02,
    n_estimators=800,
    num_leaves=8,
    max_depth=4,
    min_child_samples=200,
    colsample_bytree=0.3,
    reg_lambda=10.0,
)


def fit_predict_lgb(Xtr, ytr, Xte, seed=0, sample_weight=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return m.predict_proba(Xte)[:, 1], m


def fit_predict_lgb_reg(Xtr, ytr, Xte, seed=0, sample_weight=None):
    m = lgb.LGBMClassifier(**LGB_REG_PARAMS, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return m.predict_proba(Xte)[:, 1], m


def fit_predict_xgb(Xtr, ytr, Xte, seed=0, sample_weight=None):
    m = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.5, min_child_weight=3, reg_lambda=1.0,
        tree_method="hist", n_jobs=-1, random_state=seed, eval_metric="auc",
    )
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return m.predict_proba(Xte)[:, 1], m


def fit_predict_rf(Xtr, ytr, Xte, seed=0, sample_weight=None):
    m = RandomForestClassifier(
        n_estimators=500, max_features=0.2, min_samples_leaf=2,
        n_jobs=-1, random_state=seed,
    )
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return m.predict_proba(Xte)[:, 1], m


def fit_predict_catboost(Xtr, ytr, Xte, seed=0, sample_weight=None):
    m = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, thread_count=-1,
    )
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return m.predict_proba(Xte)[:, 1], m


def rank_mean(preds):
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


MODELS = {"lgb": fit_predict_lgb, "lgb_reg": fit_predict_lgb_reg, "xgb": fit_predict_xgb,
          "rf": fit_predict_rf, "catboost": fit_predict_catboost}
