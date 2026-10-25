"""Shared settings with no ML-library imports.

Kept separate so PyTorch scripts can use them without loading LightGBM/CatBoost:
on macOS their OpenMP runtimes clash with PyTorch's in one process (crash or hang).
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

# Expanding-window folds: train on everything before the validation window.
FOLDS = [(range(1, 24), range(24, 30)), (range(1, 30), range(30, 36))]


def metrics(y, p):
    # Best-threshold F1 is reported for reference; the ranking metrics don't need a threshold.
    ths = np.quantile(p, np.linspace(0.5, 0.99, 50))
    f1 = max(f1_score(y, p >= t) for t in ths)
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), f1=f1)
