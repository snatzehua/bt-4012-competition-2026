# BT4012: Fraudulent Bitcoin Transaction Detection

Experiments for the BT4012 Kaggle competition on classifying illicit Bitcoin transactions using anonymised transaction features and a directed transaction graph. The strongest recorded public submission combines LightGBM with a GraphSAGE-style model trained on recent time steps.

## Setup

The experiments used Python **3.12.9** on macOS with CPU training. The commands below assume a Unix-like shell and should be run from the repository root.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements file pins the direct dependencies from the experiment environment, including the plotting libraries. It is not a complete lockfile of transitive dependencies. On macOS, a missing OpenMP runtime for LightGBM may require `brew install libomp`.

Run tree-model and PyTorch training commands as separate Python processes, as shown below. Loading both training stacks in one process caused OpenMP crashes or hangs in the macOS experiment environment.

## Data placement

Download the supplied CSV files from the [Kaggle competition](https://www.kaggle.com/competitions/bt-4012-competition-2026/data) using an account with competition access. Place them directly in `data/`:

```text
data/
  train.csv
  test.csv
  txs_edgelist.csv
  sample_submission.csv
```

The dataset is excluded from Git. Do not rename the supplied columns or reorder the feature columns. The supplied files contain:

| File | Contents |
| --- | --- |
| `train.csv` | 141,772 transactions from steps 1–35; `txId`, `time_step`, `feat_1`–`feat_165`, and `label` |
| `test.csv` | 15,329 transactions from steps 36–49; `index`, `txId`, `time_step`, and `feat_1`–`feat_165` |
| `txs_edgelist.csv` | 234,355 directed edges, with columns `txId1` and `txId2` |
| `sample_submission.csv` | Expected submission columns `index,target` |

Labels are `1` for illicit and `0` for licit; missing labels are unknown and are excluded from supervised training. There are 31,235 labelled training transactions, including 3,644 illicit transactions.

The scripts read Parquet versions of the training and test files. Create these once before running any experiments:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

data = Path("data")
for name in ("train", "test"):
    frame = pd.read_csv(data / f"{name}.csv")
    frame.to_parquet(data / f"{name}.parquet", index=False)
    print(f"Prepared {name}.parquet: {len(frame):,} rows")
PY
```

Keep `txs_edgelist.csv` and `sample_submission.csv` in CSV format. Data paths are resolved relative to the repository by the scripts.

## Recorded submission outputs

The following entries are recorded candidates. Confirm the two entries actually selected for the final private leaderboard against Kaggle; this repository does not currently record that selection.

| Submission | Kaggle ID | Model | Recorded public score | Output filename inside `outputs/` |
| --- | --- | --- | --- | --- |
| #16 | 57563177 | LightGBM, 5 seeds + local-feature GraphSAGE, 5 seeds; equal-weight rank blend | 0.95938 | `submission_more_lgb5_sage_local_1_1_w15.csv` |
| #18 | 57563185 | LightGBM, 5 seeds + all-feature GraphSAGE, 10 seeds + GAT, 5 seeds; rank weights 2:1:1 | 0.95641 | `submission_final_lgb5_sage10_gat_2_1_1_w15.csv` |
| #8 | 57424323 | LightGBM + CatBoost; equal-weight rank blend | 0.94736 | `submission_arch_blend_lgb_catboost_w15.csv` |

All three train on the last 15 training time steps, 21–35. LightGBM uses all 165 supplied features; the local-feature GraphSAGE uses `feat_1`–`feat_93`. Within each model family, seed predictions are averaged before converting them to percentile ranks and combining the models. The blended outputs are ranking scores, not calibrated probabilities.

See [docs/submissions.csv](docs/submissions.csv) for the full recorded submission history. Public scores are historical records; they are not computed from the hidden labels by the local scripts.

## Reproduce the submission files

After setup and data preparation, run the following commands **in order**. This workflow trains models and regenerates the three files listed above, along with intermediate predictions and additional experimental submissions. Trained model checkpoints are not included, so a fresh checkout requires retraining.

```sh
# 1. Train recent-window LightGBM and CatBoost models.
.venv/bin/python src/exp_architectures.py trees

# 2. Train the original five-seed GraphSAGE and MLP models.
.venv/bin/python src/exp_architectures.py nn

# 3. Export the original architecture comparisons, including submission #8.
.venv/bin/python src/exp_architectures.py blend

# 4. Generate the extra early-cutoff tree predictions used by later comparisons.
#    This also trains the normalised-feature variants.
.venv/bin/python src/exp_smooth_norm.py trees

# 5. Generate the corresponding early-cutoff GraphSAGE predictions.
#    This also trains the normalised-feature GraphSAGE variants.
.venv/bin/python src/exp_smooth_norm.py nn

# 6. Train five additional GraphSAGE seeds, three-layer GraphSAGE, GAT and APPNP.
.venv/bin/python src/exp_gnn.py nn

# 7. Export graph-architecture comparison results and submissions.
.venv/bin/python src/exp_gnn.py score

# 8. Train the additional LightGBM seeds and export submission #18.
.venv/bin/python src/exp_final_blends.py

# 9. Train all later GraphSAGE variants, including local-feature GraphSAGE.
.venv/bin/python src/exp_gnn_more.py nn

# 10. Export the later blends, including submission #16.
.venv/bin/python src/exp_gnn_more.py score
```

The scripts create `outputs/` automatically. Intermediate arrays are written to `outputs/preds/`; scoring stages consume those arrays to write the submission CSVs. They are not standalone inference commands from an empty output directory. In particular, `exp_gnn_more.py score` expects predictions for **all** of its GraphSAGE variants, so the unfiltered `nn` command in step 9 is required for this workflow.

Several commands train extra variants because the existing scripts export complete experiment comparisons. CPU runtime and memory requirements depend on the machine. Use a fresh checkout for a clean reproduction: `exp_final_blends.py` reuses its additional-seed prediction arrays when they already exist, so stale caches can affect a rerun after changing data or parameters.

### Regenerate CSVs from existing prediction arrays

Once all the training steps above have completed and their prediction arrays are available, these commands regenerate the recorded CSV files without retraining:

```sh
.venv/bin/python src/exp_architectures.py blend
.venv/bin/python src/exp_final_blends.py
.venv/bin/python src/exp_gnn_more.py score
```

The middle command skips training only when all required `lgb_s1to4_*.npy` arrays already exist. Retain the complete `outputs/preds/` directory when using this cached workflow.

### Check the generated files

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

test = pd.read_parquet("data/test.parquet")
expected_index = np.sort(test["index"].to_numpy())
filenames = (
    "submission_more_lgb5_sage_local_1_1_w15.csv",
    "submission_final_lgb5_sage10_gat_2_1_1_w15.csv",
    "submission_arch_blend_lgb_catboost_w15.csv",
)
for filename in filenames:
    path = Path("outputs") / filename
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["index", "target"], path
    assert len(frame) == len(test), path
    assert np.array_equal(frame["index"].to_numpy(), expected_index), path
    assert np.isfinite(frame["target"]).all(), path
    assert frame["target"].between(0, 1).all(), path
    print(f"OK: {path} ({len(frame):,} rows)")
PY
```

Fixed seeds and pinned direct dependencies support repeatability. Retraining on different platforms or numerical-library builds may still produce floating-point differences. Regenerating a CSV from stored predictions checks the blending/export stage; it does not establish that a fresh installation and full retraining will be byte-identical to the original Kaggle upload.

## Validation and other experiments

- Initial expanding-window folds: train on steps 1–23 and validate on 24–29; then train on steps 1–29 and validate on 30–35.
- Recent-window models retain the last 15 steps before each training cutoff. The final training subset contains 12,346 labelled transactions.
- Later comparisons also evaluate a longer forecast horizon by training through step 17 and scoring labelled transactions in steps 18–35.
- Internal metrics include ROC-AUC and average precision. The F1 recorded by `common.metrics` selects a threshold on validation labels and is an exploratory best-threshold statistic.
- Earlier results generally report the mean of the two initial folds; later graph comparisons report fold 1 and the early-cutoff diagnostic separately. Check the result fields before comparing scores.

Additional scripts cover the baseline ladder (`src/run.py`), PageRank, recency weighting, pseudo-labelling, blend weights, validation diagnostics, graph smoothing and prior-shift adjustment. Historical details are in [docs/experiment_log.md](docs/experiment_log.md).

Generate the four exploratory figures with:

```sh
.venv/bin/python src/visualise.py
```

This writes PNGs to `outputs/figures/`. Report supplements are stored separately in [docs/figures/](docs/figures/).

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Data preparation, model definitions, experiment drivers and plotting |
| `docs/submissions.csv` | Recorded Kaggle submission IDs, methods and scores |
| `docs/experiment_log.md` | Historical experiment notes |
| `docs/figures/` | Supplementary figures |
| `requirements.txt` | Pinned direct Python dependencies |
| `data/` | User-supplied competition files and generated Parquet files; Git-ignored |
| `outputs/` | Generated predictions, metrics, submission CSVs and figures; Git-ignored |

The dataset, local virtual environment, generated outputs, Python caches and CatBoost working files are excluded by `.gitignore`.
