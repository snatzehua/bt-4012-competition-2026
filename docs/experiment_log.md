# BT4012 Competition 2026: experiment log

Log of all work on the Kaggle competition `bt-4012-competition-2026`, from 18 to 23 October 2026. It's written as source material for the final report, so it records what we tried, the numbers, and why each decision was made. That includes the ideas that failed.

- Submission history in machine-readable form: [`submissions.csv`](submissions.csv)
- Figures: [`figures/`](figures/)
- Raw validation results: `outputs/results*.json` (not in git; re-created by the scripts)

**Submitted vs not:** the log covers every experiment, including ones never submitted. Each phase in §5 starts with a **Submitted:** line, and in the results tables a submitted row carries its submission number, e.g. `#9`, matching §7 and `submissions.csv`. Rows without a number, or with "—" for Public LB, were never submitted; their scores are validation only.

---

## 0. Headline summary

| | Public LB score |
|---|---|
| Kaggle-provided baseline | 0.92119 |
| Our first baseline (LightGBM, all training steps) | 0.94067 |
| LightGBM trained on the last 15 time steps only | 0.94663 |
| LightGBM + GraphSAGE, rank-averaged 1:1, both on the last 15 steps | 0.95584 |
| **Our best: LightGBM (5 seeds) + GraphSAGE on the 93 local features only, rank-averaged 1:1** | **0.95938** |
| Leaderboard median, 18 Oct (71 teams) | 0.9527 |
| Leaderboard top, 19 Oct | 0.97039 |

**Main findings:**
1. **This is a concept-drift problem more than a modelling problem.** Our models score about 0.997 AUC when validated on past time steps but only 0.92–0.956 on the test period.
2. **Three changes produced real leaderboard gains:**
   - Training on recent data only (the last 15 steps): +0.006.
   - Blending LightGBM with a graph neural network (GraphSAGE): a further +0.009.
   - Giving GraphSAGE only the transaction's own 93 features, so it builds neighbourhood information itself instead of reading the dataset's precomputed summaries: a further +0.0035 (§5.13).
   Every other idea tied or lost, including GAT, deeper GraphSAGE, APPNP, domain-adversarial training and prior-shift correction.
3. **Local validation does not predict leaderboard results.**
   - Across 9 submissions, validation AUC and leaderboard score were unrelated (Spearman −0.23).
   - Rebuilding validation to predict further ahead raised that to 0.63. It then separated good tree models from bad ones, but still ranked the best submission 5th–6th of 9 (§5.9).
   - For graph models it failed outright: it ranked APPNP and the domain-adversarial GraphSAGE best, and both lost on the leaderboard (§5.11, §5.13).
4. **The graph helps only as a separate model.**
   - Graph features added to the tree model (multi-hop aggregates, PageRank) did nothing.
   - Smoothing predictions over the graph hurt (−0.014).
   - A GraphSAGE model blended with LightGBM gave our biggest gain.
   - Propagating predictions over many hops (APPNP) hurt badly (−0.024).
5. **What detects fraud:** mostly the transaction's own features (72% of the tree model's attention). The dataset's precomputed neighbour summaries account for 27%.

---

## 1. Competition and data

### 1.1 Task
- Binary classification: predict the probability that a Bitcoin transaction is illicit.
- Submission: `index,target` for each of the 15,329 test rows, where `target` is a probability.
- **Metric:** almost certainly ROC-AUC, inferred because scores are probability-ranked and fall in 0.92–0.97. The Evaluation page text was never retrieved, so confirm this for the report.
- **Submission limit:** 3 per day, resetting at 00:00 UTC (08:00 SGT).
- The data matches the structure of the public **Elliptic Bitcoin dataset** (Weber et al., 2019): 49 time steps and 165 anonymised features.

### 1.2 Files

| File | Rows | Content |
|---|---|---|
| `train.csv` | 141,772 | `txId`, `time_step` (1–35), `feat_1`…`feat_165`, `label` (1 = illicit, 0 = licit, blank = unknown) |
| `test.csv` | 15,329 | `index`, `txId`, `time_step` (36–49), `feat_1`…`feat_165` |
| `txs_edgelist.csv` | 234,355 | directed edges `txId1 → txId2` (flow of funds) |
| `sample_submission.csv` | 15,329 | `index,target` (all 0.5) |

### 1.3 Labels
- 31,235 labelled training rows: **27,591 licit and 3,644 illicit (11.7% illicit)**.
- 110,537 unlabelled training rows (78% of `train.csv`).
- The illicit share varies widely by time step: 0.4–2% in steps 1–6, then 2–36% (see `figures/2_timeline.png`).
- Features have no missing values. `index` in `test.csv` runs 0…15,328 in order.

### 1.4 Graph structure: an important quirk
- **Every edge joins two transactions in the same time step.** Time steps are separate graphs; each one we checked is a single connected component.
- Test-period edges sometimes point to transactions that aren't in either file:
  - 24,245 edges have one missing endpoint, and the endpoint that is present is always in steps 36–49.
  - 34,192 edges have both endpoints missing.
- **Interpretation:** in the test period, the organisers kept only the (formerly labelled) test transactions and removed all unlabelled ones. In training, unlabelled transactions are still present.
- **Consequence:** a test transaction's neighbourhood is systematically sparser than a training transaction's. Any neighbour-based feature computed naively means different things in train and test. Every graph feature we built is therefore computed on the **"visible" graph**: labelled training transactions plus test transactions. Degree counts use the full edge list, which is consistent in both periods.

### 1.5 Feature layout (from the Elliptic paper)
- `feat_1`–`feat_93`: "local" features of the transaction itself. In the original data these cover inputs/outputs, fees, volumes and similar.
- `feat_94`–`feat_165`: one-hop neighbour aggregates precomputed by the dataset authors.
- The features are anonymised and standardised, so we can't name what each one measures.

---

## 2. Setup and tooling

| Item | Detail |
|---|---|
| Environment | Python 3.12 virtualenv in `.venv/` (created with `uv`) |
| Libraries | pandas, numpy, scipy, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, PyTorch 2.14.1, networkx, matplotlib |
| Data access | Kaggle CLI 2.2.4 (installed with `pipx`), authenticated with `~/.kaggle/access_token` |
| Hardware | macOS laptop, 12 CPU cores, 32 GB RAM, CPU only |

**Access notes:**
- The competition page returns 404 to anonymous web requests because it's private.
- The Kaggle MCP connector authenticated but couldn't read private competition data (`competitions.get` denied), so we switched to the official CLI.
- The CLI provides data, submissions and the leaderboard, but not the Overview, Evaluation or Rules text.

**Technical issue:** on macOS, PyTorch and LightGBM/CatBoost in the same Python process crash (segfault, exit 139) or deadlock, because each ships its own OpenMP runtime. We fixed this by running tree models and neural models in separate processes that exchange saved predictions (`src/common.py`, `src/exp_architectures.py`).

---

## 3. Validation strategy

- **Time-based, expanding-window folds** (`src/common.py`):
  - Fold 0: train on steps 1–23, validate on 24–29.
  - Fold 1: train on steps 1–29, validate on 30–35.
- Only labelled rows are used, for both training and scoring.
- We report the mean ROC-AUC across the two folds, plus Average Precision (AP) and best-threshold F1.
- **No random splits:** test is strictly later in time, and a random split would leak future patterns.
- `time_step` itself is never used as a feature, because test values (36–49) are never seen in training.
- The number of boosting rounds is fixed rather than early-stopped on the validation folds, to avoid tuning to them.

**How well validation worked:** it was saturated (≈0.995–0.998 for all reasonable models) and turned out to be a poor stand-in for the test period. See §6 and §8.

---

## 4. Exploratory analysis and figures

Generated by `src/visualise.py`; the PNGs are in `docs/figures/`.

| Figure | What it shows | Key numbers |
|---|---|---|
| `1_network_snapshot.png` | All 1,653 transactions of step 28 (one connected graph), plus a 3-hop zoom (107 transactions) around an illicit hub | Long chains of unlabelled transactions; illicit transactions sit in tight clusters around shared hub transactions |
| `2_timeline.png` | Labelled licit/illicit counts per step, test counts per step, illicit share over time | Illicit share 0.4–36%, very volatile; test steps have 471–2,154 transactions each; Elliptic's dark-market shutdown at step 43 is marked |
| `3_neighbourhood.png` | Who each class links to, and degree distribution by class | Illicit tx: 23.4% of links go to illicit, 17.4% to licit, 59.2% to unknown. Licit tx: 1.5% to illicit, 53.0% to licit, 45.5% to unknown. Median degree: licit 2, illicit 1, unknown 2; a small group of illicit hubs have 30–150 links |
| `4_drift_features.png` | Median and IQR per step of the four features that best separate train from test | They increase steadily with time (see §5.3) |

**Main EDA conclusions:**
1. Fraud is strongly **homophilous**: illicit transactions are about 15× more likely than licit ones to link to an illicit transaction. So neighbour information carries signal.
2. Strong concept drift is visible even inside the training period (§6.1).

---

## 5. Experiments, in chronological order

All model settings are in `src/models.py` and `src/nn_models.py`. The default LightGBM: `learning_rate=0.03, n_estimators=700, num_leaves=31, min_child_samples=30, colsample_bytree=0.5, subsample=0.8, reg_lambda=1.0`.

### 5.1 Phase 1: baseline and first experiment ladder (`src/run.py`)

**Submitted:** #1 (baseline), #2 (Random Forest)

**Goal:** establish a baseline, then test graph features, drift-based feature removal and ensembling.

**Graph features added** (`src/features.py → graph_features`, 377 extra columns, 542 in total):
- Degree in, out and total (full edge list), plus degree in and out on the visible graph.
- Mean of the 93 local features over out-neighbours and over in-neighbours.
- Mean over 2-hop neighbours (undirected).
- Node minus 1-hop neighbour mean (contrast with the neighbourhood).

**Drift removal:** we trained a classifier to tell training rows from test rows (adversarial validation), ranked features by how much it relied on them, and dropped the top 30.

| Experiment | Val AUC | AP | F1 | Fold AUC (0 / 1) |
|---|---|---|---|---|
| **1. Baseline: LightGBM, 165 raw features** `#1` | **0.9966** | 0.9867 | **0.9558** | 0.9951 / 0.9982 |
| 1b. Random Forest, raw features `#2` | 0.9925 | 0.9808 | 0.9523 | 0.9924 / 0.9926 |
| 2. LightGBM + graph features | 0.9963 | **0.9871** | 0.9533 | 0.9943 / 0.9984 |
| 3. As 2, minus the 30 most-drifting features | 0.9950 | 0.9816 | 0.9384 | 0.9917 / 0.9983 |
| 4. Ensemble: LightGBM + XGBoost + RF × 3 seeds, graph features, rank-averaged | 0.9953 | 0.9848 | 0.9487 | 0.9934 / 0.9972 |

**Leaderboard:** the baseline scored **0.94067**, about 67th of 71 teams. Leaderboard median 0.9527, top 0.9703, Kaggle baseline 0.92119 (the lowest entry).

**Conclusions:**
- Nothing beat the baseline on validation.
- The 0.997 → 0.941 gap between validation and leaderboard was the first evidence of strong drift.
- The baseline's mean predicted fraud probability on test was 4.2%, against 11.7% in training. Test either has less fraud or looks unfamiliar to the model.

### 5.2 Phase 2: PageRank features (`src/features.py → pagerank`, `src/exp_pagerank.py`)

**Submitted:** none

**Hypothesis:** how central a transaction is in the money-flow graph says something about fraud.

**Implementation:**
- Our own PageRank, by power iteration: damping 0.85, tolerance 1e-10, dangling nodes redistribute uniformly. Verified against `networkx.pagerank` on step 28: maximum absolute difference 1.2e-10.
- Computed **separately for each time step, on the visible graph**, in both directions: forward follows the money (rewards receivers), reverse rewards senders.
- Each score is reported as log(n × score) and as a percentile within its step, because raw PageRank scales with graph size.
- 4 new features.

| Variant | Val AUC | AP | F1 |
|---|---|---|---|
| Baseline `#1` | **0.9966** | **0.9867** | **0.9558** |
| + PageRank | 0.9957 | 0.9828 | 0.9425 |
| − time-like features (`feat_100`, `101`, `103`) | 0.9963 | 0.9850 | 0.9531 |
| − time-like features + PageRank | 0.9961 | 0.9842 | 0.9525 |

**On their own, the PageRank features are weak predictors:**
- Single-feature AUC 0.56–0.61. Higher centrality means slightly *less* likely to be fraud; the raw scores give 0.39–0.44.
- The model ranks them 17th, 43rd, 58th and 72nd of 169 features by importance.

**Conclusion:** PageRank adds no measurable value. Fraud here is a *local* clustering pattern, which the neighbour aggregates already capture. Global centrality adds little.

Two variants we rejected without testing:
- PageRank personalised from known fraud: test transactions only link to other test transactions, whose labels are hidden, so there's nothing to seed it from.
- Naive PageRank on the full graph: test-period graphs are pruned (§1.4), so the scores would mean different things in train and test.

### 5.3 Phase 3: time-proxy features (finding)

**Submitted:** none (analysis)

The adversarial classifier from §5.1 mostly used `feat_101`, `feat_103`, `feat_136` and `feat_100`. Plotting them over time (figure 4) shows they **rise in a straight line with `time_step`.**

How much of each feature's variation the time step explains (R²):

| Feature | R² explained by `time_step` |
|---|---|
| `feat_101` | 0.989 (correlation with time step 0.994) |
| `feat_103` | 0.984 |
| `feat_100` | 0.953 |
| `feat_136` | 0.776 |
| `feat_139` | 0.727 |

- **The concern:** tree models can use these as a hidden clock and learn per-step fraud rates that don't carry into the future.
- **The check:** in §5.4, the model gives these features only **1.1% of its attention**, so they are *not* the main cause of the drift.
- They're kept as `TIME_PROXIES` in `features.py`.
- We built a submission without them (`submission_no_time_proxies.csv`) but never submitted it.

### 5.4 Phase 4: what actually detects fraud (model explanation)

**Submitted:** none (analysis)

We computed SHAP values (LightGBM `pred_contrib`) for the baseline model trained on all labelled rows.

**Share of mean |SHAP| by feature group:**

| Group | Share |
|---|---|
| Local features (`feat_1`–`93`) | **72.2%** |
| Precomputed neighbour aggregates (`feat_94`–`165`) | 26.7% |
| Time-like features (`feat_100`, `101`, `103`) | 1.1% |

**Top individual features:**

| Feature | Share of attention | AUC on its own | Fraud tends to be | Median: fraud vs licit |
|---|---|---|---|---|
| `feat_53` | 7.3% | 0.857 | lower | −0.488 vs 0.770 |
| `feat_59` | 6.1% | 0.626 | lower | same median; differs in the tails |
| `feat_2` | 4.4% | 0.536 | higher | same median; differs in the tails |
| `feat_90` | 3.6% | 0.837 | lower | −0.694 vs 0.067 |
| `feat_61` | 3.2% | 0.595 | lower | same median |
| `feat_55` | 2.9% | 0.840 | lower | −0.468 vs 0.552 |
| `feat_89` | 2.2% | 0.794 | lower | −0.761 vs 0.147 |
| `feat_5` | 1.9% | 0.817 | lower | −0.064 vs −0.044 |

**Concentration:**
- The top 10 features carry 37% of the attention; the top 30 carry 65%.
- A model using only the top 5 features scores 0.9775 validation AUC; top 10, 0.9882; top 20, 0.9953. The full 165 score 0.9966.

**Conclusion:** fraud is identified mainly by a consistent profile in a small group of local features: low `feat_53`, `55`, `89` and `90`. The features are anonymised, so this can't be translated into named transaction properties.

### 5.5 Phase 5: robustness ideas: cautious models, recency, blending (`src/exp_robust.py`)

**Submitted:** #2 (Random Forest), #3 (last 15 steps)

**Diagnostic first: are the main signals fading over time?** Single-feature AUC of the top features, averaged over blocks of time steps:

| Steps | `feat_53` | `feat_55` | `feat_90` | `feat_5` | `feat_89` |
|---|---|---|---|---|---|
| 1–7 | 0.658 | 0.650 | 0.765 | 0.717 | 0.719 |
| 8–14 | 0.893 | 0.892 | 0.882 | 0.829 | 0.840 |
| 15–21 | 0.891 | 0.891 | 0.888 | 0.859 | 0.861 |
| 22–28 | 0.884 | 0.878 | 0.817 | 0.771 | 0.810 |
| 29–35 | **0.815** | **0.798** | 0.804 | 0.789 | **0.748** |

→ The model's main signals weaken by 0.06–0.11 AUC between steps 15–21 and 29–35. **The way fraud shows up in the features is changing during the training period itself**, and the test period probably continues the trend. This motivated training on recent data.

**Experiments:**
- **Regularised LightGBM:** `num_leaves=8, max_depth=4, min_child_samples=200, colsample_bytree=0.3, reg_lambda=10, lr=0.02, 800 trees`.
- **Recency:**
  - 4a: train only on the last 15 steps.
  - 4b and 4c: train on all steps, weighting each row by 0.5^((last step − step) / half-life).

| Idea | Val AUC | AP | F1 | Test rank correlation with baseline | **Public LB** |
|---|---|---|---|---|---|
| Baseline LightGBM `#1` | 0.9966 | 0.9867 | 0.9558 | 1.000 | 0.94067 |
| 3a. Regularised LightGBM | 0.9857 | 0.9477 | 0.8867 | 0.928 | — |
| 3b. Random Forest `#2` | 0.9925 | 0.9808 | 0.9523 | **0.710** | **0.92120** |
| **4a. Last 15 steps only** `#3` | **0.9967** | **0.9874** | 0.9546 | 0.905 | **0.94663** |
| 4b. Recency weighting, half-life 10 | 0.9966 | 0.9869 | 0.9552 | 0.969 | — |
| 4c. Recency weighting, half-life 5 | 0.9966 | 0.9873 | 0.9603 | 0.917 | — |
| 5. Blend: LightGBM + regularised LightGBM + RF | 0.9948 | 0.9808 | 0.9425 | 0.953 | — |

**Conclusions:**
- **Recency (4a) gave our best leaderboard score: +0.006 over the baseline**, even though validation showed no difference.
- Random Forest scored 0.92120, almost identical to Kaggle's baseline (0.92119). **The provided baseline is very likely a Random Forest.**
- The regularised LightGBM was too constrained (fold 0 fell to 0.9737).

### 5.6 Phase 6: training-window size and self-training (`src/exp_window_selftrain.py`)

**Submitted:** #4 (self-training), #5 (window 10), #6 (window 20)

**Window sweep** (LightGBM on raw features, trained on the last *W* labelled steps):

| Window *W* | Val AUC | AP | Rank correlation with W=15 on test | **Public LB** |
|---|---|---|---|---|
| 5 | 0.9947 | 0.9814 | 0.813 | — |
| 8 | 0.9957 | 0.9843 | 0.917 | — |
| 10 `#5` | 0.9959 | 0.9841 | 0.942 | **0.93204** |
| 12 | 0.9958 | 0.9839 | 0.959 | — |
| **15** `#3` | **0.9967** | **0.9874** | 1.000 | **0.94663** |
| 20 `#6` | 0.9967 | 0.9869 | 0.965 | **0.94296** |
| all 35 `#1` | 0.9966 | 0.9867 | — | 0.94067 |

→ The leaderboard peaks at about 15 steps: 35 → 0.9407, 20 → 0.9430, **15 → 0.9466**, 10 → 0.9320. Shorter windows likely have too little data, and longer ones bring back stale patterns.

Caveats:
- Each score comes from a single model with one random seed, so differences of about 0.004 may be partly noise.
- Tuning the window precisely to the public leaderboard risks overfitting the public portion of the test set.

**Self-training on test (pseudo-labelling)** (function `self_train`):
1. Train on the last *W* steps.
2. Predict the target rows.
3. Label rows with p ≥ `hi` as fraud and rows with p ≤ `lo` as licit. Add them to training at weight 0.5.
4. Retrain. Repeat for 2 rounds.

In validation, the "target" is the held-out window with its labels hidden, which simulates the test situation exactly.

| Variant | Val AUC | AP | Fold AUC (0 / 1) | **Public LB** |
|---|---|---|---|---|
| W=15, hi=0.9, lo=0.02 `#4` | **0.9977** | **0.9920** | 0.9974 / 0.9981 | **0.93033** |
| W=15, hi=0.97, lo=0.005 | 0.9962 | 0.9849 | 0.9944 / 0.9981 | — |
| W=10, hi=0.9, lo=0.02 | 0.9971 | 0.9887 | 0.9960 / 0.9981 | — |
| W=10, hi=0.97, lo=0.005 | 0.9944 | 0.9778 | 0.9908 / 0.9980 | — |

**Step-by-step trace on fold 0** (train steps 9–23: 11,762 rows, 1,824 fraud; target steps 24–29: 3,901 rows, 789 fraud):

| Stage | Pseudo-labelled as fraud (wrong) | Pseudo-labelled as licit (wrong) | Left out (of which fraud) | AUC |
|---|---|---|---|---|
| Plain model | — | — | — | 0.9956 |
| Round 1 | 434 (2) | 2,935 (13) | 532 (344) | 0.9967 |
| Round 2 | 525 (3) | 3,076 (28) | 300 (239) | 0.9974 |

- On fold 1 (target steps 30–35) the gain was smaller: 0.9978 → 0.9981.
- **In validation, the pseudo-labels were about 99.5% accurate for fraud and about 99% for licit.** The uncertain rows were mostly fraud, which is the drift in action, and confident fraud pseudo-labels taught the model the new period's patterns.
- On test, round 1 labelled 14,914 rows (97%), of which 534 as fraud. Round 2 labelled 15,234 rows, of which 549 as fraud.

**Leaderboard result: 0.93033, −0.016 vs the plain 15-step model.**

**Conclusion:** the test shift is much larger than the shift inside training. On test, the model's confident predictions are more often wrong, and self-training reinforces those errors (confirmation bias). This is the clearest example of validation pointing the wrong way.

### 5.7 Phase 7: new architectures (`src/exp_architectures.py`, `src/nn_models.py`)

**Submitted:** #7 (CatBoost), #8 (LightGBM + CatBoost), #9 (LightGBM + GraphSAGE)

All trained on the last 15 labelled steps, the same setting as our best submission.

**CatBoost:** `iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3`.

**MLP:**
- Architecture: 165 → 256 → 128 → 1, with BatchNorm, ReLU and dropout 0.3.
- Features standardised using statistics from the training rows, then clipped to ±5.

**GraphSAGE:** written directly in PyTorch, without a graph library.
- An input layer, then two SAGE layers. Each layer computes h′ = W_self·h + W_out·mean(out-neighbours) + W_in·mean(in-neighbours), followed by LayerNorm, ReLU and dropout 0.3.
- The output layer reads the input, 1-hop and 2-hop representations together (skip connections).
- Uses the visible graph (§1.4).
- Training: full-batch, loss only on the training rows, AdamW (lr 1e-3, weight decay 1e-4), 150 epochs.

Both neural models are averaged over 5 seeds.

| Model | Val AUC | AP | F1 | Fold AUC (0 / 1) | Test rank correlation with LightGBM | Predicted fraud share on test |
|---|---|---|---|---|---|---|
| LightGBM (LB 0.94663) `#3` | **0.9967** | **0.9874** | 0.9546 | 0.9956 / 0.9978 | 1.000 | 3.9% |
| CatBoost `#7` | 0.9948 | 0.9801 | 0.9363 | 0.9911 / **0.9986** | 0.854 | 4.0% |
| MLP | 0.9452 | 0.8470 | 0.7649 | 0.9100 / 0.9803 | 0.418 | 4.2% |
| GraphSAGE | 0.9655 | 0.8721 | 0.7983 | 0.9512 / 0.9799 | 0.501 | 3.0% |
| Blend: LightGBM + CatBoost `#8` | 0.9965 | 0.9862 | 0.9494 | 0.9946 / 0.9984 | — | — |
| Blend: LightGBM + MLP | 0.9845 | 0.9536 | 0.8909 | 0.9727 / 0.9964 | — | — |
| Blend: LightGBM + GraphSAGE `#9` | 0.9922 | 0.9728 | 0.9193 | 0.9878 / 0.9967 | — | — |
| Blend: all four | 0.9897 | 0.9655 | 0.9035 | 0.9821 / 0.9973 | — | — |

(Blends average the models' rankings, not their probabilities.)

**Conclusions from validation:**
- CatBoost is competitive, and the best of all models on the later window.
- The neural models trail by 0.03–0.05 AUC. They're likely under-trained: only 150 full-batch steps, with untuned settings.
- GraphSAGE beats the MLP by 0.02, so learned message passing adds signal over features alone.
- The neural models rank test very differently from the tree models (correlation 0.42–0.50).

**Leaderboard results (submitted 20 Oct):**

| Submission | Val AUC | Public LB |
|---|---|---|
| CatBoost alone `#7` | 0.9948 | 0.94350 |
| LightGBM + CatBoost `#8` | 0.9965 | 0.94736 |
| **LightGBM + GraphSAGE** `#9` | 0.9922 | **0.95584** |

- **Validation ranked these three in exactly the reverse order of the leaderboard.**
- The CatBoost blend's +0.0007 over LightGBM alone is within noise.
- The GraphSAGE blend's +0.009 is the largest single gain of the project. Validation can't see it.

**Why GraphSAGE helps (interpretation):**
- GraphSAGE is weaker alone, but it ranks test transactions very differently from LightGBM (correlation 0.50).
- On the drifted test period, the two models' errors partly cancel.
- One plausible mechanism: fraud rings stay connected even when their feature values drift, so a model that reads neighbourhoods is hurt less by drift.

### 5.8 Phase 8: blend weighting (`src/exp_weighted_blend.py`)

**Submitted:** #10 (probability average), #11 (1:1:2)

All blends use the saved 15-step predictions; nothing was retrained.

**Fixed weights:** seven rank-weighted mixes of LightGBM, CatBoost and GraphSAGE.
- Validation AUC fell steadily as GraphSAGE's weight rose: 1:0:1 scored 0.9922, 1:0:2 scored 0.9870, 1:0:3 scored 0.9833.
- Every mix ranked test 96–99% the same as the 1:1 blend.

**Learned / dynamic weights (analysis, not submitted):**
- **Stacking:** any weights fitted on validation would favour LightGBM, which beats GraphSAGE on both folds. That would rebuild the 0.9466 submission.
- **Per-node weights by number of neighbours:** LightGBM beats GraphSAGE on validation in every degree group (0, 1, 2–3 and 4+ neighbours). On test, the two models disagree about equally in every group (rank correlation 0.52–0.68), so there's no group to target.
- **Confidence weighting**, using |p − 0.5| or 1 − entropy:
  - Both models are near-certain on almost all test nodes (mean confidence 0.99), so weighting rankings by confidence reproduces the 1:1 blend exactly (correlation 1.000).
  - LightGBM is the more confident model on 85% of test nodes, so confidence weighting would lean back toward LightGBM.
  - A model's own confidence can't flag drift errors it doesn't know it's making.
- **A side finding:** averaging probabilities instead of rankings changes the blend noticeably (0.944 correlation with the rank blend). LightGBM's near-zero scores for legitimate transactions stop dominating the ordering.

**Leaderboard (21 Oct):**

| Submission | Public LB |
|---|---|
| LightGBM + GraphSAGE 1:1, rankings averaged (reference) `#9` | **0.95584** |
| LightGBM : CatBoost : GraphSAGE 1:1:2, rankings averaged `#11` | 0.95518 |
| LightGBM + GraphSAGE 1:1, probabilities averaged `#10` | 0.95507 |

**Conclusion:** blend re-weighting is exhausted. All three are tied within noise. CatBoost adds nothing once GraphSAGE is in the blend.

### 5.9 Phase 9: rebuilding validation to track the leaderboard (`src/exp_resample_val.py`)

**Submitted:** none (analysis)

**Goal:** a validation scheme whose scores rank our 9 submitted models in leaderboard order.

**Method:**
- Each submitted model was re-run from 8 training cutoffs (steps 17, 19, …, 31), each with its own recency window, predicting every labelled node after the cutoff.
- 40 scoring schemes were then compared.

**Schemes tried:**
- different cutoffs and horizons, including a 14-step horizon to match the test period
- scoring only far-ahead steps
- adversarial importance weighting and test-like subsets
- per-step averaged AUC
- low-fraud steps only

**Success measures:**
- Spearman rank correlation with the leaderboard (primary)
- Kendall's tau and Pearson correlation
- **Pair agreement:** the share of the 31 model pairs separated by more than 0.003 on the leaderboard that validation orders the same way
- leave-one-out minimum Spearman
- a permutation test for best-of-40 selection

| Scheme | Spearman | Pair agreement |
|---|---|---|
| Original folds (23→24–29, 29→30–35) | 0.08 | 61% |
| **Train ≤ 17, score steps 18–35** | **0.63** | **77%** |
| Score only ≥ 10 steps after each cutoff | 0.60 | 77% |
| Per-step mean AUC, all cutoffs | 0.52 | 74% |
| Best adversarial weighting | 0.50 | 71% |

**Findings:**
- **Predicting far ahead is what matters.** It brings validation AUCs (0.84–0.91) into the leaderboard's range and correctly identifies the bad models: Random Forest, self-training and the 10-step window.
- **Adversarial weighting failed.** A classifier separates train from test perfectly (AUC 1.000), even without the three time-proxy features, so the weights degenerate. Dropping the most period-revealing features until separation fell to 0.79–0.89 still only reached Spearman 0.50.
- **Not statistically convincing.** With the leaderboard scores shuffled, the best of 40 schemes reaches Spearman 0.52 on average, and at least 0.63 31% of the time.
- **Breakdown for the best scheme:**
  - good vs bad models: 18/18 correct
  - good vs good: 6/11 correct, a coin flip
  - every comparison between the GraphSAGE blend and another good model: wrong (0/5)
- **Every scheme penalises GraphSAGE's weight steadily.** It ranks LightGBM alone above the 0.95584 blend, which the leaderboard contradicts.

**Use going forward:** a filter that rejects clearly bad ideas, with the leaderboard deciding among good ones. It's useless for GraphSAGE weighting.

*Correction, 22–23 Oct:* even the filter role doesn't hold for graph models. Train ≤ 17 ranked APPNP and the domain-adversarial GraphSAGE best, and both lost on the leaderboard (§5.11, §5.13).

### 5.10 Phase 10: graph smoothing and per-step normalisation (`src/exp_smooth_norm.py`)

**Submitted:** #12 (smoothed α = 0.3)

**Motivation:** fraud clusters strongly in the graph. An illicit node's labelled neighbour is illicit 57% of the time, against 2.8% for a licit node (base rate 11.7%).

**Smoothing** (label-propagation style post-processing):
- s ← (1 − α)·s₀ + α·mean(neighbours' s), repeated k times, over the test graph.
- Tried α ∈ {0.2, 0.3, 0.5, 0.7} and k ∈ {1, 3, 10}, on both LightGBM and the LightGBM + GraphSAGE blend.

**Per-step normalisation:** every feature replaced by its percentile rank within its time step, either alone or alongside the raw features. LightGBM and GraphSAGE were retrained on it.

| Variant | Original folds | Train ≤ 17 | Public LB |
|---|---|---|---|
| LightGBM + GraphSAGE (reference) `#9` | 0.9922 | 0.8846 | **0.95584** |
| Same blend, smoothed α = 0.2, k = 3 | 0.9907 | 0.8815 | — |
| Same blend, smoothed α = 0.3, k = 3 `#12` | 0.9872 | 0.8784 | 0.94175 |
| Same blend, smoothed α = 0.7, k = 3 | 0.9594 | 0.8593 | — |
| LightGBM, normalised features only | 0.8638 | 0.8688 | — |
| LightGBM, raw + normalised features | 0.9858 | 0.8967 | — |
| Both models normalised, blended | 0.8731 | 0.8604 | — |

**Conclusions:**
- **Smoothing hurts, and more the stronger it is.** On the leaderboard it cost 0.014, about three times what validation predicted. The models already use neighbour information, through the dataset's aggregates and GraphSAGE, so smoothing double-counts it. Fraud nodes also have many legitimate neighbours (43%), which get pulled upward.
- **Validation called the direction correctly here.** It catches harmful changes; its blind spot is specifically the gain from adding GraphSAGE as a separate model.
- **Per-step normalisation destroys signal (−0.04 to −0.13 everywhere).** Fraud rates vary from 0.4% to 36% between steps, and per-step ranks erase that.

### 5.11 Phase 11: other graph architectures (`src/exp_gnn.py`, `src/nn_models.py`)

**Submitted:** #13 (+ GAT), #14 (2:1:1), #15 (+ APPNP)

**Motivation:** GraphSAGE was the only idea that helped, so we tried other graph neural networks as blend partners for LightGBM. All train on the last 15 steps, 5 seeds averaged.

**Models:**
- **sage10:** the existing GraphSAGE with 5 more seeds (10 in total), to reduce seed noise.
- **sage3:** GraphSAGE with a third layer, so each node sees 3-hop neighbourhoods.
- **GAT:** graph attention. Like GraphSAGE, but each neighbour's message is weighted by a learned attention score (4 heads, separate for in- and out-edges).
- **APPNP:** an MLP's per-node scores, propagated 10 hops over the graph (z ← 0.9·P z + 0.1·z₀).

**Bug found and fixed in APPNP:** the first version had no self-loops in P. About a quarter of nodes have no neighbours, so their score shrank to 0.1·z₀ and they all collapsed to the middle of the ranking. Adding self-loops raised APPNP alone from 0.7889 to 0.8378 on train ≤ 17.

Validation scores (fold 1 = train ≤ 29, score 30–35; train ≤ 17 = score 18–35; §5.9) and the correlation of test rankings with the LightGBM + GraphSAGE blend:

| Variant | Fold 1 | Train ≤ 17 | Test corr. | Public LB |
|---|---|---|---|---|
| LightGBM alone | 0.9978 | 0.9117 | 0.86 | — |
| GraphSAGE (5 seeds) alone | 0.9799 | 0.8503 | 0.86 | — |
| GAT alone | 0.9835 | 0.8597 | 0.85 | — |
| sage3 alone | 0.9814 | 0.8492 | 0.72 | — |
| APPNP alone | 0.9517 | 0.8378 | 0.25 | — |
| LightGBM + GraphSAGE 1:1 (reference) `#9` | 0.9967 | 0.8846 | 1.00 | 0.95584 |
| LightGBM + sage10 1:1 | 0.9969 | 0.8859 | 1.00 | — |
| LightGBM + sage3 1:1 | 0.9968 | 0.8835 | 0.97 | — |
| LightGBM + GAT 1:1 `#13` | 0.9971 | 0.8893 | 0.99 | 0.95573 |
| **LightGBM : sage10 : GAT 2:1:1** `#14` | 0.9971 | 0.8876 | 1.00 | **0.95627** |
| LightGBM + APPNP 1:1 `#15` | 0.9913 | **0.9016** | 0.70 | 0.93173 |

**Conclusions:**
- **GAT is as good as GraphSAGE but not better.** The 2:1:1 blend with both was a new best by +0.0004, within noise.
- **APPNP was a clear failure (−0.024),** although train ≤ 17 ranked it best of all. Spreading scores 10 hops over the graph is a stronger form of the smoothing that hurt in §5.10.
- **The rebuilt validation can't be trusted for graph models.** This corrects §5.9's claim that it at least filters out bad ideas: for graph models, it rated the worst one highest.

### 5.12 Phase 12: per-step fraud-rate (prior-shift) correction (`src/exp_prior_shift.py`)

**Submitted:** none

**Motivation:** AUC is pooled across all test steps. The models predict near-zero fraud after step 43 (consistent with the dark-market shutdown), but in a rank blend those steps' nodes still sit mid-ranking. Correcting each step's scores to its own fraud rate could fix the pooled ranking.

**Method** (Saerens et al. 2002): estimate each step's fraud rate with EM from the models' own probabilities, re-weight each model's probabilities from the training rate to that step's rate with Bayes' rule, then rank-blend as before. `strength` ∈ {0.25, 0.5, 0.75, 1} interpolates between no change and the full estimate. No test labels are used. An **oracle** variant uses the true step rates, on validation only, to show the ceiling.

| Variant (on LightGBM : sage10 : GAT 2:1:1) | Fold 1 | Train ≤ 17 |
|---|---|---|
| No adjustment | **0.9971** | 0.8876 |
| Strength 0.25 | 0.9967 | 0.8860 |
| Strength 0.5 | 0.9963 | 0.8844 |
| Strength 1.0 | 0.9953 | 0.8790 |
| Oracle (true step rates) | 0.9952 | **0.9090** |

**Conclusions:**
- The idea is right in principle: knowing the true per-step rates would add +0.021 on train ≤ 17.
- **The EM estimates aren't good enough.** They underestimate fraud in exactly the drifted steps where the models are wrong, so every strength made things worse. Not submitted.

### 5.13 Phase 13: GraphSAGE variants and stability (`src/exp_gnn_more.py`, `src/exp_final_blends.py`)

**Submitted:** #16 (+ sage_local), #17 (+ sage_dann), #18 (5-seed 2:1:1)

**Motivation:** find a GraphSAGE that ranks test differently from LightGBM, since diversity is what made the blend work. Each variant changes one thing; LightGBM is now averaged over 5 seeds (**lgb5**).

- **sage_all, sage_w20, sage_w10:** training window of all steps, 20 or 10 steps (instead of 15).
- **sage_local:** only the 93 local features (the transaction's own); the network does all neighbour aggregation itself instead of also reading the dataset's 72 precomputed neighbour summaries.
- **sage_teststd:** the prediction period standardised with its own mean and spread (simple test-time adaptation).
- **sage_dann:** domain-adversarial GraphSAGE. A second head tries to tell training-period nodes from prediction-period nodes from the embedding, and its reversed gradient pushes the embedding to look the same in both (Ganin et al. 2016).

| Variant | Fold 1 | Train ≤ 17 | Test corr. vs sage10 | Public LB |
|---|---|---|---|---|
| sage_all alone | 0.9774 | 0.8502 | 0.92 | — |
| sage_w20 alone | 0.9805 | 0.8502 | 0.96 | — |
| sage_w10 alone | 0.9821 | 0.8527 | 0.95 | — |
| **sage_local alone** | **0.9919** | 0.8571 | **0.74** | — |
| sage_teststd alone | 0.8991 | 0.8459 | 0.95 | — |
| sage_dann alone | 0.9841 | 0.8669 | 0.78 | — |
| **lgb5 + sage_local 1:1** `#16` | **0.9977** | 0.8841 | — | **0.95938** |
| lgb5 + sage_dann 1:1 `#17` | 0.9962 | **0.8921** | — | 0.94863 |
| lgb5 : sage10 : GAT 2:1:1 `#18` | 0.9971 | 0.8846 | — | 0.95641 |
| lgb5 : sage10 : GAT : sage_local 3:1:1:1 | 0.9977 | 0.8848 | — | — |

**Conclusions:**
- **Local features only gave a new best: 0.95938, +0.0035** over the same blend with all features. It's the second-largest gain of the project. It's the most accurate GraphSAGE alone and also the most different from the others (test correlation 0.74). Likely reasons: the precomputed summaries drift with everything else, and LightGBM already relies on them heavily, so a graph model that builds its own neighbourhood view adds a genuinely different signal.
- **Domain-adversarial training hurt (−0.007),** despite the best train ≤ 17 score. Forcing train and test to look alike threw away real signal along with the drift. It's the second graph model (after APPNP) that train ≤ 17 rated best and the leaderboard rejected.
- **Window variants are near-copies of sage10 (0.92–0.96 correlation),** so they add little to a blend.
- **Test-period standardisation broke the model** (fold 1 0.899).
- **5-seed LightGBM tied the single seed** (0.95641 vs 0.95627), as expected. It's the more stable choice for final selection.
- **Fold 1 picked the winner this time; train ≤ 17 didn't.** We stopped using train ≤ 17 to judge graph models.

---

## 6. Key findings for the report

### 6.1 Concept drift is the central challenge
Evidence:
- The validation-to-leaderboard gap (0.997 → 0.93–0.947).
- The top features lose discriminative power over time (§5.5 table).
- Predicted fraud share on test is about 3–4%, against 11.7% in training.
- The original Elliptic dataset has a known dark-market shutdown around step 43, inside our test period.

### 6.2 Recency beats volume
Training on the last 15 of 35 steps (12,346 labelled rows instead of 31,235) improved the public score. Old fraud patterns actively hurt.

### 6.3 Most added complexity didn't pay off under drift
None of these beat a plain LightGBM trained on recent data:
- extra graph features (377 columns)
- PageRank
- drift-based feature removal
- the XGBoost/RF ensemble
- Random Forest
- regularised boosting
- self-training
- CatBoost
- graph smoothing
- per-step normalisation
- deeper GraphSAGE, GAT (a tie), APPNP
- domain-adversarial training, test-period standardisation
- per-step fraud-rate correction

The exception was **adding a different kind of model: GraphSAGE in a blend**, and then **making that model more different** by giving it only local features.

### 6.4 Validation becomes unreliable under strong drift
Our time-based validation was saturated, and its drift was much milder than test's. Examples:
- Recency: no validation gain, +0.006 on the leaderboard.
- Window 10: −0.0008 in validation, −0.0146 on the leaderboard.
- Self-training: +0.001 in validation (the best score), −0.016 on the leaderboard (the second-worst).
- LightGBM + GraphSAGE: −0.0045 in validation (6th of 9), +0.009 on the leaderboard (the best).

Across the 9 submitted models, Spearman correlation between validation and leaderboard was −0.23.

Validating further ahead (train ≤ 17, score steps 18–35) raises it to 0.63 and gets every good-vs-bad comparison right. It's still a coin flip among good models and always wrong about GraphSAGE (§5.9).

For graph models it was worse than useless: train ≤ 17 ranked APPNP (0.9016) and the domain-adversarial GraphSAGE (0.8921) above everything, and they were two of the worst graph submissions (0.93173, 0.94863). The original fold 1 did pick the local-features GraphSAGE, our best.

Lesson: under heavy drift, local validation can screen out clearly bad tree models, but it can't rank good ones, and for graph models it can point the wrong way.

### 6.5 Graph information
- Fraud is strongly homophilous (§4, §5.10), but the dataset's precomputed one-hop aggregates already capture much of that for the tree model.
- Graph information fed *into* the tree model (multi-hop aggregates, PageRank) added nothing.
- Graph information applied *after* it (smoothing) hurt.
- Graph information as a *separate learned model* (GraphSAGE) blended with it gave the largest gain of the project. Its value appears only under the test period's drift, not in validation.
- **Shallow, learned aggregation works; propagation doesn't.** Two-layer GraphSAGE and GAT helped; a third layer added nothing; 10-hop APPNP and smoothing hurt.
- **The graph model works best on local features only** (0.95938). Letting it build neighbourhood information from raw transaction features, rather than the dataset's precomputed summaries, makes it both stronger and more different from LightGBM.
- The pruned test graph (§1.4) is a data-design constraint that any graph method must handle consistently.

---

## 7. Submission history

Full CSV: [`submissions.csv`](submissions.csv).

| # | Date (UTC) | Approach | Val AUC | Public LB |
|---|---|---|---|---|
| 1 | 18 Oct 11:07 | LightGBM, all steps (baseline) | 0.9966 | 0.94067 |
| 2 | 18 Oct 16:31 | Random Forest, all steps | 0.9925 | 0.92120 |
| 3 | 18 Oct 16:31 | **LightGBM, last 15 steps** | 0.9967 | **0.94663** |
| 4 | 19 Oct 01:35 | LightGBM, last 15 steps + self-training | 0.9977 | 0.93033 |
| 5 | 19 Oct 01:35 | LightGBM, last 10 steps | 0.9959 | 0.93204 |
| 6 | 19 Oct 01:37 | LightGBM, last 20 steps | 0.9967 | 0.94296 |
| 7 | 20 Oct 05:42 | CatBoost, last 15 steps | 0.9948 | 0.94350 |
| 8 | 20 Oct 05:44 | LightGBM + CatBoost, rank blend | 0.9965 | 0.94736 |
| 9 | 20 Oct 05:45 | **LightGBM + GraphSAGE, rank blend 1:1** | 0.9922 | **0.95584** |
| 10 | 21 Oct 05:26 | LightGBM + GraphSAGE, probability average | 0.9951 | 0.95507 |
| 11 | 21 Oct 05:26 | LightGBM : CatBoost : GraphSAGE, rank blend 1:1:2 | 0.9917 | 0.95518 |
| 12 | 21 Oct 05:52 | #9 smoothed over the graph (α = 0.3, k = 3) | 0.9872 | 0.94175 |
| 13 | 22 Oct 04:02 | LightGBM + GAT, rank blend 1:1 | 0.9971 | 0.95573 |
| 14 | 22 Oct 04:02 | **LightGBM : GraphSAGE (10 seeds) : GAT, 2:1:1** | 0.9971 | **0.95627** |
| 15 | 22 Oct 04:15 | LightGBM + APPNP (with self-loops), 1:1 | 0.9913 | 0.93173 |
| 16 | 23 Oct 14:05 | **LightGBM (5 seeds) + GraphSAGE on local features, 1:1** | 0.9977 | **0.95938** |
| 17 | 23 Oct 14:05 | LightGBM (5 seeds) + domain-adversarial GraphSAGE, 1:1 | 0.9962 | 0.94863 |
| 18 | 23 Oct 14:05 | LightGBM (5 seeds) : GraphSAGE (10 seeds) : GAT, 2:1:1 | 0.9971 | 0.95641 |

Val AUC for rows 1–12 is the mean of both original folds; for rows 13–18 it's fold 1 only.

**Built but not submitted** (in `outputs/`):
- `submission_ensemble_graph.csv`
- `submission_no_time_proxies.csv`, `submission_pagerank.csv`, `submission_no_time_proxies_pagerank.csv`
- `submission_window_{5,8,12}.csv`
- other self-training variants
- `submission_arch_*_w15.csv`: MLP, GraphSAGE alone, and other blends
- `submission_wblend_*_w15.csv`: other weightings (§5.8)
- `submission_sn_*_w15.csv`: other smoothing and normalisation variants (§5.10)
- `submission_gnn_*_w15.csv`: sage3 and other GAT/GraphSAGE blends (§5.11)
- `submission_prior_s*_lgb_sage10_gat_211_w15.csv`: prior-shift corrections (§5.12)
- `submission_more_*_w15.csv`: other GraphSAGE variants, including lgb5 : sage10 : GAT : sage_local 3:1:1:1 (§5.13)
- `submission_final_lgb5_sage10_gat_sage3_3_1_1_1_w15.csv` (§5.13)

**Final selection note:** Kaggle ranks you on the private leaderboard using your chosen final submissions. Choose two genuinely different approaches, not just the top two public scores. Current plan:
- **primary:** #16, LightGBM (5 seeds) + local-features GraphSAGE (0.95938)
- **backup:** either #18, the 2:1:1 blend with all-feature graph models (0.95641), or #8, LightGBM + CatBoost (0.94736), the tree-only hedge in case the graph gain doesn't hold on the private split

---

## 8. Limitations and open questions
- **The metric isn't confirmed** from the Evaluation page (AUC is inferred).
- **We couldn't check the competition rules:** external data, AI-tool policy and team rules. Check the course policy on AI-assisted work, since this project was developed with an AI coding assistant (Claude Code). Disclose it in the report if required.
- Leaderboard comparisons use the **public** portion of the test set. Earlier tree results use a single seed, while later neural and LightGBM ensembles average multiple seeds as described above. Small differences may reflect training variability or the particular public split and may not hold on the private leaderboard.
- The neural models were not tuned: no learning-rate or epoch search, no mini-batching.
- With 18 leaderboard points and dozens of candidate schemes or blends, choices driven by the public leaderboard risk overfitting its split. We limited this to a few coarse probes.
- **Ideas not yet tried:**
  - local-features versions of GAT, and sage_local with 10 seeds
  - blends combining sage_local with sage10 and GAT (built, not submitted)
  - blending the 15- and 20-step tree models
  - temporal graph models (e.g. EvolveGCN; its paper also reports failure after the dark-market shutdown)
- **Tried since the first draft of this list:** more seeds, deeper GraphSAGE, GAT, APPNP (§5.11); other GraphSAGE windows and seed-averaged LightGBM (§5.13).

---

## 9. Reproducibility

```bash
# environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas pyarrow numpy scipy scikit-learn lightgbm xgboost catboost torch networkx matplotlib

# data (needs Kaggle token and accepted competition rules)
kaggle competitions download -c bt-4012-competition-2026 -p data && unzip -o data/*.zip -d data
# the scripts read data/train.parquet and data/test.parquet; create them once with pandas:
#   pd.read_csv('data/train.csv').to_parquet('data/train.parquet'), likewise for test

# experiments, in the order they were run
.venv/bin/python src/run.py                      # §5.1 baseline ladder
.venv/bin/python src/visualise.py                # §4 figures
.venv/bin/python src/exp_pagerank.py             # §5.2
.venv/bin/python src/exp_robust.py               # §5.5
.venv/bin/python src/exp_window_selftrain.py     # §5.6
.venv/bin/python src/exp_architectures.py trees  # §5.7: run these three separately (OpenMP clash)
.venv/bin/python src/exp_architectures.py nn
.venv/bin/python src/exp_architectures.py blend
.venv/bin/python src/exp_weighted_blend.py       # §5.8 (needs the §5.7 predictions)
.venv/bin/python src/exp_resample_val.py trees   # §5.9: three separate runs again
.venv/bin/python src/exp_resample_val.py nn
.venv/bin/python src/exp_resample_val.py score
.venv/bin/python src/exp_smooth_norm.py trees    # §5.10
.venv/bin/python src/exp_smooth_norm.py nn
.venv/bin/python src/exp_smooth_norm.py score
.venv/bin/python src/exp_gnn.py nn               # §5.11
.venv/bin/python src/exp_gnn.py score
.venv/bin/python src/exp_prior_shift.py          # §5.12 (needs the §5.11 predictions)
.venv/bin/python src/exp_final_blends.py         # §5.13: 5-seed LightGBM and stability blends
.venv/bin/python src/exp_gnn_more.py nn          # §5.13: GraphSAGE variants
.venv/bin/python src/exp_gnn_more.py score
```

Some analyses were one-off inline scripts. Their numbers are recorded above:
- §5.3 and §5.4: time-proxy R², SHAP attention, top-k models. Re-create them from `features.build("raw")` and `models.fit_predict_lgb` if they're needed as code.
- §5.8: the degree-group and confidence-weighting analyses.
- §5.10: the homophily count.

### File map

| File | Purpose |
|---|---|
| `src/common.py` | Shared folds, metrics and output path (no ML-library imports) |
| `src/features.py` | Data loading; visible-graph neighbour features; PageRank; `TIME_PROXIES` |
| `src/models.py` | LightGBM (default and regularised), XGBoost, RF, CatBoost; rank averaging |
| `src/nn_models.py` | MLP, GraphSAGE, GAT, APPNP and domain-adversarial GraphSAGE in PyTorch |
| `src/run.py` | Phase 1 experiment ladder and submission writer |
| `src/exp_pagerank.py` | Phase 2 |
| `src/exp_robust.py` | Phase 5 (and the drift diagnostic) |
| `src/exp_window_selftrain.py` | Phase 6 |
| `src/exp_architectures.py` | Phase 7 |
| `src/exp_weighted_blend.py` | Phase 8 |
| `src/exp_resample_val.py` | Phase 9 |
| `src/exp_smooth_norm.py` | Phase 10 |
| `src/exp_gnn.py` | Phase 11 |
| `src/exp_prior_shift.py` | Phase 12 |
| `src/exp_gnn_more.py`, `src/exp_final_blends.py` | Phase 13 |
| `src/visualise.py` | Figures 1–4 |
| `docs/experiment_log.md` | This log |
| `docs/submissions.csv` | Submission history |
| `docs/figures/` | Figures for the report |
