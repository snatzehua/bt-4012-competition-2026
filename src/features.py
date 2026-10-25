"""Feature construction for the BT4012 (Elliptic-style) Bitcoin fraud competition.

Graph quirk: for time steps 36-49 only the test transactions were kept, so a
test node's neighbours that still exist are all other test (i.e. labelled)
nodes. In train every unlabelled node is still present. To keep neighbour
aggregates meaning the same thing in train and test, aggregates are computed
over "visible" nodes only: labelled train nodes plus all test nodes.
Degrees are counted from the full edge list, which is consistent in both.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data"

# Elliptic layout: first 93 features describe the transaction itself, the rest
# are one-hop aggregates precomputed by the dataset authors.
N_LOCAL = 93


def load():
    tr = pd.read_parquet(DATA / "train.parquet")
    te = pd.read_parquet(DATA / "test.parquet")
    edges = pd.read_csv(DATA / "txs_edgelist.csv")
    return tr, te, edges


def raw_feature_cols(df):
    return [c for c in df.columns if c.startswith("feat_")]


def _row_normalise(m):
    deg = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    return sp.diags(inv) @ m


def graph_features(nodes, edges, n_agg=N_LOCAL):
    """nodes: frame with txId, visible flag and feat_* columns, one row per node.

    Returns a frame aligned with `nodes` holding degree and neighbour-mean features.
    """
    idx = pd.Series(np.arange(len(nodes)), index=nodes.txId.values)
    out = pd.DataFrame(index=nodes.index)

    # Degrees over the full edge list (edges to removed nodes still count).
    out["deg_out"] = nodes.txId.map(edges.txId1.value_counts()).fillna(0).values
    out["deg_in"] = nodes.txId.map(edges.txId2.value_counts()).fillna(0).values
    out["deg_tot"] = out.deg_out + out.deg_in

    s = edges.txId1.map(idx)
    d = edges.txId2.map(idx)
    keep = s.notna() & d.notna()
    s, d = s[keep].astype(int).values, d[keep].astype(int).values
    vis = nodes.visible.values
    keep = vis[s] & vis[d]
    s, d = s[keep], d[keep]
    n = len(nodes)
    A_out = sp.csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    A_in = A_out.T.tocsr()
    A_und = ((A_out + A_in) > 0).astype(float)

    out["vdeg_out"] = np.asarray(A_out.sum(1)).ravel()
    out["vdeg_in"] = np.asarray(A_in.sum(1)).ravel()

    cols = raw_feature_cols(nodes)[:n_agg]
    X = nodes[cols].to_numpy(dtype=float, copy=True)
    X[~vis] = 0.0
    P_out, P_in, P_und = map(_row_normalise, (A_out, A_in, A_und))

    blocks = {
        "out_mean": P_out @ X,
        "in_mean": P_in @ X,
    }
    hop1 = P_und @ X
    blocks["hop2_mean"] = P_und @ hop1
    # Contrast between the node and its neighbourhood.
    blocks["diff_mean"] = X - hop1
    for name, M in blocks.items():
        out[[f"{name}_{c}" for c in cols]] = M
    return out


def pagerank(A, alpha=0.85, tol=1e-10, max_iter=200):
    """PageRank by power iteration. A[i, j] = 1 means an edge i -> j.

    Dangling nodes (no out-links) spread their score uniformly, as in the
    standard formulation, so the scores always sum to 1.
    """
    n = A.shape[0]
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = out_deg == 0
    inv = np.divide(1.0, out_deg, out=np.zeros(n), where=~dangling)
    PT = (sp.diags(inv) @ A).T.tocsr()  # column-stochastic transpose: PT @ r pushes score along edges
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (PT @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def pagerank_features(nodes, edges):
    """PageRank per time step on the visible graph (see module docstring).

    Raw scores depend on graph size, so each is reported as n * score (mean 1
    within a step, logged) and as a percentile rank within the step.
    """
    vis = nodes[nodes.visible]
    out = pd.DataFrame(np.nan, index=nodes.index,
                       columns=["pr_fwd_log", "pr_fwd_pct", "pr_rev_log", "pr_rev_pct"])
    for _, g in vis.groupby("time_step"):
        idx = pd.Series(np.arange(len(g)), index=g.txId.values)
        s, d = edges.txId1.map(idx), edges.txId2.map(idx)
        keep = s.notna() & d.notna()
        s, d = s[keep].astype(int).values, d[keep].astype(int).values
        n = len(g)
        A = sp.csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
        # Forward follows the money (rewards receivers); reverse rewards senders.
        for name, M in (("fwd", A), ("rev", A.T.tocsr())):
            r = pagerank(M)
            out.loc[g.index, f"pr_{name}_log"] = np.log(r * n)
            out.loc[g.index, f"pr_{name}_pct"] = pd.Series(r).rank(pct=True).values
    return out


# Features that are almost a linear function of time_step (see outputs/figures/4_drift_features.png).
TIME_PROXIES = ["feat_100", "feat_101", "feat_103"]


def build(variant="raw", pagerank=False, drop=()):
    """variant: 'raw' (165 given features) or 'graph' (raw + graph features).

    pagerank adds per-time-step PageRank features; drop removes columns by name.
    """
    tr, te, edges = load()
    tr = tr.copy()
    te = te.copy()
    tr["is_test"] = False
    te["is_test"] = True
    nodes = pd.concat([tr.drop(columns="label"), te.drop(columns="index")], ignore_index=True)
    nodes["visible"] = np.r_[tr.label.notna().values, np.ones(len(te), bool)]

    feats = raw_feature_cols(nodes)
    X = nodes[feats].copy()
    if variant == "graph":
        X = pd.concat([X, graph_features(nodes, edges)], axis=1)
    if pagerank:
        X = pd.concat([X, pagerank_features(nodes, edges)], axis=1)
    X = X.drop(columns=list(drop))

    meta = nodes[["txId", "time_step", "is_test"]].copy()
    meta["label"] = np.r_[tr.label.values, np.full(len(te), np.nan)]
    meta["index"] = np.r_[np.full(len(tr), -1), te["index"].values]
    return X.astype(np.float32), meta
