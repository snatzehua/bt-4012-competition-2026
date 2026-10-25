"""Neural models: a plain MLP on node features, and graph models on the transaction graph
(GraphSAGE, GAT and APPNP).

Both train full-batch on CPU. The graph is the "visible" graph from features.py
(labelled train nodes + test nodes), so a node's neighbourhood looks the same in
train and test. Edges never cross time steps, so message passing stays within a step.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

NN_PARAMS = dict(hidden=128, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=150)


def standardise(X, fit_rows):
    """Scale with statistics from the training rows; clip the heavy tails."""
    mu = X[fit_rows].mean(0)
    sd = X[fit_rows].std(0) + 1e-6
    return np.clip((X - mu) / sd, -5, 5).astype(np.float32)


def _to_torch_sparse(m):
    m = m.tocoo()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape).coalesce()


def neighbour_operators(txids, edges):
    """Row-normalised in- and out-neighbour mean operators over the given nodes."""
    pos = {t: i for i, t in enumerate(txids)}
    s = edges.txId1.map(pos)
    d = edges.txId2.map(pos)
    keep = s.notna() & d.notna()
    s, d = s[keep].astype(int).values, d[keep].astype(int).values
    n = len(txids)
    A = sp.csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    ops = []
    for M in (A, A.T.tocsr()):
        deg = np.asarray(M.sum(1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
        ops.append(_to_torch_sparse(sp.diags(inv) @ M))
    return ops  # [mean over out-neighbours, mean over in-neighbours]


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, ops=None):
        return self.net(x).squeeze(-1)


class SAGELayer(nn.Module):
    """h' = W_self h + W_out mean(out-neighbours) + W_in mean(in-neighbours)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.out_lin = nn.Linear(d_in, d_out, bias=False)
        self.in_lin = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h, ops):
        P_out, P_in = ops
        return self.self_lin(h) + self.out_lin(torch.sparse.mm(P_out, h)) + self.in_lin(torch.sparse.mm(P_in, h))


class GraphSAGE(nn.Module):
    def __init__(self, d_in, hidden, dropout, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.layers = nn.ModuleList(SAGELayer(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden * (layers + 1), 1)  # skip connections: input and every hop's view
        self.dropout = dropout

    def forward(self, x, ops, return_emb=False):
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        hs = [h]
        for layer, norm in zip(self.layers, self.norms):
            h = F.dropout(F.relu(norm(layer(h, ops))), self.dropout, self.training)
            hs.append(h)
        emb = torch.cat(hs, dim=1)
        out = self.head(emb).squeeze(-1)
        return (out, emb) if return_emb else out


class GATLayer(nn.Module):
    """Like SAGELayer, but each neighbour's message is weighted by a learned attention
    score (softmax over the node's neighbours), separately for out- and in-edges."""

    def __init__(self, d_in, d_out, heads=4):
        super().__init__()
        self.heads, self.d_head = heads, d_out // heads
        self.self_lin = nn.Linear(d_in, d_out)
        self.lins = nn.ModuleList(nn.Linear(d_in, d_out, bias=False) for _ in range(2))
        self.att = nn.ParameterList(nn.Parameter(torch.randn(2, heads, self.d_head) * 0.1) for _ in range(2))

    def forward(self, h, ops):
        n = h.shape[0]
        out = self.self_lin(h)
        for P, lin, att in zip(ops, self.lins, self.att):
            i, j = P.indices()  # node i receives from neighbour j
            z = lin(h).view(n, self.heads, self.d_head)
            e = F.leaky_relu((z[i] * att[0]).sum(-1) + (z[j] * att[1]).sum(-1), 0.2)  # (edges, heads)
            emax = torch.full((n, self.heads), -1e9).scatter_reduce(0, i[:, None].expand_as(e), e, "amax")
            w = torch.exp(e - emax[i])
            w = w / torch.zeros(n, self.heads).index_add(0, i, w)[i].clamp_min(1e-12)
            msg = torch.zeros(n, self.heads, self.d_head).index_add(0, i, w[..., None] * z[j])
            out = out + msg.reshape(n, -1)
        return out


class GAT(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.l1, self.l2 = GATLayer(hidden, hidden), GATLayer(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden * 3, 1)
        self.dropout = dropout

    def forward(self, x, ops):
        h0 = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        h1 = F.dropout(F.relu(self.norm1(self.l1(h0, ops))), self.dropout, self.training)
        h2 = F.dropout(F.relu(self.norm2(self.l2(h1, ops))), self.dropout, self.training)
        return self.head(torch.cat([h0, h1, h2], dim=1)).squeeze(-1)


class APPNP(nn.Module):
    """An MLP's per-node logits, then propagated: z <- (1 - a) * P z + a * z0, K times,
    where P is the mean over the node itself and its undirected neighbours. The
    self-loop keeps isolated nodes (about a quarter of them) at their own logits;
    without it they shrink to a * z0. Reaches K hops without adding parameters."""

    def __init__(self, d_in, hidden, dropout, k=10, alpha=0.1):
        super().__init__()
        self.mlp = MLP(d_in, hidden, dropout)
        self.k, self.alpha = k, alpha
        self._P = None

    @staticmethod
    def propagation_operator(ops):
        n = ops[0].shape[0]
        idx = torch.cat([ops[0].indices(), ops[1].indices(), torch.arange(n).repeat(2, 1)], dim=1)
        A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (n, n)).coalesce()
        idx = A.indices()  # coalesced, so each (i, j) appears once even if the edge is in both directions
        deg = torch.zeros(n).index_add(0, idx[0], torch.ones(idx.shape[1]))
        return torch.sparse_coo_tensor(idx, 1 / deg[idx[0]], (n, n)).coalesce()

    def forward(self, x, ops):
        if self._P is None:
            self._P = self.propagation_operator(ops)
        z0 = self.mlp(x)[:, None]
        z = z0
        for _ in range(self.k):
            z = (1 - self.alpha) * torch.sparse.mm(self._P, z) + self.alpha * z0
        return z.squeeze(-1)


def fit_predict_nn(kind, X, y, train_idx, pred_idx, ops=None, seed=0, params=NN_PARAMS):
    """Full-batch training on all nodes in X; loss only on train_idx.

    X: standardised features for every node in the graph (float32 array).
    y: labels aligned with X (NaN where unknown).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = {"mlp": MLP, "sage": GraphSAGE, "sage3": lambda *a: GraphSAGE(*a, layers=3),
           "gat": GAT, "appnp": APPNP}[kind]
    model = cls(X.shape[1], params["hidden"], params["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    x = torch.from_numpy(X)
    tr = torch.as_tensor(train_idx)
    yt = torch.tensor(y[train_idx], dtype=torch.float32)

    for _ in range(params["epochs"]):
        model.train()
        opt.zero_grad()
        if kind == "mlp":
            loss = F.binary_cross_entropy_with_logits(model(x[tr]), yt)
        else:
            loss = F.binary_cross_entropy_with_logits(model(x, ops)[tr], yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(x[torch.as_tensor(pred_idx)]) if kind == "mlp" else model(x, ops)[pred_idx]
    return torch.sigmoid(out).numpy()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def fit_predict_dann(X, y, train_idx, pred_idx, ops, seed=0, params=NN_PARAMS, lam=0.3):
    """Domain-adversarial GraphSAGE: a second head tries to tell training-period nodes
    from prediction-period nodes using the GraphSAGE embedding, and its gradient is
    reversed into the encoder, pushing the embedding towards features that look the
    same in both periods. lam ramps up from 0 over training (Ganin et al. 2016)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GraphSAGE(X.shape[1], params["hidden"], params["dropout"])
    dom = nn.Sequential(nn.Linear(params["hidden"] * 3, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(list(model.parameters()) + list(dom.parameters()),
                            lr=params["lr"], weight_decay=params["weight_decay"])
    x = torch.from_numpy(X)
    tr = torch.as_tensor(train_idx)
    yt = torch.tensor(y[train_idx], dtype=torch.float32)
    d_idx = torch.as_tensor(np.r_[train_idx, pred_idx])
    d_y = torch.cat([torch.zeros(len(train_idx)), torch.ones(len(pred_idx))])

    for ep in range(params["epochs"]):
        model.train()
        opt.zero_grad()
        out, emb = model(x, ops, return_emb=True)
        ramp = lam * (2 / (1 + np.exp(-10 * ep / params["epochs"])) - 1)
        d_out = dom(_GradReverse.apply(emb[d_idx], ramp)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(out[tr], yt) + F.binary_cross_entropy_with_logits(d_out, d_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(x, ops)[pred_idx]
    return torch.sigmoid(out).numpy()
