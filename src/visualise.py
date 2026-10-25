"""Figures describing the transaction graph. Writes PNGs to outputs/figures/.

Usage: .venv/bin/python src/visualise.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LICIT, ILLICIT, UNKNOWN, TEST = "#2a78d6", "#eb6834", "#c3c2b7", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load():
    tr = pd.read_parquet(DATA / "train.parquet")
    te = pd.read_parquet(DATA / "test.parquet")
    e = pd.read_csv(DATA / "txs_edgelist.csv")
    nodes = pd.concat([tr.assign(split="train"), te.drop(columns="index").assign(split="test")],
                      ignore_index=True)
    nodes["cls"] = np.select(
        [nodes.split == "test", nodes.label == 1, nodes.label == 0],
        ["test", "illicit", "licit"], "unknown")
    return nodes, e


def network_snapshot(nodes, e, ts=28):
    sub = nodes[nodes.time_step == ts].set_index("txId")
    G = nx.Graph()
    G.add_nodes_from(sub.index)
    ee = e[e.txId1.isin(sub.index) & e.txId2.isin(sub.index)]
    G.add_edges_from(ee.itertuples(index=False, name=None))
    # Zoom on the illicit transaction with the most illicit neighbours.
    ill = [n for n in G if sub.at[n, "cls"] == "illicit"]
    hub = max(ill, key=lambda n: sum(sub.at[m, "cls"] == "illicit" for m in G[n]))
    zoom = nx.ego_graph(G, hub, radius=3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax, H, title in [
        (axes[0], G, f"Every transaction in time step {ts} ({len(G):,}, one connected graph)"),
        (axes[1], zoom, f"Zoom: 3 hops around one illicit hub ({len(zoom)} transactions)"),
    ]:
        pos = nx.spring_layout(H, seed=4, k=1.4 / np.sqrt(len(H)), iterations=120)
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color=GRID, width=0.6)
        for cls, col, size, z in [("unknown", UNKNOWN, 8, 1), ("licit", LICIT, 16, 2), ("illicit", ILLICIT, 26, 3)]:
            nl = [n for n in H if sub.at[n, "cls"] == cls]
            coll = nx.draw_networkx_nodes(H, pos, nodelist=nl, ax=ax, node_color=col, node_size=size,
                                          linewidths=0.6, edgecolors=SURFACE, label=f"{cls} ({len(nl):,})")
            coll.set_zorder(z)
        ax.set_title(title)
        ax.grid(False)
        ax.axis("off")
    axes[0].legend(loc="lower left", frameon=False, markerscale=2)
    fig.suptitle("Long chains of unlabelled transactions; illicit ones cluster together",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "1_network_snapshot.png", dpi=160)
    plt.close(fig)


def timeline(nodes):
    c = nodes.groupby(["time_step", "cls"]).size().unstack(fill_value=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    x = c.index.values
    a1.bar(x, c.get("licit", 0), color=LICIT, width=0.8, label="licit (train)")
    a1.bar(x, c.get("illicit", 0), bottom=c.get("licit", 0), color=ILLICIT, width=0.8, label="illicit (train)")
    a1.bar(x, c.get("test", 0), color=TEST, width=0.8, label="test (labels hidden)")
    a1.set_ylabel("labelled transactions")
    a1.set_title("Labelled transactions per time step")
    a1.legend(frameon=False, ncol=3, loc="upper left")

    lab = c.get("licit", 0) + c.get("illicit", 0)
    rate = (c.get("illicit", 0) / lab.replace(0, np.nan)).dropna()
    a2.plot(rate.index, rate.values * 100, color=ILLICIT, lw=2, marker="o", ms=4)
    a2.set_ylabel("illicit share, %")
    a2.set_xlabel("time step")
    a2.set_title("Share of labelled transactions that are illicit (train only)")
    for a in (a1, a2):
        a.axvspan(35.5, 49.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    a2.text(42.5, a2.get_ylim()[1] * 0.85, "test period", ha="center", color=INK2)
    a2.axvline(43, color=INK2, lw=1, ls="--")
    a2.text(43.2, a2.get_ylim()[1] * 0.6, "dark market\nshutdown (Elliptic)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "2_timeline.png", dpi=160)
    plt.close(fig)


def neighbourhood(nodes, e):
    cls = nodes.set_index("txId").cls
    a, b = e.txId1.map(cls), e.txId2.map(cls)
    both = a.isin(["licit", "illicit"]) & b.isin(["licit", "illicit"])
    und = pd.concat([pd.DataFrame({"u": e.txId1, "v": e.txId2}), pd.DataFrame({"u": e.txId2, "v": e.txId1})], ignore_index=True)
    und["cu"], und["cv"] = und.u.map(cls), und.v.map(cls)
    und = und[und.cu.isin(["licit", "illicit"]) & und.cv.isin(["licit", "illicit", "unknown"])]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: who do labelled transactions link to?
    mix = pd.crosstab(und.cu, und.cv, normalize="index").reindex(index=["licit", "illicit"],
                                                                columns=["licit", "illicit", "unknown"])
    left = np.zeros(2)
    for col, color in [("licit", LICIT), ("illicit", ILLICIT), ("unknown", UNKNOWN)]:
        vals = mix[col].values * 100
        a1.barh(["licit tx", "illicit tx"], vals, left=left, color=color, height=0.55,
                edgecolor=SURFACE, linewidth=2, label=f"neighbour is {col}")
        for i, v in enumerate(vals):
            if v > 6:
                a1.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=10)
        left += vals
    a1.set_xlim(0, 100)
    a1.set_xlabel("% of links")
    a1.set_title("Who each class links to")
    a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    a1.grid(axis="y", visible=False)
    print(f"labelled-labelled edges: {both.sum():,}")

    # Right: degree distribution (complementary CDF) by class.
    deg = (pd.concat([e.txId1, e.txId2]).value_counts())
    nd = nodes.assign(deg=nodes.txId.map(deg).fillna(0))
    for c, color in [("licit", LICIT), ("illicit", ILLICIT), ("unknown", UNKNOWN)]:
        d = np.sort(nd.loc[nd.cls == c, "deg"].values)
        y = 1 - np.arange(len(d)) / len(d)
        a2.step(d, y, where="post", color=color, lw=2, label=f"{c} (median {np.median(d):.0f})")
    a2.set_xscale("symlog", linthresh=1)
    a2.set_yscale("log")
    a2.set_xlabel("number of links (degree)")
    a2.set_ylabel("share of transactions with ≥ this many links")
    a2.set_title("Degree distribution")
    a2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "3_neighbourhood.png", dpi=160)
    plt.close(fig)
    return mix


def drift(nodes, feats=("feat_101", "feat_103", "feat_136", "feat_100")):
    vis = nodes[nodes.cls.isin(["licit", "illicit", "test"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, f in zip(axes.ravel(), feats):
        g = vis.groupby("time_step")[f]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(med.index, lo, hi, color=LICIT, alpha=0.18, lw=0)
        ax.plot(med.index, med, color=LICIT, lw=2)
        ax.axvspan(35.5, 49.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.set_title(f)
    for ax in axes[1]:
        ax.set_xlabel("time step")
    fig.suptitle("The features that most separate train from test just count time: median and IQR per "
                 "time step (shaded = test)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "4_drift_features.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    nodes, e = load()
    network_snapshot(nodes, e)
    timeline(nodes)
    print(neighbourhood(nodes, e).round(3))
    drift(nodes)
    print("saved to", FIG)
