# -*- coding: utf-8 -*-
"""Edge homophily and reach of every relation (Sec. 5.3, 6.5).

Coverage says how many flows carry a field. It does not say whether the edges
that field induces join flows of the same class. A relation can be present on
every flow and still connect the two classes at the rate chance would give.

Homophily is the fraction of edges whose two endpoints share a label. It is
read against the value chance produces, which is the sum of the squared class
proportions and differs from dataset to dataset. Class imbalance alone can put
chance above 0.9, so the raw fraction means little on its own.

    normalised homophily = (homophily - chance) / (1 - chance)

Reach is the fraction of flows that receive at least one edge. Two relations
can have the same homophily and differ by an order of magnitude in how many
flows they touch.

Neither quantity needs a model. Both are computed on the graph as built.

Cross-partition share is reported per relation as well, split into the pairs of
partitions the edges join, so that the effect of the masking control can be read
per relation rather than for the graph as a whole.

Node order follows the concatenation used when the graph is built, which is
train, then validation, then test.

Usage:
    python edge_homophily.py --data data/processed_time_strat \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/homophily/edge_homophily.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import SPLITS


def labels_and_partitions(d: Path):
    ys, ns = [], []
    for s in SPLITS:
        p = d / ("y_%s.npy" % s)
        if not p.exists():
            raise FileNotFoundError(str(p))
        y = np.load(p)
        ys.append(y)
        ns.append(len(y))
    part = np.concatenate([np.full(n, i, dtype=np.int8) for i, n in enumerate(ns)])
    return np.concatenate(ys), part


def run(root: Path, name: str) -> dict:
    d = root / name
    y, part = labels_and_partitions(d)
    n = len(y)

    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())

    out = {
        "n_flows": int(n),
        "class_counts": {str(int(k)): int(v) for k, v in zip(classes, counts)},
        "chance_homophily": round(chance, 4),
        "relations": {},
    }

    files = sorted(d.glob("hin_edges_*.npy"))
    if not files:
        print("  [%s] no edge files" % name)
        return out

    print("  [%s] %d flows, chance homophily %.4f" % (name, n, chance))
    for f in files:
        rel = f.stem.replace("hin_edges_", "")
        e = np.load(f)
        if e.size == 0:
            continue
        src, dst = e[0], e[1]
        if max(src.max(), dst.max()) >= n:
            raise SystemExit(
                "%s/%s: a node index exceeds the flow count. The graph and the "
                "labels may come from different builds." % (name, rel))
        same = float((y[src] == y[dst]).mean())
        cross = float((part[src] != part[dst]).mean())
        # Edges are stored in both directions, so counting the sources alone
        # gives the degree.
        deg = np.bincount(src, minlength=n)
        isolated = int((deg == 0).sum())
        pairs = {}
        for a in range(3):
            for b in range(3):
                c = int(((part[src] == a) & (part[dst] == b)).sum())
                if c:
                    pairs["%s-%s" % (SPLITS[a][:2], SPLITS[b][:2])] = c
        train_test = (pairs.get("tr-te", 0) + pairs.get("te-tr", 0)) / e.shape[1]
        out["relations"][rel] = {
            "n_edges": int(e.shape[1]),
            "homophily": round(same, 4),
            "lift_over_chance": round(same - chance, 4),
            "normalised_homophily": round((same - chance) / (1 - chance), 4)
            if chance < 1 else None,
            "mean_degree": round(float(deg.mean()), 3),
            "isolated_nodes": isolated,
            "reach": round(1 - isolated / n, 4),
            "cross_partition": round(cross, 4),
            "train_test_share": round(float(train_test), 4),
            "by_partition_pair": pairs,
        }
        print("    %-24s homophily %.4f (normalised %.3f)  reach %.3f  "
              "cross %.3f  train-test %.3f"
              % (rel, same, (same - chance) / (1 - chance) if chance < 1 else float("nan"),
                 1 - isolated / n, cross, train_test))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    res = {}
    for name in args.datasets:
        try:
            res[name] = run(root, name)
        except FileNotFoundError as e:
            print("  [%s] skipped, missing %s" % (name, e))
        except SystemExit as e:
            print("  [%s] skipped: %s" % (name, e))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
