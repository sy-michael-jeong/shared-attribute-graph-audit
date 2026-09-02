# -*- coding: utf-8 -*-
"""Per-relation structure statistics.

An edge count alone does not say how many flows a relation reaches, because
one node can appear in many edges. Under a per-flow neighbour budget the edge
count of a relation follows how many flows share a value with at least one
other flow, so a populated field can still reach very few of them. This script
reports the quantities needed to tell those cases apart.

  edges                  number of edges
  incident_nodes         flows touched by at least one edge
  incident_rate_covered  incident nodes over the flows that carry the field
  incident_rate_all      incident nodes over all flows
  isolated_rate          flows touched by no edge
  mean_degree            mean degree among incident nodes, at most the
                         per-flow neighbour budget of the construction
  degree_by_class        mean degree per class
  components             number of connected components, largest, median size
  edge_homophily         fraction of edges joining flows of the same label
  cross_partition_rate   fraction of edges crossing a partition boundary

Node indices follow the train, val, test concatenation used when the graphs
are built.

Usage:
    python graph_stats.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_time_strat --out runs/graph_stats
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RELATION_COLUMN, SPLITS, is_valid

def components(n, ei):
    """Union-find, to avoid a scipy dependency."""
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(ei[0], ei[1]):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(i) for i in range(n)])
    _, sizes = np.unique(roots, return_counts=True)
    return sizes


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", default="results/structure")
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in args.datasets:
        p = root / name
        if not p.is_dir():
            print("[skip] %s" % p)
            continue

        ys = [np.load(p / ("y_%s.npy" % s)) for s in SPLITS]
        y = np.concatenate(ys)
        n = len(y)
        part = np.concatenate([np.full(len(a), i) for i, a in enumerate(ys)])
        metas = [pd.read_csv(p / ("meta_%s.csv" % s), low_memory=False) for s in SPLITS]
        allm = pd.concat(metas, ignore_index=True)

        rels = {}
        print("\n=== %s  (n=%d)" % (name, n))
        for f in sorted(p.glob("hin_edges_*.npy")):
            rel = f.stem.replace("hin_edges_", "")
            ei = np.load(f)
            if ei.size == 0:
                continue
            col = RELATION_COLUMN.get(rel)
            covered = (allm[col].astype(str).apply(is_valid).values.sum()
                       if col and col in allm.columns and rel != "via_timebin" else n)

            # Edges are stored in both directions, so a node appears once as
            # a source and once as a target for each neighbour it has. Counting
            # the sources alone therefore gives the degree.
            deg = np.bincount(ei[0], minlength=n)
            inc = int((deg > 0).sum())
            sizes = components(n, ei)
            same = int((y[ei[0]] == y[ei[1]]).sum())
            cross = int((part[ei[0]] != part[ei[1]]).sum())
            dbc = {int(c): round(float(deg[y == c].mean()), 2)
                   for c in np.unique(y)}

            rels[rel] = {
                "edges": int(ei.shape[1]),
                "covered_flows": int(covered),
                "incident_nodes": inc,
                "incident_rate_covered": round(inc / covered, 4) if covered else None,
                "incident_rate_all": round(inc / n, 4),
                "isolated_rate": round(1 - inc / n, 4),
                "mean_degree": round(float(deg[deg > 0].mean()), 2) if inc else 0.0,
                "degree_by_class": dbc,
                "components": int(len(sizes)),
                "component_max": int(sizes.max()),
                "component_median": int(np.median(sizes)),
                "edge_homophily": round(same / ei.shape[1], 4),
                "cross_partition_rate": round(cross / ei.shape[1], 4),
            }
            r = rels[rel]
            print("  %-22s edges %8d  incident %7d (%.3f of covered, %.3f of all)"
                  " homophily %.3f  cross %.3f"
                  % (rel, r["edges"], r["incident_nodes"],
                     r["incident_rate_covered"] or 0.0, r["incident_rate_all"],
                     r["edge_homophily"], r["cross_partition_rate"]))
        report[name] = {"n_flows": n, "relations": rels}

    json.dump(report, open(out / "summary.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "summary.json"))


if __name__ == "__main__":
    main()
