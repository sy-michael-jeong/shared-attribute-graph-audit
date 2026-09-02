# -*- coding: utf-8 -*-
"""Remove HIN edges that cross a partition boundary.

Writes a copy of the processed splits in which only within-partition edges
survive, so that the graph models can be re-trained without the train-test
edges that the transductive construction otherwise provides.

It also records per-relation statistics, the raw, kept and crossing edge counts,
the breakdown by partition pair, and the fraction of nodes that still hold an
edge.

Usage:
    python mask_edges.py --data data/processed_time_strat \
        --out data/processed_time_strat_masked --datasets bccc_dohbrw
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def process_dataset(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    n_tr = len(np.load(src / "X_train.npy"))
    n_va = len(np.load(src / "X_val.npy"))
    n_te = len(np.load(src / "X_test.npy"))
    N = n_tr + n_va + n_te

    # copy everything except the edges
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("hin_edges_"):
            shutil.copy(f, dst / f.name)

    part = np.zeros(N, dtype=np.int8)
    part[n_tr:n_tr + n_va] = 1
    part[n_tr + n_va:] = 2

    stats = {"n_train": n_tr, "n_val": n_va, "n_test": n_te, "relations": {}}
    deg_before = np.zeros(N, dtype=bool)
    deg_after = np.zeros(N, dtype=bool)

    for f in sorted(src.glob("hin_edges_*.npy")):
        rel = f.stem.replace("hin_edges_", "")
        ed = np.load(f)
        if ed.size == 0:
            np.save(dst / f.name, ed)
            stats["relations"][rel] = {"raw": 0, "kept": 0}
            continue
        s, d = ed[0], ed[1]
        same = part[s] == part[d]
        kept = ed[:, same]
        np.save(dst / f.name, kept)
        pair = part[s] * 3 + part[d]  # 0..8
        cross = {}
        names = {0: "tr-tr", 1: "tr-va", 2: "tr-te", 3: "va-tr", 4: "va-va",
                 5: "va-te", 6: "te-tr", 7: "te-va", 8: "te-te"}
        for code, nm in names.items():
            c = int((pair == code).sum())
            if c:
                cross[nm] = c
        deg_before[s] = True
        deg_before[d] = True
        if kept.size:
            deg_after[kept[0]] = True
            deg_after[kept[1]] = True
        stats["relations"][rel] = {
            "raw": int(ed.shape[1]),
            "kept": int(kept.shape[1]),
            "cross_ratio": round(1.0 - kept.shape[1] / ed.shape[1], 4),
            "by_partition_pair": cross,
        }

    te = part == 2
    stats["nodes_with_edge_before"] = int(deg_before.sum())
    stats["nodes_with_edge_after"] = int(deg_after.sum())
    stats["test_nodes_with_edge_before"] = int(deg_before[te].sum())
    stats["test_nodes_with_edge_after"] = int(deg_after[te].sum())
    return stats


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", default="data/processed_deg2_masked")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--stats", default="reports/mask_stats.json")
    args = ap.parse_args()

    all_stats = {}
    for name in args.datasets:
        src = Path(args.data) / name
        dst = Path(args.out) / name
        print("== mask " + name)
        all_stats[name] = process_dataset(src, dst)
        for rel, st in all_stats[name]["relations"].items():
            if st.get("raw"):
                print("   %-22s raw=%8d kept=%8d cross=%.1f%%"
                      % (rel, st["raw"], st["kept"], 100 * st.get("cross_ratio", 0)))
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_stats, open(args.stats, "w"), indent=2)
    print("[saved] " + args.stats)


if __name__ == "__main__":
    main()
