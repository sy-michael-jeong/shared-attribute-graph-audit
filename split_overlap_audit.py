# -*- coding: utf-8 -*-
"""Field-value overlap between the training and test partitions (Sec. 3.3).

A split protocol that is valid at the level of rows can still leave the same
identifier value on both sides of the boundary, in which case a relation built
on that field can reconnect test flows to training flows. The overlap is
therefore a property of the realized split, not of the dataset, and has to be
recomputed for every protocol the paper reports.

For each field, overlap is the fraction of distinct test values that also
occur in training, counted over the flows that carry the field. Missing-value
tokens are excluded, so an empty field contributes nothing. A weighted variant
is also reported, in which each test flow rather than each distinct value is
counted, because a single dominant value can carry most of the traffic while
counting once.

Run this once per processed data root and pass the protocol name, so that the
random and the order-preserving splits land in separate files.

Usage:
    python split_overlap_audit.py --protocol random \
        --data data/processed_random \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/split_protocol/overlap/random.json

    python split_overlap_audit.py --protocol order_preserving \
        --data data/processed_time_strat \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/split_protocol/overlap/order_preserving.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RELATION_COLUMN, SPLITS, is_valid, relation_values


def field_overlap(train_vals: pd.Series, test_vals: pd.Series) -> dict:
    """Overlap of test values with training values, by type and by flow."""
    tr = train_vals[train_vals.map(is_valid)]
    te = test_vals[test_vals.map(is_valid)]
    if len(te) == 0:
        return {"coverage_train": round(len(tr) / max(len(train_vals), 1), 4),
                "coverage_test": 0.0, "n_test_flows_covered": 0,
                "distinct_train": int(tr.nunique()), "distinct_test": 0,
                "overlap_by_value": None, "overlap_by_flow": None}
    seen = set(tr.unique())
    distinct_te = set(te.unique())
    hit = distinct_te & seen
    by_flow = float(te.isin(seen).mean())
    return {"coverage_train": round(len(tr) / max(len(train_vals), 1), 4),
            "coverage_test": round(len(te) / max(len(test_vals), 1), 4),
            "n_test_flows_covered": int(len(te)),
            "distinct_train": len(seen), "distinct_test": len(distinct_te),
            "overlap_by_value": round(len(hit) / len(distinct_te), 4),
            "overlap_by_flow": round(by_flow, 4)}


def audit_dataset(data_dir: Path, dataset: str) -> dict:
    d = data_dir / dataset
    meta = {s: pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
            for s in SPLITS}
    n = {s: len(meta[s]) for s in SPLITS}
    allm = pd.concat([meta[s] for s in SPLITS], ignore_index=True)

    out = {"n_flows": int(sum(n.values())),
           "n_by_split": {s: int(n[s]) for s in SPLITS},
           "fields": {}}
    for rel, col in RELATION_COLUMN.items():
        if col not in allm.columns:
            continue
        series = relation_values(allm, rel)
        if series is None:
            continue
        lo_te = n["train"] + n["val"]
        # Keyed by relation, because TimeBin compares the 300-second bucket
        # identifier the relation uses and not the raw timestamp column.
        out["fields"][rel] = {"column": col, **field_overlap(
            series.iloc[0:n["train"]].reset_index(drop=True),
            series.iloc[lo_te:].reset_index(drop=True))}
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--protocol", required=True,
                    help="name recorded in the output, e.g. random")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {"protocol": args.protocol, "data_root": str(root), "datasets": {}}
    for ds in args.datasets:
        if not (root / ds).is_dir():
            raise RuntimeError("%s is missing under %s" % (ds, root))
        report["datasets"][ds] = audit_dataset(root, ds)
        print("== %s ==" % ds)
        for rel, v in report["datasets"][ds]["fields"].items():
            print("   %-22s value %-7s flow %-7s (test covered %d)"
                  % (rel, v["overlap_by_value"], v["overlap_by_flow"],
                     v["n_test_flows_covered"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
