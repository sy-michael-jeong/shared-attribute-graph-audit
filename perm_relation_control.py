# -*- coding: utf-8 -*-
"""Relation value permutation, separating missingness from relation semantics.

A relation joins flows that carry the same value. Two things are entangled in
that construction, which flows carry a value at all, and what the value is.
The permutation separates them.

  missingness preserved   a flow that carried a value still carries one, and a
                          flow that carried none still carries none
  semantics destroyed     the valid values are shuffled among the flows that
                          carry them, so no two flows match for a reason

If the score holds, the gain came from the missingness and degree pattern. If
it falls, matching values contributed.

The script also prints two reference lines that need no training, the macro-F1
of always predicting the majority class, and the macro-F1 of a rule that reads
only whether the field is populated.

Usage:
    # 1) build the permuted metadata and print the reference lines
    python perm_relation_control.py --datasets bccc_dohbrw \
        --data data/processed_time_strat --out data/perm_s42 \
        --fields cert_subject cert_validity_bucket --seed 42

    # 2) rebuild the edges on the permuted metadata

    # 3) re-train over the same five seeds as the reported configuration
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import SPLITS, is_valid, macro_f1





def reference_lines(meta_te: pd.DataFrame, y_te: np.ndarray, fields):
    """Reference lines that need no training."""
    u, c = np.unique(y_te, return_counts=True)
    major = int(u[np.argmax(c)])
    minor = int(u[np.argmin(c)])
    out = {"test_size": int(len(y_te)),
           "class_counts": {int(k): int(v) for k, v in zip(u, c)},
           "always_majority_macro_f1": round(macro_f1(y_te, np.full_like(y_te, major)), 4)}
    for f in fields:
        if f not in meta_te.columns:
            continue
        present = meta_te[f].astype(str).apply(is_valid).values
        # present implies majority, absent implies minority
        pred = np.where(present, major, minor)
        out["availability_rule_%s" % f] = {
            "present_ratio": round(float(present.mean()), 4),
            "macro_f1": round(macro_f1(y_te, pred), 4),
            "present_class_mix": {int(k): int(v) for k, v in
                                  zip(*np.unique(y_te[present], return_counts=True))},
            "absent_class_mix": {int(k): int(v) for k, v in
                                 zip(*np.unique(y_te[~present], return_counts=True))},
        }
    return out


def permute_field(metas, field, rng):
    """Shuffle the valid values across all three partitions, keeping the
    missingness pattern intact."""
    lens = [len(m) for m in metas]
    col = pd.concat([m[field].astype(str) for m in metas], ignore_index=True)
    mask = col.apply(is_valid).values
    vals = col.values.copy()
    idx = np.where(mask)[0]
    vals[idx] = vals[rng.permutation(idx)]
    off = 0
    for m, n in zip(metas, lens):
        m[field] = vals[off:off + n]
        off += n
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="meta columns to permute, e.g. cert_subject cert_validity_bucket")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_root, out_root = Path(args.data), Path(args.out)
    report = {}

    for name in args.datasets:
        src, dst = src_root / name, out_root / name
        if not src.is_dir():
            print("[skip] %s: %s not found" % (name, src)); continue
        dst.mkdir(parents=True, exist_ok=True)

        # copy everything except the edges, which are rebuilt afterwards
        for p in sorted(src.iterdir()):
            if p.is_file() and not p.name.startswith("hin_edges_") and p.name != "hin_summary.json":
                shutil.copy2(p, dst / p.name)

        metas = [pd.read_csv(dst / ("meta_%s.csv" % s)) for s in SPLITS]
        ys = [np.load(dst / ("y_%s.npy" % s)) for s in SPLITS]

        ref = reference_lines(metas[2], ys[2], args.fields)
        rng = np.random.default_rng(args.seed)
        moved = {f: permute_field(metas, f, rng) for f in args.fields if f in metas[0].columns}

        for m, s in zip(metas, SPLITS):
            m.to_csv(dst / ("meta_%s.csv" % s), index=False)

        report[name] = {"permuted_fields": moved, "reference_lines": ref}
        print("\n=== %s" % name)
        print("  permuted:", moved)
        print("  test size %d  class counts %s" % (ref["test_size"], ref["class_counts"]))
        print("  always-majority macro-F1: %.4f" % ref["always_majority_macro_f1"])
        for f in args.fields:
            k = "availability_rule_%s" % f
            if k in ref:
                r = ref[k]
                print("  availability rule [%s]: macro-F1 %.4f (present %.3f)"
                      % (f, r["macro_f1"], r["present_ratio"]))
                print("     present class mix %s | absent class mix %s"
                      % (r["present_class_mix"], r["absent_class_mix"]))

    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_root / "permutation_report.json", "w"), indent=2)
    print("\n[saved] %s" % (out_root / "permutation_report.json"))


if __name__ == "__main__":
    main()
