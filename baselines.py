# -*- coding: utf-8 -*-
"""Reference lines computed without training (Table 3 availability row,
Table 7 majority column).

  majority      macro-F1 on the test partition of always predicting the
                majority class.
  availability  macro-F1 of a depth-one rule that reads only whether a field
                is populated. It reads no value at all, so it shows whether a
                relation can separate the classes without any of its values
                being matched.

The direction of the availability rule is not fixed. Which class to predict
when the field is present is chosen on the training partition, and that choice
is applied to the test partition. Fixing the direction to present-implies-
majority would miss the opposite arrangement, in which the flows that carry a
value belong to the minority class, and such a dataset would look free of the
shortcut when it is not.

Both figures are predictive scores on the test partition rather than
descriptive statistics over the whole corpus. The training partition is used
only to pick the direction.

When the presence pattern of a field is constant, either populated everywhere
or empty everywhere, the rule is undefined. Those fields are marked constant
and excluded. A dataset in which every field is constant has no availability
entry.

Usage:
    python baselines.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_time_strat --out runs/baselines
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import SPLITS, is_valid, macro_f1

# meta columns matching the metapath list of config.yaml
FIELDS = [
    ("sni", "SNI"),
    ("ja3", "JA3"),
    ("cert_subject", "CertSubject"),
    ("alpn", "ALPN"),
    ("cert_issuer_org", "CertIssuerOrg"),
    ("tls_cipher_group", "TLSCipherGroup"),
    ("cert_validity_bucket", "CertValidity"),
    ("src_ip", "SrcHost"),
    ("dst_ip", "DstHost"),
]



def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", default="results/reference_lines")
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in args.datasets:
        p = root / name
        if not p.is_dir():
            print("[skip] %s" % p)
            continue

        y_tr = np.load(p / "y_train.npy")
        y = np.load(p / "y_test.npy")
        meta_tr = pd.read_csv(p / "meta_train.csv", low_memory=False)
        meta = pd.read_csv(p / "meta_test.csv", low_memory=False)
        n_total = sum(len(np.load(p / ("y_%s.npy" % s))) for s in SPLITS)

        u, c = np.unique(y, return_counts=True)
        major, minor = int(u[np.argmax(c)]), int(u[np.argmin(c)])
        maj_f1 = macro_f1(y, np.full_like(y, major))

        fields, best = {}, None
        for col, label in FIELDS:
            if col not in meta.columns or col not in meta_tr.columns:
                continue
            pr_tr = meta_tr[col].astype(str).apply(is_valid).values
            pr_te = meta[col].astype(str).apply(is_valid).values
            ratio = float(pr_te.mean())
            if pr_te.all() or (~pr_te).all() or pr_tr.all() or (~pr_tr).all():
                fields[label] = {"present_ratio": round(ratio, 4), "status": "constant"}
                continue

            # Direction chosen on train, score measured on test.
            cand = {"present_major": (major, minor), "present_minor": (minor, major)}
            tr_scores = {k: macro_f1(y_tr, np.where(pr_tr, a, b))
                         for k, (a, b) in cand.items()}
            polarity = max(tr_scores, key=tr_scores.get)
            a, b = cand[polarity]
            score = macro_f1(y, np.where(pr_te, a, b))

            fields[label] = {
                "present_ratio": round(ratio, 4),
                "polarity": polarity,
                "train_macro_f1": {k: round(v, 4) for k, v in tr_scores.items()},
                "macro_f1": round(score, 4),
                "macro_f1_both_polarities_on_test": {
                    k: round(macro_f1(y, np.where(pr_te, x, z)), 4)
                    for k, (x, z) in cand.items()},
                "status": "ok"}
            if best is None or score > best[1]:
                best = (label, score)

        report[name] = {
            "n_flows_total": n_total,
            "n_test": int(len(y)),
            "test_class_counts": {int(k): int(v) for k, v in zip(u, c)},
            "majority_macro_f1": round(maj_f1, 4),
            "availability_per_field": fields,
            "availability_best": ({"field": best[0], "macro_f1": round(best[1], 4)}
                                  if best else None),
        }

        print("\n=== %s  (n=%d, test=%d)" % (name, n_total, len(y)))
        print("  majority macro-F1        : %.4f" % maj_f1)
        if best:
            print("  availability best        : %s %.4f" % (best[0], best[1]))
        else:
            print("  availability best        : --- (no field varies in presence)")
        for label, d in fields.items():
            if d["status"] == "ok":
                print("     %-16s present %.3f  macro-F1 %.4f  [%s]"
                      % (label, d["present_ratio"], d["macro_f1"], d["polarity"]))
            else:
                print("     %-16s present %.3f  constant" % (label, d["present_ratio"]))

    json.dump(report, open(out / "summary.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "summary.json"))


if __name__ == "__main__":
    main()
