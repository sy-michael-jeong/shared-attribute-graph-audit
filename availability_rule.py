# -*- coding: utf-8 -*-
"""What a rule that reads only field presence achieves (Sec. 6.1, 6.4).

A relation can only join flows that carry its field. So before asking what the
values mean, ask what the pattern of who carries the field is worth on its own.
The rule here reads nothing but presence: a flow that carries the field is
called one class, a flow that does not is called the other, and the direction
is fixed on the training partition.

`availability_null.py` answers a neighbouring question — it takes the maximum
over fields and compares it against a label permutation. That maximum is one
number for a dataset. This script reports **every field separately**, which is
what the paper needs when it follows one field through a preprocessing change.

The two are easy to confuse. On BCCC-DoH with the extractor's fill-in string
left in place, both give the same answer because CertValidity is the argmax:

    CertValidity alone      0.9999
    maximum over 6 fields   1.0000  (CertValidity)

Once the fill-in string is returned to missing, they part:

    CertValidity alone      0.5800
    maximum over 6 fields   0.7136  (CertSubject)

The shortcut does not disappear. It moves to another field. A script that only
reports the maximum cannot say that, and a paper that quotes one field cannot
be checked against a file that stores the other.

Host relations are also profiled: how many distinct endpoints the attack class
occupies. A class that sits on a handful of endpoints can be recovered from
endpoint-correlated features with no edges at all.

    python availability_rule.py --data data/processed_deg2 \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/reference_lines/availability_rule.json

    python availability_rule.py --data data/processed_deg2_bccc10_nounk \
        --datasets bccc_dohbrw \
        --out results/reference_lines/availability_rule_nounk.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RELATION_COLUMN, SPLITS, is_valid, macro_f1

# Presence is only defined for fields that carry a value. The time relation is
# derived from a timestamp that every flow has, so it is not a presence signal.
FIELDS = [c for r, c in RELATION_COLUMN.items() if r != "via_timebin"]
HOST_FIELDS = ["src_ip", "dst_ip"]


def load(d: Path):
    metas, ys = [], []
    for s in SPLITS:
        metas.append(pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False))
        ys.append(np.load(d / ("y_%s.npy" % s)))
    ns = [len(y) for y in ys]
    return pd.concat(metas, ignore_index=True), np.concatenate(ys), ns


def rule_score(present, y, ns):
    """Macro-F1 of the presence rule, on three scopes.

    All three are reported because the number changes a lot between them and
    the difference is invisible once it is rounded. On BCCC-DoH with the
    extractor's fill-in string returned to missing:

        test only     0.7136
        val + test    0.7047
        all flows     0.5800

    With the string left in place all three round to 1.000, so a table that
    pairs one scope on the left with another on the right looks consistent and
    is not. The paper reports the test partition, as it does everywhere else.

    The direction of the rule is fixed on the training partition. Choosing it
    where the score is read would be fitting the rule to the answer.
    """
    n_train, n_val = ns[0], ns[1]
    tr = slice(0, n_train)
    agree_1 = float((present[tr] == (y[tr] != 0)).mean())
    positive_means_attack = agree_1 >= 0.5
    pred = present.astype(np.int64) if positive_means_attack \
        else (~present).astype(np.int64)
    yb = (y != 0).astype(np.int64)
    scopes = {"test": slice(n_train + n_val, len(y)),
              "val_and_test": slice(n_train, len(y)),
              "all_flows": slice(0, len(y))}
    return ({k: round(macro_f1(yb[sl], pred[sl]), 4) for k, sl in scopes.items()},
            bool(positive_means_attack))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {"data_root": str(root),
              "rule": "a flow that carries the field is called one class; the "
                      "direction is fixed on the training partition",
              "datasets": {}}

    for name in args.datasets:
        d = root / name
        meta, y, ns = load(d)
        block = {"n_flows": int(len(y)),
                 "n_attack": int((y != 0).sum()),
                 "n_benign": int((y == 0).sum()),
                 "fields": {}}

        for col in FIELDS:
            if col not in meta.columns:
                continue
            present = np.array([is_valid(str(v)) for v in meta[col].values])
            scores, direction = rule_score(present, y, ns)
            block["fields"][col] = {
                "coverage": round(float(present.mean()), 4),
                "macro_f1": scores["test"],
                "macro_f1_by_scope": scores,
                "presence_means_attack": direction}

        if block["fields"]:
            best = max(block["fields"], key=lambda c: block["fields"][c]["macro_f1"])
            block["best_field"] = best
            block["best_macro_f1"] = block["fields"][best]["macro_f1"]
            block["scope_of_reported_score"] = "test"

        # How concentrated the attack class is on endpoints. A class sitting on
        # a few endpoints is recoverable without any edge.
        block["attack_endpoints"] = {}
        for col in HOST_FIELDS:
            if col not in meta.columns:
                continue
            v = meta[col].astype(str).values
            block["attack_endpoints"][col] = {
                "n_distinct_attack": int(len(set(v[y != 0]))),
                "n_distinct_benign": int(len(set(v[y == 0])))}

        report["datasets"][name] = block
        print("== %s" % name)
        for col, f in sorted(block["fields"].items(),
                             key=lambda kv: -kv[1]["macro_f1"]):
            sc = f["macro_f1_by_scope"]
            print("   %-22s coverage %.4f   macro-F1  test %.4f  "
                  "val+test %.4f  all %.4f"
                  % (col, f["coverage"], sc["test"], sc["val_and_test"],
                     sc["all_flows"]))
        for col, e in block["attack_endpoints"].items():
            print("   %-22s distinct attack endpoints %d (benign %d)"
                  % (col, e["n_distinct_attack"], e["n_distinct_benign"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
