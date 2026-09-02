# -*- coding: utf-8 -*-
"""How large an availability score chance alone produces (Sec. 3.3).

The availability row of Table 3 names, for each dataset, the field whose
presence rule scores highest. That field is picked by its test score, so the
row is a maximum over the fields of the dataset. A maximum over nine candidates
is larger than any single candidate would be, even when no field carries any
signal, and the reported score has to be read against that.

This script measures the size of that effect directly. The class labels are
permuted while the partition boundaries, the class marginal and the presence
pattern of every field are held fixed, so the only thing destroyed is the
association between presence and label. The whole selection procedure is then
repeated on the permuted labels, maximum included. Comparing the observed
maximum against this distribution says how much of it chance can explain.

  observed        best availability macro-F1 over the fields of the dataset
  null mean       average of that maximum under permuted labels
  null p95        95th percentile of the same maximum
  p_value         fraction of permutations reaching the observed value

A dataset whose observed maximum sits inside the null distribution has no
demonstrated availability shortcut, whatever the raw score looks like.

Usage:
    python availability_null.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_time_strat --permutations 1000 \
        --out results/reference_lines/availability_null.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SPLITS, is_valid, macro_f1

FIELDS = [
    ("sni", "SNI"), ("ja3", "JA3"), ("cert_subject", "CertSubject"),
    ("alpn", "ALPN"), ("cert_issuer_org", "CertIssuerOrg"),
    ("tls_cipher_group", "TLSCipherGroup"),
    ("cert_validity_bucket", "CertValidity"),
    ("src_ip", "SrcHost"), ("dst_ip", "DstHost"),
]


def best_availability(pres_tr, pres_te, y_tr, y_te, major, minor):
    """Maximum availability macro-F1 over the fields, direction fixed on train.

    Identical to the rule of baselines.py: the polarity that scores higher on
    the training partition is the one measured on the test partition.
    """
    best = -1.0
    best_field = None
    for label in pres_tr:
        ptr, pte = pres_tr[label], pres_te[label]
        cand = ((major, minor), (minor, major))
        tr = [macro_f1(y_tr, np.where(ptr, a, b)) for a, b in cand]
        a, b = cand[int(np.argmax(tr))]
        s = macro_f1(y_te, np.where(pte, a, b))
        if s > best:
            best, best_field = s, label
    return best, best_field


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {"data_root": str(root), "permutations": args.permutations,
              "seed": args.seed,
              "procedure": "labels permuted over all flows, partition "
                           "boundaries and field presence held fixed, the "
                           "maximum over fields recomputed each time",
              "datasets": {}}

    for name in args.datasets:
        p = root / name
        if not p.is_dir():
            raise RuntimeError("%s is missing under %s" % (name, root))

        ys = {s: np.load(p / ("y_%s.npy" % s)) for s in SPLITS}
        metas = {s: pd.read_csv(p / ("meta_%s.csv" % s), low_memory=False)
                 for s in SPLITS}
        n_tr, n_va = len(ys["train"]), len(ys["val"])
        y_all = np.concatenate([ys[s] for s in SPLITS])
        allm = pd.concat([metas[s] for s in SPLITS], ignore_index=True)

        # Fields whose presence is not constant on either partition.
        pres_tr, pres_te = {}, {}
        for col, label in FIELDS:
            if col not in allm.columns:
                continue
            pr = allm[col].astype(str).apply(is_valid).values
            ptr, pte = pr[:n_tr], pr[n_tr + n_va:]
            if ptr.all() or (~ptr).all() or pte.all() or (~pte).all():
                continue
            pres_tr[label], pres_te[label] = ptr, pte
        if not pres_tr:
            report["datasets"][name] = {
                "n_candidate_fields": 0,
                "note": "every field is constant, so no rule can be formed"}
            print("== %s ==  no usable field" % name)
            continue

        u, c = np.unique(ys["test"], return_counts=True)
        major, minor = int(u[np.argmax(c)]), int(u[np.argmin(c)])
        obs, obs_field = best_availability(
            pres_tr, pres_te, ys["train"], ys["test"], major, minor)

        rng = np.random.default_rng(args.seed)
        null = np.empty(args.permutations, dtype=float)
        for i in range(args.permutations):
            yp = rng.permutation(y_all)
            null[i] = best_availability(
                pres_tr, pres_te, yp[:n_tr], yp[n_tr + n_va:], major, minor)[0]

        pval = float((null >= obs).mean())
        report["datasets"][name] = {
            "n_candidate_fields": len(pres_tr),
            "observed_best_field": obs_field,
            "observed_macro_f1": round(obs, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_std": round(float(null.std()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4),
            "p_value": round(pval, 4),
            "exceeds_null_p95": bool(obs > np.percentile(null, 95))}
        print("== %s ==  %s observed %.4f | null %.4f+-%.4f p95 %.4f | p=%.4f"
              % (name, obs_field, obs, null.mean(), null.std(),
                 np.percentile(null, 95), pval))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
