# -*- coding: utf-8 -*-
"""How many extractor-filled strings each field carries (Sec. 3.3, 6.2).

`extract_pcap.py` fills a string instead of leaving the field empty when a
value cannot be read: `unknown` when a certificate date fails to parse,
`invalid_range` when the validity period is negative, `Unknown` when a cipher
suite cannot be mapped, `Private_or_Unknown` when an issuer cannot be
normalized. None is in `common.MISSING_TOKENS`, so each counts as a value and
**flows that share the same failure get connected to each other.**

The fifth control reverts those tokens to missing. Which fields make that
control meaningful can be seen here before running it; reverting a field with
nineteen tokens removes thirty-eight edges and nothing more.

Counting only: no model, no GPU.

    python token_census.py --data data/processed_deg2_bccc10 \\
        --datasets bccc_dohbrw \\
        --out results/reference_lines/token_census.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import MISSING_TOKENS, RELATION_COLUMN, SPLITS, is_valid

# Strings the extractor writes to mark a failure. column name -> tokens.
SYNTHETIC = {
    "cert_validity_bucket": ["unknown", "invalid_range"],
    "tls_cipher_group": ["Unknown"],
    "cert_issuer_org": ["Private_or_Unknown", "PublicCA_Other"],
}

FIELDS = [c for r, c in RELATION_COLUMN.items() if r != "via_timebin"]


def load(d: Path):
    metas = [pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
             for s in SPLITS]
    ys = [np.load(d / ("y_%s.npy" % s)) for s in SPLITS]
    return pd.concat(metas, ignore_index=True), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {
        "data_root": str(root),
        "question": "how much of what is counted as a value is a string the "
                    "extractor wrote because it could not read one",
        "missing_tokens": list(MISSING_TOKENS),
        "synthetic_tokens": SYNTHETIC,
        "datasets": {},
    }

    for name in args.datasets:
        meta, y = load(root / name)
        n = len(meta)
        block = {"n_flows": int(n), "fields": {}}
        for col in FIELDS:
            if col not in meta.columns:
                continue
            v = meta[col].astype(str).str.strip()
            valid = np.array([is_valid(x) for x in v.values])
            n_valid = int(valid.sum())
            toks = {}
            for t in SYNTHETIC.get(col, []):
                c = int((v == t).sum())
                toks[t] = {"n_flows": c,
                           "share_of_valid": round(c / n_valid, 4) if n_valid else 0.0,
                           "n_attack": int(((v == t) & (y != 0)).sum())}
            synth = sum(x["n_flows"] for x in toks.values())
            block["fields"][col] = {
                "coverage": round(n_valid / n, 4),
                "n_valid": n_valid,
                "n_missing": int(n - n_valid),
                "synthetic_tokens": toks,
                "n_synthetic": synth,
                # Coverage that remains once the filler strings are reverted to missing.
                "coverage_after_reversal": round((n_valid - synth) / n, 4),
                "share_of_valid_that_is_synthetic":
                    round(synth / n_valid, 4) if n_valid else 0.0}
        report["datasets"][name] = block

        print("== %s  (%d flow)" % (name, n))
        for col, f in sorted(block["fields"].items(),
                             key=lambda kv: -kv[1]["n_synthetic"]):
            if not f["synthetic_tokens"]:
                continue
            print("   %-22s coverage %.4f -> %.4f   filled %d (%.1f%% of valid)"
                  % (col, f["coverage"], f["coverage_after_reversal"],
                     f["n_synthetic"],
                     100 * f["share_of_valid_that_is_synthetic"]))
            for t, x in f["synthetic_tokens"].items():
                print("        %-22s %8d   of which attacks %d" %
                      (t, x["n_flows"], x["n_attack"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
