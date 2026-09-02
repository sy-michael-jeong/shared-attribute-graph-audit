# -*- coding: utf-8 -*-
"""Revert extractor-filled synthetic tokens in one column of meta_*.csv to the missing marker.

When a value cannot be read the extractor does not leave the cell empty; it
writes a string such as 'unknown'. That string is not in the graph builder's
MISSING_TOKENS, so flows whose value could not be read are judged to "share a
value" and get connected. The resulting relation links a parsing-failure
pattern, not TLS values.

This script replaces those tokens with '-', which the builder treats as
missing. It is the control that shows what remains when a relation depends
only on values that were actually read: if the result holds, the finding does
not rest on the token; if it collapses, that collapse is the result.

The meta CSV is rewritten in place, so always pass a copy; the original root
names are refused.

Usage:
    cp -r data/processed_deg2_bccc10 data/processed_deg2_bccc10_nounk
    python -u mask_synthetic_token.py \
        --data data/processed_deg2_bccc10_nounk --datasets bccc_dohbrw \
        --column cert_validity_bucket --tokens unknown invalid_range
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

# One of the markers the builder reads as missing, taken from common.MISSING_TOKENS.
MISSING = "-"

# Roots read by other experiments. Writing here would break their
# reproducibility, so they are refused; run on a copy only.
PROTECTED = {
    "processed",
    "processed_time_strat",
    "processed_deg2",
    "processed_deg2_bccc10",
    "processed_dst_ip_disjoint",
    "processed_random_v2",
    "processed_degree4",
    "processed_time_strat_remap",
}

PARTS = ("train", "val", "test")


def refuse_if_source(root: Path) -> None:
    if root.name in PROTECTED:
        raise SystemExit(
            "'%s' is a root read by other experiments. The meta CSV is rewritten in place,\n"
            "so make a copy and pass that path:\n"
            "    cp -r data/%s data/%s_nounk" % (root.name, root.name, root.name))


def run(root: Path, name: str, column: str, tokens: list) -> None:
    d = root / name
    paths = [d / ("meta_%s.csv" % s) for s in PARTS]
    absent = [str(p) for p in paths if not p.exists()]
    if absent:
        raise SystemExit("missing files: " + str(absent))

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    for p, f in zip(paths, frames):
        if column not in f.columns:
            raise SystemExit(
                "'%s' has no column '%s'. Available: %s"
                % (p, column, str(list(f.columns))))

    joined = pd.concat(frames, ignore_index=True)
    before = joined[column].astype(str).str.strip().value_counts()

    if MISSING in [str(t).strip() for t in tokens]:
        raise SystemExit("the token to revert equals the missing marker")
    hit = {t: int(before.get(t, 0)) for t in tokens}
    if not any(hit.values()):
        raise SystemExit(
            "column '%s' contains none of %s. Top values: %s"
            % (column, str(tokens), str(before.head(8).to_dict())))

    written = 0
    for p, f in zip(paths, frames):
        col = f[column].astype(str).str.strip()
        f[column] = col.where(~col.isin(tokens), MISSING)
        f.to_csv(p, index=False)
        written += len(f)

    after = pd.concat(frames, ignore_index=True)[column].astype(str).str.strip()
    n_missing = int(after.eq(MISSING).sum())
    n_value = written - n_missing

    print("  [%s] %s, flow %s" % (name, column, f"{written:,}"))
    for t, c in hit.items():
        print("      '%s' -> '%s'  %s" % (t, MISSING, f"{c:,}"))
    print("      flows now missing %s, flows keeping a value %s"
          % (f"{n_missing:,}", f"{n_value:,}"))
    print("      distinct values remaining %d" % after[after != MISSING].nunique())


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True,
                    help="processed data root; pass a copy, not the original")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--column", required=True,
                    help="column name in the meta CSV, e.g. cert_validity_bucket")
    ap.add_argument("--tokens", nargs="+", required=True,
                    help="strings to revert to missing, e.g. unknown invalid_range")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit("no such path: " + str(root))
    refuse_if_source(root)

    for name in args.datasets:
        run(root, name, args.column, list(args.tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
