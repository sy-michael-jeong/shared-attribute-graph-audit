# -*- coding: utf-8 -*-
"""Recompute the summary blocks from the seed-wise records.

Every multi-seed summary carries two things: `per_seed`, which is the record of
what each run produced, and an aggregate block holding mean, std and a rounded
copy of the scores. The aggregate is derived; `per_seed` is the evidence.

Two problems make the derived block disagree with the paper.

  definition   Most files store the sample standard deviation while the paper
               reports the population one. Over five seeds the two differ by
               twelve percent: HAN on ISCX-VPN gives 0.0375 and 0.0336, and the
               paper reports 0.0336. A reader who opens the artifact sees a
               number the paper does not contain.

  rounding     The `values` copy is rounded to four places, so recomputing from
               it is not the same as recomputing from `per_seed`. The last digit
               moves. `per_seed` is therefore the only correct input.

This script rewrites every aggregate block from `per_seed`, storing both
definitions under `std` (population, matching the paper) and `std_sample`. It
does not touch `per_seed`, so it can be run again and will produce the same
file. Nothing is re-trained.

    python reaggregate.py --results results
    python reaggregate.py --results results --check    # report, write nothing
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

METRICS = ("macro_f1", "minority_f1", "weighted_f1")


def agg(values):
    """Aggregate block for one metric. Both definitions of the spread."""
    ok = [v for v in values if v >= 0]
    if not ok:
        return {"mean": -1.0, "std": 0.0, "std_sample": 0.0, "n": 0,
                "values": []}
    return {"mean": round(st.fmean(ok), 4),
            "std": round(st.pstdev(ok), 4) if len(ok) > 1 else 0.0,
            "std_sample": round(st.stdev(ok), 4) if len(ok) > 1 else 0.0,
            "n": len(ok),
            "values": [round(v, 4) for v in ok]}


def visit(node, changes, path=""):
    """Rewrite every block that sits next to a `per_seed` list."""
    if isinstance(node, list):
        for i, x in enumerate(node):
            visit(x, changes, "%s[%d]" % (path, i))
        return
    if not isinstance(node, dict):
        return

    ps = node.get("per_seed")
    if isinstance(ps, list) and ps and isinstance(ps[0], dict):
        for m in METRICS:
            if m not in node or not isinstance(node[m], dict):
                continue
            vals = [float(p[m]) for p in ps if m in p]
            if not vals:
                continue
            new = agg(vals)
            old = node[m]
            if abs(old.get("std", 0.0) - new["std"]) > 1e-9:
                changes.append((path + "/" + m, old.get("std"), new["std"],
                                old.get("mean"), new["mean"]))
            node[m] = new

    for k, v in node.items():
        if k != "per_seed":
            visit(v, changes, path + "/" + str(k))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--results", default="results")
    ap.add_argument("--check", action="store_true",
                    help="report the differences and write nothing")
    args = ap.parse_args()

    root = Path(args.results)
    n_files = n_changed = n_blocks = 0
    for f in sorted(root.rglob("*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        n_files += 1
        changes = []
        visit(d, changes)
        if not changes:
            continue
        n_changed += 1
        n_blocks += len(changes)
        print("%s  (%d blocks)" % (f.relative_to(root), len(changes)))
        for p, o, n, om, nm in changes[:3]:
            note = "" if om == nm else "   mean %s -> %s" % (om, nm)
            print("    %-52s std %s -> %s%s" % (p[-52:], o, n, note))
        if not args.check:
            json.dump(d, open(f, "w", encoding="utf-8"), indent=2)

    print("\n%d files read, %d rewritten, %d blocks changed%s"
          % (n_files, n_changed, n_blocks,
             "  (check only, nothing written)" if args.check else ""))


if __name__ == "__main__":
    main()
