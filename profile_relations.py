# Relation profiling (paper 3.3): coverage, cardinality, top-10 dominance,
# class MI (nats, binary task label), train-test value overlap.
#
# The shipped results/profiling/relation_profiling.json was produced under the
# conventional stratified random split, which is what Table 3 reports. Running
# this script against data/processed_deg2 measures the audited split instead,
# so the `protocol` field records which one a file holds.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALL_RELATIONS, RELATION_COLUMN, is_valid

FIELDS = [RELATION_COLUMN[r] for r in ALL_RELATIONS if r != "via_timebin"]


valid = is_valid


def class_mi(values, labels):
    pair = pd.DataFrame({"v": values, "y": labels})
    pair = pair[pair["v"].apply(valid)]
    if len(pair) == 0:
        return 0.0
    N = len(pair)
    pxy = pair.groupby(["v", "y"]).size() / N
    px = pair["v"].value_counts() / N
    py = pair["y"].value_counts() / N
    mi = 0.0
    for (v, y), p in pxy.items():
        d = px[v] * py[y]
        if d > 0 and p > 0:
            mi += p * np.log(p / d)
    return float(mi)


def profile(name, data_root):
    d = data_root / name
    metas = [pd.read_csv(d / ("meta_%s.csv" % p)) for p in ("train", "val", "test")]
    ys = [np.load(d / ("y_%s.npy" % p)) for p in ("train", "val", "test")]
    meta = pd.concat(metas, ignore_index=True)
    y = np.concatenate(ys)
    print("\n== %s (%d flows, %d classes)" % (name, len(meta), len(np.unique(y))))
    print("%-24s %10s %10s %10s %10s %10s"
          % ("field", "coverage", "distinct", "top10_dom", "class_MI", "tr/te_ovl"))

    out = {"dataset": name, "n_flows": len(meta), "fields": {}}
    for f in FIELDS:
        if f not in meta.columns:
            continue
        col = meta[f].astype(str)
        mask = col.apply(valid)
        cov = mask.mean()
        vals = col[mask]
        distinct = vals.nunique()
        dom = vals.value_counts().head(10).sum() / len(vals) if len(vals) else 0.0
        mi = class_mi(col.values, y)
        tr = set(v for v in metas[0][f].astype(str) if valid(v))
        te = set(v for v in metas[2][f].astype(str) if valid(v))
        ovl = len(tr & te) / len(te) if te else 0.0
        print("%-24s %10.4f %10d %10.4f %10.4f %10.4f"
              % (f, cov, distinct, dom, mi, ovl))
        out["fields"][f] = {"coverage": float(cov), "distinct": int(distinct),
                            "top10_dominance": float(dom),
                            "class_mi_nats": float(mi),
                            "train_test_overlap": float(ovl)}
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/profiling/relation_profiling.json")
    ap.add_argument("--protocol", default=None,
                    help="how the split was drawn; recorded with the numbers")
    args = ap.parse_args()

    results = {n: profile(n, Path(args.data)) for n in args.datasets}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The split protocol and the label the mutual information is taken against
    # are recorded next to the numbers. A coverage or MI value read without
    # them says nothing, and the two were reported inconsistently once.
    json.dump({"protocol": args.protocol or Path(args.data).name,
               "mi_label": "binary task label",
               "datasets": results}, open(out, "w"), indent=2)
    rows = [{"dataset": ds, "field": f, **fr}
            for ds, r in results.items() for f, fr in r["fields"].items()]
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
