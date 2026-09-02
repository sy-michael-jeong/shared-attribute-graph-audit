# -*- coding: utf-8 -*-
"""Effect of an index-like feature column on HIKARI (Sec. 4.4).

The released HIKARI CSV carries row-number columns, 'Unnamed: 0' and
'Unnamed: 0.1'. Row position tracks the capture schedule, so a pipeline that
keeps such a column as a numeric feature can score on the schedule rather than
on traffic behaviour. Every result in this paper is computed with those columns
removed, which is also what `datasets.py` does when it builds HIKARI.

Measuring what the column is worth therefore requires the raw table, because
the processed arrays do not carry it. This script reads the CSV, builds
the feature matrix twice under the same split, and fits HGB on each.

  keep    the index-like columns stay in the feature matrix
  drop    they are removed, matching every other result in the paper

Both fit partitions are reported, because the column can be worth different
amounts to a model that sees the validation rows and one that does not.
Spreads are population standard deviations over the seeds, as everywhere else
in this artifact.

Usage:
    python check_index_feature.py --raw data/raw/hikari \
        --split random --out results/feature_audit/hikari_random.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics
from datasets import split_random, split_time_stratified

SEEDS = [41, 42, 43, 44, 45]
LABEL_COL = "Label"
NON_FEATURE = ["uid", "flow_id", "Label", "traffic_category",
               "originh", "responh", "src_ip", "dst_ip"]


def build_features(df: pd.DataFrame, keep_index_cols: bool, extra_drop=()):
    """Feature matrix built the way datasets.py builds it, index columns aside.

    `extra_drop` removes further columns by name. The released CSV carries two
    kinds of column that describe where a flow sat rather than what it carried:
    the row numbers, and the two port numbers. `datasets.py` renames
    `originh`/`responh` to `src_ip`/`dst_ip` and so moves them out of the
    feature matrix, but `originp`/`responp` are not renamed and stay in it. So
    the shipped HIKARI feature matrix begins with the two port numbers, and the
    same one-column-out measurement that was used for the row numbers can be
    used for them.
    """
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    drop = [c for c in NON_FEATURE if c in df.columns]
    if not keep_index_cols:
        drop += unnamed
    drop += [c for c in extra_drop if c in df.columns]
    X_df = df.drop(columns=drop).select_dtypes(include=[np.number]).fillna(0.0)
    return X_df.values.astype(np.float32), X_df.columns.tolist()


def fit_hgb(X_tr, y_tr, X_te, y_te, seeds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    vals = []
    for s in seeds:
        # The HGB definition of the paper: a fixed 400 iterations with the
        # library's automatic early stopping disabled.
        m = HistGradientBoostingClassifier(max_iter=400, random_state=s,
                                           class_weight="balanced",
                                           early_stopping=False)
        m.fit(X_tr, y_tr)
        vals.append(classification_metrics(y_te, m.predict(X_te))["macro_f1"])
    return {"macro_f1_mean": round(float(np.mean(vals)), 4),
            "macro_f1_std": round(float(np.std(vals)), 4),
            "per_seed": [round(v, 4) for v in vals]}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--raw", default="data/raw/hikari",
                    help="directory holding ALLFLOWMETER_HIKARI2021.csv")
    ap.add_argument("--split", choices=["random", "order_preserving"],
                    default="random")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ports", nargs="+", default=["originp", "responp"],
                    help="port column names, removed together in the third configuration")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    csv = Path(args.raw) / "ALLFLOWMETER_HIKARI2021.csv"
    if not csv.exists():
        raise RuntimeError("%s not found. Run datasets.py first so that the "
                           "CSV is downloaded." % csv)
    df = pd.read_csv(csv)
    y = df[LABEL_COL].astype(int).values
    print("rows = %d | label counts = %s" % (len(df), np.bincount(y).tolist()))

    if args.split == "random":
        idx_tr, idx_va, idx_te = split_random(
            len(df), y, args.test_size, args.val_size, args.split_seed)
    else:
        ts = np.arange(len(df), dtype=float)
        idx_tr, idx_va, idx_te = split_time_stratified(
            ts, y, args.test_size, args.val_size)
    print("split %s -> train %d / val %d / test %d"
          % (args.split, len(idx_tr), len(idx_va), len(idx_te)))

    report = {"csv": str(csv), "split": args.split,
              "split_seed": args.split_seed, "seeds": list(args.seeds),
              "split_sizes": {"train": int(len(idx_tr)),
                              "val": int(len(idx_va)),
                              "test": int(len(idx_te))},
              "runs": {}}

    # Third configuration: with the row number already removed, also drop the two
    # port columns. They are the same kind of column as the row number: they say
    # where a flow sat, not what it carried.
    trval = np.concatenate([idx_tr, idx_va])
    variants = (("keep_index_col", True, ()),
                ("drop_index_col", False, ()),
                ("drop_index_and_ports", False, tuple(args.ports)))
    for label, keep, extra in variants:
        X, names = build_features(df, keep, extra)
        kept = [n for n in names if str(n).startswith("Unnamed")]
        ports_kept = [n for n in names if n in args.ports]
        print("\n[%s] features = %d | index-like kept = %s | ports kept = %s"
              % (label, len(names), kept, ports_kept))
        for fit, tr in (("fit_trainval", trval), ("fit_train", idx_tr)):
            r = fit_hgb(X[tr], y[tr], X[idx_te], y[idx_te], args.seeds)
            r["n_features"] = len(names)
            r["index_like_columns"] = kept
            r["port_columns"] = ports_kept
            report["runs"]["%s__%s" % (label, fit)] = r
            print("  HGB %-13s %.4f +- %.4f  %s"
                  % (fit, r["macro_f1_mean"], r["macro_f1_std"], r["per_seed"]))

    for fit in ("fit_trainval", "fit_train"):
        d = (report["runs"]["keep_index_col__%s" % fit]["macro_f1_mean"]
             - report["runs"]["drop_index_col__%s" % fit]["macro_f1_mean"])
        report["inflation__%s" % fit] = round(d, 4)
        print("\ninflation from the index columns (%s): %+.4f" % (fit, d))
        p = (report["runs"]["drop_index_col__%s" % fit]["macro_f1_mean"]
             - report["runs"]["drop_index_and_ports__%s" % fit]["macro_f1_mean"])
        report["port_inflation__%s" % fit] = round(p, 4)
        print("inflation from the port columns  (%s): %+.4f" % (fit, p))
    report["note"] = (
        "Two kinds of column describe where a flow sat rather than what it "
        "carried: the row numbers and the two port numbers. datasets.py moves "
        "the host addresses out of the feature matrix by renaming them and "
        "removes the row numbers explicitly, but the ports are neither renamed "
        "nor removed, so they are features in every reported HIKARI result.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
