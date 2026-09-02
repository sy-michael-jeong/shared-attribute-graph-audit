# -*- coding: utf-8 -*-
"""Same-source tabular baseline, matched to what the graph holds.

The graph builds its edges from TLS and host metadata while the tabular model
sees only flow statistics, so a comparison between them mixes the effect of
the graph with the effect of that metadata. This script hands the same metadata
to gradient boosting as tabular features.

Two things keep the comparison matched.

  TimeBin enters as the 300-second bucket identifier the relation itself uses,
  not as a raw timestamp, so the tabular model sees the same granularity.

  The relation set is the set the graph actually holds for that dataset, taken
  from the edge files rather than from the configuration, so the tabular model
  receives the same metadata and not a superset of it.

Four values are derived per flow for each relation.

  present       one when the field is populated
  train_freq    how often the value occurs in the training partition, log1p
  seen          one when the value occurs in the training partition
  group_size    size of the group that shares the value, log1p

present and group_size are computed over all three partitions, matching the
transductive scope of the graph. train_freq and seen are computed out of fold
inside the training partition. A value observed in a training row is by
construction present in the training statistics, so using them directly would
leave both features constant on the partition the model is fitted on and let
them vary only at test time. No feature reads the label, so the encoding
carries no target leakage.

Both relation sets and both fit partitions are run, giving four combinations.

    relations = selected | full
    fit       = train    | trainval

Usage:
    python same_info_matched.py \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_time_strat --out runs/same_info_matched
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ALL_RELATIONS, SELECTED, SPLITS, is_valid, macro_f1,
                    minority_f1, relation_values)


def build_features(allm, relations, n_tr, bin_seconds, n_folds=5, seed=0):
    """Turn the relation metadata into numeric features.

    train_freq and seen ask whether a value was observed in the training
    partition. Computing them for training rows from the full training
    statistics makes both degenerate, because a valid value in a training row
    is by construction present in those statistics. seen would equal present
    and train_freq would never be zero, so the two features would look constant
    on the partition the model is fitted on and vary only at test time.

    Training rows are therefore computed out of fold. The training partition is
    split into n_folds, and the rows of each fold are filled from the
    statistics of the remaining folds. Validation and test rows use the full
    training statistics. Both features then carry the same meaning in all three
    partitions.
    """
    cols, names, used = [], [], []
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, n_folds, size=n_tr)

    for rel in relations:
        s = relation_values(allm, rel, bin_seconds)
        if s is None:
            continue
        valid = s.apply(is_valid).values
        if not valid.any():
            continue

        n = len(s)
        train_freq = np.zeros(n, dtype=np.float32)
        seen = np.zeros(n, dtype=np.float32)

        # training rows, out of fold
        tr = s.iloc[:n_tr]
        tr_valid = valid[:n_tr]
        for k in range(n_folds):
            other = (fold != k) & tr_valid
            freq_k = tr[other].value_counts()
            idx = np.where((fold == k) & tr_valid)[0]
            if len(idx) == 0:
                continue
            v = tr.iloc[idx]
            train_freq[idx] = v.map(freq_k).fillna(0.0).values.astype(np.float32)
            seen[idx] = v.map(freq_k).notna().values.astype(np.float32)

        # val and test rows, full training statistics
        freq_all = tr[tr_valid].value_counts()
        rest = np.arange(n_tr, n)
        if len(rest):
            v = s.iloc[rest]
            rv = valid[n_tr:]
            train_freq[rest] = (v.map(freq_all).fillna(0.0).values.astype(np.float32) * rv)
            seen[rest] = (v.map(freq_all).notna().values & rv).astype(np.float32)

        present = valid.astype(np.float32)
        grp = s[valid].value_counts()
        group_size = s.map(grp).fillna(0.0).values.astype(np.float32) * present

        cols += [present, np.log1p(train_freq), seen, np.log1p(group_size)]
        tag = rel.replace("via_", "")
        names += [tag + x for x in ("_present", "_train_freq", "_seen", "_group_size")]
        used.append(rel)
    if not cols:
        return None, [], []
    return np.vstack(cols).T.astype(np.float32), names, used


def run_hgb(Xtr, ytr, Xte, yte, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, random_state=seed,
                                       class_weight="balanced",
                                       early_stopping=False)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return macro_f1(yte, p), minority_f1(yte, p)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--timebin-seconds", type=float, default=300.0)
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in args.datasets:
        p = root / name
        if not p.is_dir():
            print("[skip] %s" % p)
            continue
        Xs = [np.load(p / ("X_%s.npy" % s)) for s in SPLITS]
        ys = [np.load(p / ("y_%s.npy" % s)) for s in SPLITS]
        metas = [pd.read_csv(p / ("meta_%s.csv" % s), low_memory=False) for s in SPLITS]
        allm = pd.concat(metas, ignore_index=True)
        n_tr, n_va = len(Xs[0]), len(Xs[1])

        # full = the relations the graph actually holds. Handing the tabular
        # model a relation with no edges would break the match.
        pool = sorted(f.stem.replace("hin_edges_", "") for f in p.glob("hin_edges_*.npy"))
        pool = [r for r in ALL_RELATIONS if r in pool]
        if not pool:
            print("[warn] %s: no hin_edges_*.npy, falling back to all relations" % name)
            pool = ALL_RELATIONS
        M_full, names_full, used_full = build_features(
            allm, pool, n_tr, args.timebin_seconds)
        sel = SELECTED[name]
        M_sel, names_sel, used_sel = build_features(
            allm, sel, n_tr, args.timebin_seconds)

        res = {"relations_full": used_full, "relations_selected": used_sel,
               "n_features_full": len(names_full), "n_features_selected": len(names_sel),
               "feature_names_selected": names_sel, "runs": {}}
        print("\n=== %s" % name)
        print("  full     : %d relations, %d features" % (len(used_full), len(names_full)))
        print("  selected : %s (%d features)" % (used_sel, len(names_sel)))

        variants = {}
        for rtag, M in (("selected", M_sel), ("full", M_full)):
            if M is None:
                continue
            variants[rtag] = [M[:n_tr], M[n_tr:n_tr + n_va], M[n_tr + n_va:]]

        for ftag in ("train", "trainval"):
            if ftag == "train":
                sl = slice(0, n_tr)
                yfit = ys[0]
            else:
                sl = slice(0, n_tr + n_va)
                yfit = np.concatenate([ys[0], ys[1]])

            # the flow-only baseline uses the same fit partition
            Xfit = np.vstack(Xs[:1] if ftag == "train" else Xs[:2])
            combos = [("flow_only", Xfit, Xs[2])]
            for rtag, Ms in variants.items():
                Mall = np.vstack(Ms[:1] if ftag == "train" else Ms[:2])
                combos.append(("meta_%s" % rtag, Mall, Ms[2]))
                combos.append(("flow_plus_meta_%s" % rtag,
                               np.hstack([Xfit, Mall]),
                               np.hstack([Xs[2], Ms[2]])))

            for tag, A, B in combos:
                ms, mn = [], []
                for sd in args.seeds:
                    a, b = run_hgb(A, yfit, B, ys[2], sd)
                    ms.append(a); mn.append(b)
                key = "%s__fit_%s" % (tag, ftag)
                res["runs"][key] = {
                    "macro_f1_mean": round(float(np.mean(ms)), 4),
                    "macro_f1_std": round(float(np.std(ms)), 4),
                    "minority_f1_mean": round(float(np.mean(mn)), 4),
                    "per_seed": [round(x, 4) for x in ms]}
                print("    %-34s macro %.4f ± %.4f  minority %.4f"
                      % (key, np.mean(ms), np.std(ms), np.mean(mn)))
        report[name] = res

    json.dump(report, open(out / "tabular_summary.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "tabular_summary.json"))


if __name__ == "__main__":
    main()
