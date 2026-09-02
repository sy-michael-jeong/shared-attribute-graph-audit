# Post-split audits (paper 4.3-4.4 and 5.4):
#   audit.py split    - class coverage, minority counts, identifier overlap
#   audit.py drift    - attack-category composition per partition (HIKARI)
#   audit.py indexcol - sensitivity of scores to index-like feature columns
#   audit.py features - the feature matrix itself, before any model is run
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from common import is_valid
from datasets import split_time_stratified


def audit_split(args):
    d = Path(args.data) / args.dataset
    ys = {p: np.load(d / ("y_%s.npy" % p)) for p in ("train", "val", "test")}
    metas = {p: pd.read_csv(d / ("meta_%s.csv" % p)) for p in ("train", "val", "test")}
    label_map = json.load(open(d / "label_map.json"))
    names = {v: k for k, v in label_map.items()}

    print("== class coverage and minority counts")
    ok = True
    for p, y in ys.items():
        vals, cnts = np.unique(y, return_counts=True)
        missing = set(label_map.values()) - set(vals.tolist())
        if missing:
            ok = False
            print("  [FAIL] %s is missing classes: %s"
                  % (p, [names[m] for m in missing]))
        print("  %-6s %s" % (p, {names[v]: int(c) for v, c in zip(vals, cnts)}))
    print("  coverage check:", "pass" if ok else "FAIL")

    print("== identifier overlap on the realized split")
    for col in ("sni", "ja3", "src_ip", "dst_ip", "cert_subject"):
        if col not in metas["train"].columns:
            continue
        tr = set(v for v in metas["train"][col].astype(str) if is_valid(v))
        te = set(v for v in metas["test"][col].astype(str) if is_valid(v))
        if te:
            print("  %-14s tr/te overlap = %.4f" % (col, len(tr & te) / len(te)))


def audit_drift(args):
    df = pd.read_csv(args.csv)
    y = df["Label"].astype(int).values
    # the released CSV has no timestamps, so row order is the ordering axis
    ts = np.arange(len(df), dtype=float)
    idx_tr, idx_va, idx_te = split_time_stratified(ts, y, 0.2, 0.1)

    cat = df["traffic_category"].astype(str)
    table = pd.DataFrame({p: cat.iloc[i].value_counts()
                          for p, i in (("train", idx_tr), ("val", idx_va), ("test", idx_te))})
    table = table.fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    print(table.to_string())

    tr_types = set(cat.iloc[idx_tr][y[idx_tr] == 1])
    te_types = set(cat.iloc[idx_te][y[idx_te] == 1])
    only_test = te_types - tr_types
    print("\ntest-only attack categories:", sorted(only_test) or "none")
    n_att = int((y[idx_te] == 1).sum())
    n_open = int(cat.iloc[idx_te].isin(only_test)[y[idx_te] == 1].sum()) \
        if only_test else 0
    if only_test:
        print("open-set share of test attacks: %d / %d = %.3f"
              % (n_open, n_att, n_open / n_att))

    # The measurement is written, not only printed. A printed table cannot be
    # checked against the paper months later, and the artifact ships the file
    # the paper cites rather than a transcript of a terminal.
    if getattr(args, "out", None):
        report = {
            "csv": args.csv,
            "n_rows": int(len(df)),
            "timestamp_column": None,
            "test_size": 0.2,
            "val_size": 0.1,
            "split_sizes": {"train": int(len(idx_tr)), "val": int(len(idx_va)),
                            "test": int(len(idx_te))},
            "category_by_split": {k: {kk: int(vv) for kk, vv in v.items()}
                                  for k, v in table.to_dict("index").items()},
            "attack_types_train": sorted(tr_types),
            "attack_types_test": sorted(te_types),
            "test_only_attack_types": sorted(only_test),
            "test_attack_flows": n_att,
            "open_set_test_attack_flows": n_open,
            "open_set_rate": round(n_open / n_att, 4) if n_att else 0.0,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=2)
        print("[saved] %s" % out)


def audit_indexcol(args):
    """Index-like columns in an already processed directory.

    HIKARI is the dataset this matters for, and datasets.load_hikari drops
    every `Unnamed` column while loading, so a processed directory built by
    this artifact never carries one. The measurement the paper reports comes
    from the raw table instead, which is what check_index_feature.py reads.
    This subcommand is the check that the drop actually happened.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from models import metrics
    d = Path(args.data) / args.dataset
    feats = json.load(open(d / "feature_names.json"))
    drop = [i for i, f in enumerate(feats) if "Unnamed" in f]
    print("features: %d, index-like: %s" % (len(feats), [feats[i] for i in drop]))
    if not drop:
        print("no index-like column survived preprocessing, which is the "
              "expected outcome. For what such a column is worth, run "
              "check_index_feature.py against the raw table.")
        return
    X = {p: np.load(d / ("X_%s.npy" % p)) for p in ("train", "val", "test")}
    y = {p: np.load(d / ("y_%s.npy" % p)) for p in ("train", "val", "test")}
    keep = [i for i in range(len(feats)) if i not in drop]

    for label, cols in (("with index col", None), ("without index col", keep)):
        Xtr = X["train"] if cols is None else X["train"][:, cols]
        Xva = X["val"] if cols is None else X["val"][:, cols]
        Xte = X["test"] if cols is None else X["test"][:, cols]
        vals = []
        for s in args.seeds:
            m = HistGradientBoostingClassifier(max_iter=400, random_state=s,
                                               class_weight="balanced",
                                               early_stopping=False)
            m.fit(np.vstack([Xtr, Xva]), np.concatenate([y["train"], y["val"]]))
            vals.append(metrics(y["test"], m.predict(Xte))["macro_f1"])
        print("[%s] macro_f1 = %.4f +- %.4f  %s"
              % (label, statistics.fmean(vals),
                 float(np.std(vals)) if len(vals) > 1 else 0.0,
                 [round(v, 4) for v in vals]))


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks, so that ties do not fabricate an ordering."""
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return 0.0 if d == 0 else float((ra * rb).sum() / d)


NAME_PATTERNS = ("unnamed", "index", "row_id", "rowid", "flow_id", "flowid",
                 "no.", "serial", "seq")


def _counter_like(col: np.ndarray) -> bool:
    """Is this column a row counter rather than a measurement?

    Counting distinct values does not answer this. A duration in seconds is
    distinct on almost every row too, and on a small dataset that alone puts it
    above any threshold — ISCX-VPN has 891 flows, so `duration` came back at
    0.9944 and was flagged as an index. It is not one.

    What separates a counter from a measurement is not how many values it has
    but how they are spaced. A counter is integer valued and its sorted values
    step by a constant, because they were assigned in order rather than
    measured. So: integers, almost all distinct, and a near-constant step.
    """
    if not np.all(np.isfinite(col)):
        return False
    if not np.allclose(col, np.round(col), atol=1e-9):
        return False
    u = np.unique(col)
    if len(u) < 8 or len(u) < 0.98 * len(col):
        return False
    # The second half is what separates a counter from any other integer that
    # happens to be distinct. A counter fills its own range: the values run
    # 0..n with at most the gaps left by rows that went to another partition,
    # so distinct count over range is high. Byte counts, ports and integer
    # timestamps are also distinct but scattered across a range far wider than
    # their count, so the same ratio is near zero.
    span = float(u.max() - u.min()) + 1.0
    return len(u) / span >= 0.1


def audit_features(args):
    """The feature matrix itself, before any model is run.

    Sec. 7.5 asks for two things of every feature column, and asking them of a
    trained score is too late: by then a leaked column has already been paid
    for. Both are decidable from the matrix alone.

      1. Is it a row number, or something that behaves like one? A column whose
         values are distinct on almost every row carries the row's identity and
         not the flow's behaviour. HIKARI's two `Unnamed` columns are that, and
         they moved the boosting score from 0.725 to 1.000.
      2. Does it track the timestamp? Time is used here to order the split and
         to cut TimeBin, and it is deliberately kept out of the features. A
         column that reproduces the time ordering puts it back in, and then the
         split axis is inside the model's input.

    A high score here is not proof of leakage. It says the column encodes
    position rather than behaviour, which is the thing that has to be looked at
    before it is trained on.
    """
    report = {"data_root": args.data, "thresholds":
              {"unique_ratio": args.unique_ratio, "abs_spearman": args.rho},
              "datasets": {}}
    for name in args.datasets:
        d = Path(args.data) / name
        feats = json.load(open(d / "feature_names.json"))
        X = np.vstack([np.load(d / ("X_%s.npy" % p))
                       for p in ("train", "val", "test")])
        metas = [pd.read_csv(d / ("meta_%s.csv" % p), low_memory=False)
                 for p in ("train", "val", "test")]
        meta = pd.concat(metas, ignore_index=True)

        ts = None
        if "ts" in meta.columns:
            t = pd.to_numeric(meta["ts"], errors="coerce").values
            if np.isfinite(t).sum() > 0.5 * len(t) and np.nanstd(t) > 0:
                ts = np.nan_to_num(t, nan=float(np.nanmedian(t)))

        n = len(X)
        flagged, rows = [], []
        for j, f in enumerate(feats):
            col = X[:, j].astype(float)
            uniq = int(len(np.unique(col)))
            ratio = uniq / n if n else 0.0
            rho = _spearman(col, ts) if ts is not None else None
            counter = _counter_like(col)
            why = []
            if any(p in f.lower() for p in NAME_PATTERNS):
                why.append("name")
            if counter:
                why.append("counter")
            if rho is not None and abs(rho) >= args.rho:
                why.append("tracks time")
            row = {"feature": f, "unique_ratio": round(ratio, 6),
                   "counter_like": bool(counter),
                   "spearman_with_ts": None if rho is None else round(rho, 4),
                   "flags": why}
            rows.append(row)
            if why:
                flagged.append(row)

        # A dataset with no timestamp is ordered by the row order of the
        # released CSV, and this audit reads a processed directory where that
        # order is already gone. HIKARI is that case, and it is also the
        # dataset whose row-number columns matter — the loader drops them, so
        # a clean result here is a statement about the processed matrix and
        # not about the release. check_index_feature.py reads the raw table.
        report["datasets"][name] = {
            "n_rows": n, "n_features": len(feats),
            "timestamp_available": ts is not None,
            "ordering_axis": "timestamp" if ts is not None else
                             "row order of the released file (not visible here)",
            "flagged": flagged, "columns": rows,
        }
        print("== %s  features %d, rows %d, ordering axis %s"
              % (name, len(feats), n,
                 "timestamp" if ts is not None else "row order (not visible here)"))
        if ts is None:
            print("   the time test cannot run on a processed directory for this "
                  "dataset; check_index_feature.py reads the raw table instead")
        if not flagged:
            print("   no column counts rows or tracks the timestamp")
        for r in flagged:
            print("   [flag] %-28s unique=%.4f  rho=%s  %s"
                  % (r["feature"][:28], r["unique_ratio"],
                     r["spearman_with_ts"], ",".join(r["flags"])))

    # A test that never fires is indistinguishable from a test that cannot
    # fire, and four of these five datasets come back clean. So the detector is
    # run against a column that is a row number by construction, and against
    # one that is a measurement, and both answers are recorded next to the
    # result. Without this the clean result says nothing.
    n_probe, rg = 4000, np.random.default_rng
    probe = {
        "row number, contiguous":
            (np.arange(n_probe, dtype=float), True),
        "row number, 70% of the rows kept":
            (np.sort(rg(4).choice(5714, n_probe, replace=False)).astype(float), True),
        "row number, 20% of the rows kept":
            (np.sort(rg(0).choice(20000, n_probe, replace=False)).astype(float), True),
        "duration in seconds":
            (np.round(rg(1).exponential(3.0, 891), 6), False),
        "packet count":
            (rg(2).integers(1, 400, n_probe).astype(float), False),
        "byte count":
            (np.unique(rg(5).integers(0, 10 ** 8, n_probe)).astype(float), False),
        "timestamp in whole seconds":
            (np.unique(rg(6).integers(0, 10 ** 7, n_probe)).astype(float), False),
        "port number":
            (rg(7).integers(1, 65535, n_probe).astype(float), False),
    }
    report["self_test"] = {}
    print("== detector self-test")
    for label, (col, want) in probe.items():
        got = _counter_like(col)
        report["self_test"][label] = {"expected": want, "detected": bool(got),
                                      "pass": bool(got == want)}
        print("   %-32s expected %-5s detected %-5s %s"
              % (label, want, got, "ok" if got == want else "FAIL"))
    if not all(v["pass"] for v in report["self_test"].values()):
        raise SystemExit("the detector fails its own probe; the audit above "
                         "cannot be read as a clean result")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=2)
        print("[saved] %s" % out)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--dataset", required=True)

    p = sub.add_parser("drift")
    p.add_argument("--csv", required=True,
                   help="path to ALLFLOWMETER_HIKARI2021.csv")
    p.add_argument("--out", default=None,
                   help="write the measurement to this file")

    p = sub.add_parser("indexcol")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--dataset", default="hikari")
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])

    p = sub.add_parser("features")
    p.add_argument("--data", default="data/processed_deg2")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--unique-ratio", type=float, default=0.99,
                   help="above this share of distinct values a column is "
                        "carrying the row's identity, not the flow's behaviour")
    p.add_argument("--rho", type=float, default=0.99,
                   help="above this rank correlation with the timestamp a "
                        "column reproduces the split axis")
    p.add_argument("--out", default=None)

    args = ap.parse_args()
    {"split": audit_split, "drift": audit_drift, "indexcol": audit_indexcol,
     "features": audit_features}[args.cmd](args)


if __name__ == "__main__":
    main()
