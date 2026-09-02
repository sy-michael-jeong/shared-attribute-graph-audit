# -*- coding: utf-8 -*-
"""Class and campaign composition of each dataset (Sec. 3.1, 3.4, 6.5).

Every count the paper quotes about how a dataset is made up is produced here,
so that a reader can check a stated flow count without rebuilding a graph.
Three groups of numbers are reported.

  binary composition   flows per binary class, overall and per partition, and
                       the majority-class macro-F1 that follows from the test
                       partition alone
  campaign drift       flows per original fine-grained category and how each
                       category is distributed over the partitions, which
                       gives the open-set rate, that is the share of test
                       attacks whose category never appears in training
  field by class       coverage of each relation field within each class,
                       which is where a coverage asymmetry shows up

The fine-grained category column is located by name. Datasets that ship no
such column report the binary composition only.

Usage:
    python dataset_composition.py --data data/processed_time_strat \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/composition/order_preserving.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RELATION_COLUMN, SPLITS, is_valid, macro_f1

# Column names the adapters use for the original fine-grained category.
CATEGORY_COLUMNS = ("traffic_category", "attack_category", "category",
                    "label_original", "orig_label", "app", "application",
                    "class_name", "fine_label")


def majority_baseline(y_test: np.ndarray, n_classes: int) -> float:
    """Macro-F1 of always predicting the most frequent test class."""
    maj = int(np.bincount(y_test, minlength=n_classes).argmax())
    return round(macro_f1(y_test, np.full_like(y_test, maj), n_classes), 4)


def binary_composition(y: dict, label_map: dict) -> dict:
    inv = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    allv = np.concatenate([y[s] for s in SPLITS])
    out = {"n_classes": n_classes, "n_flows": int(len(allv)),
           "labels": {inv.get(c, str(c)): int((allv == c).sum())
                      for c in range(n_classes)},
           "by_split": {}, "majority_macro_f1_test": majority_baseline(
               y["test"], n_classes)}
    for s in SPLITS:
        out["by_split"][s] = {"n": int(len(y[s])),
                              **{inv.get(c, str(c)): int((y[s] == c).sum())
                                 for c in range(n_classes)}}
    minor = min(out["labels"], key=lambda k: out["labels"][k])
    out["minority_label"] = minor
    out["minority_rate"] = round(out["labels"][minor] / len(allv), 4)
    return out


def campaign_drift(meta: dict, y: dict, col: str, label_map: dict) -> dict:
    """Per-category flow counts and the resulting open-set rate."""
    inv = {v: k for k, v in label_map.items()}
    benign = None
    for name, code in label_map.items():
        if str(name).lower() in ("benign", "normal", "0", "nonvpn", "non-vpn"):
            benign = code
            break

    rows = []
    for s in SPLITS:
        df = pd.DataFrame({"cat": meta[s][col].astype(str).values,
                           "y": y[s], "split": s})
        rows.append(df)
    allrows = pd.concat(rows, ignore_index=True)

    cats = {}
    for cat, g in allrows.groupby("cat"):
        per = {s: int((g["split"] == s).sum()) for s in SPLITS}
        cats[cat] = {"total": int(len(g)), **per,
                     "label": inv.get(int(g["y"].mode().iat[0]), "?"),
                     "train_only": per["train"] > 0 and per["test"] == 0,
                     "test_only": per["test"] > 0 and per["train"] == 0}

    out = {"category_column": col, "n_categories": len(cats),
           "categories": dict(sorted(cats.items(),
                                     key=lambda kv: -kv[1]["total"]))}
    if benign is not None:
        atk = allrows[allrows["y"] != benign]
        seen = set(atk[atk["split"] == "train"]["cat"].unique())
        te = atk[atk["split"] == "test"]
        if len(te):
            out["test_attack_flows"] = int(len(te))
            out["open_set_test_attack_flows"] = int((~te["cat"].isin(seen)).sum())
            out["open_set_rate"] = round(
                float((~te["cat"].isin(seen)).mean()), 4)
    return out


def field_by_class(meta: dict, y: dict, label_map: dict) -> dict:
    inv = {v: k for k, v in label_map.items()}
    allm = pd.concat([meta[s] for s in SPLITS], ignore_index=True)
    ally = np.concatenate([y[s] for s in SPLITS])
    out = {}
    for rel, col in RELATION_COLUMN.items():
        if col not in allm.columns or rel == "via_timebin":
            continue
        ok = allm[col].astype(str).map(is_valid).values
        out[col] = {"coverage_overall": round(float(ok.mean()), 6)}
        for c in sorted(set(ally.tolist())):
            m = ally == c
            out[col][inv.get(c, str(c))] = {
                "n_flows": int(m.sum()),
                "n_populated": int(ok[m].sum()),
                "coverage": round(float(ok[m].mean()), 6)}
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {"data_root": str(root), "datasets": {}}
    for ds in args.datasets:
        d = root / ds
        if not d.is_dir():
            raise RuntimeError("%s is missing under %s" % (ds, root))
        y = {s: np.load(d / ("y_%s.npy" % s)) for s in SPLITS}
        meta = {s: pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
                for s in SPLITS}
        label_map = json.load(open(d / "label_map.json"))

        node = {"binary": binary_composition(y, label_map),
                "field_by_class": field_by_class(meta, y, label_map)}
        # A column that survived preprocessing but holds nothing is not a
        # category column. frame_to_xy fills an absent metadata column with an
        # empty string, so presence alone does not mean the dataset ships one.
        col = next((c for c in CATEGORY_COLUMNS
                    if c in meta["train"].columns
                    and meta["train"][c].astype(str).str.strip().ne("").any()),
                   None)
        node["drift"] = campaign_drift(meta, y, col, label_map) if col else \
            {"category_column": None,
             "note": "no fine-grained category column in the metadata table"}
        report["datasets"][ds] = node

        b = node["binary"]
        print("== %s ==  %d flows, %s, minority %s %.4f, majority macro-F1 %.4f"
              % (ds, b["n_flows"], b["labels"], b["minority_label"],
                 b["minority_rate"], b["majority_macro_f1_test"]))
        if node["drift"].get("open_set_rate") is not None:
            print("   open-set test attacks %d / %d = %.4f"
                  % (node["drift"]["open_set_test_attack_flows"],
                     node["drift"]["test_attack_flows"],
                     node["drift"]["open_set_rate"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
