# -*- coding: utf-8 -*-
"""Whether a group-disjoint split can be built at all (Sec. 4.1).

Guidance on leakage control asks that an identifier value seen in training
never reappear in testing. Whether that is achievable is a property of the
dataset, decided before any model is fitted, and this script decides it.

A group-disjoint split assigns every flow carrying an identifier value to one
partition. Three obstacles are measured for each candidate axis.

  class starvation  a class present in fewer groups than there are partitions
                    must vanish from at least one partition
  group dominance   the share of the axis held by its largest group, which
                    is reported as a diagnostic rather than treated as a
                    blocker, because a large group can be placed in the
                    training partition
  realized split    StratifiedGroupKFold is run when scikit-learn provides it,
                    and the resulting per-partition class counts are reported,
                    so the failure is shown rather than argued
  direct search     whole groups are assigned to the three target partitions
                    many times over, which decides feasibility by construction
                    instead of by one library call

An axis is called feasible when the search finds an assignment in which every
partition keeps every class and no partition size deviates from its target by
more than the tolerance. Flows whose identifier is missing are excluded from
the axis, and their number is reported, because an axis can look feasible only
because most flows drop out.

Usage:
    python group_split_feasibility.py --data data/processed_time_strat \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/group_split/feasibility.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RELATION_COLUMN, SPLITS, is_valid, relation_values

# Axes on which a group-disjoint split is normally attempted.
AXES = ("via_sni", "via_ja3", "via_cert_subject", "via_src_host",
        "via_dst_host", "via_timebin")
FRACTIONS = (0.7, 0.1, 0.2)


def random_group_search(groups: pd.Series, yy: np.ndarray, label_map: dict,
                        tolerance: float, restarts: int, seed: int) -> dict:
    """Assign whole groups to the three target partitions, many times over.

    Groups are shuffled and taken in turn, each going to whichever partition is
    furthest below its target share. Every flow of a group therefore lands in
    one partition, which is what a group-disjoint split requires. An attempt
    counts as covered when no partition loses a class, and as feasible when it
    is covered and no partition size deviates from its target by more than the
    tolerance. Large groups are placed first, because leaving them to the end
    is what makes a size target impossible to hit.
    """
    inv = {v: k for k, v in label_map.items()}
    classes = sorted(label_map.values())
    n = len(groups)
    targets = np.array(FRACTIONS) * n

    # Collapse to group level once. Everything an attempt needs is the size of
    # each group and its class composition, so no attempt has to touch a flow.
    codes, _ = pd.factorize(groups.values)
    n_groups = codes.max() + 1
    sizes = np.bincount(codes, minlength=n_groups).astype(np.int64)
    cls_index = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((n_groups, len(classes)), dtype=np.int64)
    np.add.at(counts, (codes, np.array([cls_index[c] for c in yy])), 1)
    order_by_size = np.argsort(-sizes)

    rng = np.random.default_rng(seed)
    best = None
    n_covered = 0
    for _ in range(restarts):
        # keep the largest groups first, shuffle the rest
        head = order_by_size[:3]
        tail = order_by_size[3:].copy()
        rng.shuffle(tail)
        order = np.concatenate([head, tail]) if len(tail) else head

        filled = np.zeros(3, dtype=np.int64)
        part_counts = np.zeros((3, len(classes)), dtype=np.int64)
        for j in order:
            p = int(np.argmax(targets - filled))
            filled[p] += sizes[j]
            part_counts[p] += counts[j]

        missing = [[inv.get(c, str(c)) for i, c in enumerate(classes)
                    if part_counts[p, i] == 0] for p in range(3)]
        covered = not any(missing)
        dev = float(np.max(np.abs(filled - targets) / targets))
        if covered:
            n_covered += 1
        key = (covered, -dev)
        if best is None or key > best[0]:
            best = (key, {"covered": covered,
                          "max_size_deviation": round(dev, 4),
                          "partition_sizes": [int(x) for x in filled],
                          "target_sizes": [int(t) for t in targets],
                          "missing_classes": missing})
    out = {"random_restarts": restarts,
           "restarts_preserving_class_coverage": n_covered,
           "best_attempt": best[1] if best else None}
    out["feasible_under_search"] = bool(
        best and best[1]["covered"]
        and best[1]["max_size_deviation"] <= tolerance)
    return out


def axis_report(values: pd.Series, y: np.ndarray, label_map: dict,
                tolerance: float, n_splits: int = 3,
                restarts: int = 200, rng_seed: int = 42) -> dict:
    inv = {v: k for k, v in label_map.items()}
    keep = values.map(is_valid).values
    n_all = len(values)
    v, yy = values[keep].values, y[keep]
    out = {"n_flows": int(n_all), "n_missing": int((~keep).sum()),
           "n_usable": int(keep.sum())}
    if out["n_usable"] == 0:
        out["feasible"] = False
        out["classes_absent_from_axis"] = [inv.get(c, str(c))
                                           for c in sorted(label_map.values())]
        out["reason"] = "the field is empty on every flow"
        out["random_restarts"] = 0
        out["restarts_preserving_class_coverage"] = 0
        out["feasible_under_search"] = False
        return out

    groups = pd.Series(v)
    sizes = groups.value_counts()
    out["n_groups"] = int(len(sizes))
    out["largest_group"] = int(sizes.iloc[0])
    out["largest_group_share"] = round(float(sizes.iloc[0] / len(v)), 4)

    # class starvation, counted over every class in the dataset rather than
    # over the classes that survive the missing-value filter. A class carried
    # by no flow on this axis disappears from the split entirely, which is the
    # strongest form of the failure and the easiest one to miss.
    per_class = {}
    for c in sorted(label_map.values()):
        name = inv.get(c, str(c))
        n_total = int((y == c).sum())
        sel = yy == c
        per_class[name] = {"n_flows_dataset": n_total,
                           "n_flows_on_axis": int(sel.sum()),
                           "n_groups": int(groups[sel].nunique())}
    out["groups_per_class"] = per_class
    dropped = [k for k, x in per_class.items() if x["n_flows_on_axis"] == 0]
    starved = [k for k, x in per_class.items()
               if 0 < x["n_groups"] < n_splits]
    out["classes_absent_from_axis"] = dropped
    out["classes_with_too_few_groups"] = starved

    # size domination against the smallest target partition
    smallest_target = min(FRACTIONS) * len(v)
    out["smallest_target_partition"] = int(smallest_target)
    out["largest_group_exceeds_smallest_partition"] = \
        bool(sizes.iloc[0] > smallest_target)

    # realized split. A refusal by the splitter is itself the answer, so the
    # message is recorded rather than allowed to end the run. BCCC-DoH is the
    # case that matters here: eleven attack flows share one certificate
    # subject, so no third fold can receive that class.
    realized, refusal = None, None
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=42)
        realized = []
        for _, idx in sgk.split(np.zeros(len(v)), yy, groups=v):
            part = {"n": int(len(idx))}
            for c in sorted(set(yy.tolist())):
                part[inv.get(c, str(c))] = int((yy[idx] == c).sum())
            part["missing_classes"] = [
                inv.get(c, str(c)) for c in sorted(set(yy.tolist()))
                if (yy[idx] == c).sum() == 0]
            realized.append(part)
    except ImportError:
        out["note"] = "StratifiedGroupKFold unavailable in this environment"
    except Exception as e:
        realized, refusal = None, "%s: %s" % (type(e).__name__, e)
    out["realized_folds"] = realized
    out["splitter_refused"] = refusal

    # A single splitter deciding once says only that one procedure failed.
    # The search below assigns whole groups to the three target partitions
    # directly, many times, and keeps the best attempt by class coverage and
    # then by size deviation. An axis that never yields a covered split over
    # this many attempts is infeasible in a stronger sense than an axis that
    # one library call refused.
    out.update(random_group_search(groups, yy, label_map, tolerance,
                                   restarts, rng_seed))

    reasons = []
    if dropped:
        reasons.append("classes carried by no flow on this axis: %s"
                       % ", ".join(dropped))
    if starved:
        reasons.append("classes present in fewer than %d groups: %s"
                       % (n_splits, ", ".join(starved)))
    if refusal:
        reasons.append("the splitter refused to build %d folds (%s)"
                       % (n_splits, refusal))
    if realized:
        empty = [i for i, p in enumerate(realized) if p["missing_classes"]]
        if empty:
            reasons.append("folds losing a class entirely: %s" % empty)
        target = len(v) / n_splits
        if max(abs(p["n"] - target) / target for p in realized) > tolerance:
            reasons.append("fold sizes deviate from the target by more than %.0f%%"
                           % (100 * tolerance))
    if not out.get("feasible_under_search", False):
        b = out.get("best_attempt")
        if b and not b["covered"]:
            reasons.append("no assignment out of %d preserved every class in "
                           "every partition" % out.get("random_restarts", 0))
        elif b:
            reasons.append("the closest covered assignment still misses a "
                           "partition target by %.0f%%"
                           % (100 * b["max_size_deviation"]))
    out["feasible"] = not reasons
    out["reason"] = "; ".join(reasons) if reasons else "no obstacle found"
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--axes", nargs="+", default=list(AXES))
    ap.add_argument("--tolerance", type=float, default=0.30,
                    help="allowed relative deviation of a partition size")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=200,
                    help="randomized group assignments tried per axis")
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {"data_root": str(root), "n_splits": args.n_splits,
              "size_tolerance": args.tolerance,
              "target_fractions": list(FRACTIONS),
              "random_restarts": args.restarts, "datasets": {}}
    for ds in args.datasets:
        d = root / ds
        if not d.is_dir():
            raise RuntimeError("%s is missing under %s" % (ds, root))
        allm = pd.concat([pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
                          for s in SPLITS], ignore_index=True)
        y = np.concatenate([np.load(d / ("y_%s.npy" % s)) for s in SPLITS])
        label_map = json.load(open(d / "label_map.json"))

        node = {}
        print("== %s ==" % ds)
        for ax in args.axes:
            col = RELATION_COLUMN.get(ax)
            if col is None or col not in allm.columns:
                continue
            vals = relation_values(allm, ax)
            if vals is None:
                continue
            r = axis_report(vals, y, label_map, args.tolerance,
                            args.n_splits, args.restarts, args.rng_seed)
            node[ax] = r
            print("   %-20s %-10s covered %d/%d | %s"
                  % (ax, "feasible" if r["feasible"] else "infeasible",
                     r.get("restarts_preserving_class_coverage", 0),
                     r.get("random_restarts", 0), r["reason"]))
        report["datasets"][ds] = node

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
