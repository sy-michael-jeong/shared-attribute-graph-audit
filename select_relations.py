# -*- coding: utf-8 -*-
"""Validation-based configuration selection (Sec. 5.2, 6.3).

Trains every configuration covered by the search once (seed 42) and ranks
the configurations by validation macro-F1. Selection rule (Sec. 5.2): the
smallest configuration whose validation macro-F1 lies within 0.005 of the
best score, taking the highest-scoring configuration at that size; exact
ties fall back to the stability criterion of Sec. 6.3. Test scores are
recorded only for the stage-wise comparison printed at the end.

Each summary file holds one dataset, so run the script once per dataset and
point --summary at that dataset's file.

Usage:
    python select_relations.py --datasets iscx_vpn --data data/processed_deg2 \
        --summary results/saturation/iscx_vpn/combinatorial_grand_summary.json
"""
import argparse
import json
from pathlib import Path

import yaml

# `train` pulls in torch and PyG. --reuse-scores does not train anything, so
# the import happens where it is used and the reuse path runs without a GPU.

# Sec. 5.2: a configuration counts as tied with the best when its validation
# macro-F1 is within this margin.
TOL = 0.005


def _short(mps):
    return sorted(m.replace("via_", "") for m in mps)


def _realized(row):
    """The relations that actually put edges in the graph.

    Two configurations that differ only in a relation with no edges produce
    the same graph and the same score, so counting them separately would
    overstate how many distinct candidates the validation split failed to
    separate. BCCC-DoH requests SNI, which yields no edges there.
    """
    return tuple(sorted(row.get("metapaths_used") or row["metapaths"]))


def summarize(name, rows, by):
    """Apply the selection rule and record what the validation split could
    and could not separate.

    Ties at the ceiling are counted over distinct realized graphs and at the
    exact best score. A configuration that merely lands within TOL is a
    candidate for selection, not a tie.
    """
    ok = [r for r in rows if r["val_macro_f1"] >= 0]
    if not ok:
        raise SystemExit("%s: every configuration failed" % name)
    best = max(r["val_macro_f1"] for r in ok)

    graphs = {}
    for r in ok:
        k = _realized(r)
        if k not in graphs or r["val_macro_f1"] > graphs[k]["val_macro_f1"]:
            graphs[k] = r
    tied = [r for r in graphs.values() if r["val_macro_f1"] == best]

    near = [r for r in ok if r["val_macro_f1"] >= best - TOL]
    smallest = min(r["size"] for r in near)
    pick = max((r for r in near if r["size"] == smallest),
               key=lambda r: r["val_macro_f1"])

    return {
        "dataset": name,
        "rule": "smallest configuration within %.3f of the best validation "
                "score" % TOL,
        "best_validation_macro_f1": round(best, 4),
        "n_configurations_searched": len(rows),
        "n_distinct_graphs": len(graphs),
        "n_tied_at_ceiling": len(tied),
        "tied_at_ceiling": sorted(_short(r["metapaths"]) for r in tied),
        "selected": _short(pick["metapaths"]),
        "selected_validation_macro_f1": round(pick["val_macro_f1"], 4),
        "by_relation_count": {
            str(k): round(max(r["val_macro_f1"] for r in v), 4)
            for k, v in sorted(by.items())},
    }


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_deg2")
    ap.add_argument("--summary", required=True,
                    help="one dataset's combinatorial_grand_summary.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-scores", action="store_true",
                    help="take the scores the search already recorded instead "
                         "of training each configuration again")
    args = ap.parse_args()

    cfg = None
    if not args.reuse_scores:
        import train as T  # noqa: F401
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    summary = json.load(open(args.summary))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in summary:
            raise SystemExit(
                "%s holds %s, not %s. Each summary file covers one dataset, so "
                "pass that dataset's own file."
                % (args.summary, [k for k in summary if not k.startswith("_")],
                   name))
        block = summary[name]
        # Older summaries keep only the ten best under `top_overall`. The beam
        # visited more than that, and the selection rule needs the full set, so
        # fall back to flattening the per-stage lists.
        visited = block.get("all_results")
        if visited is None:
            visited = [r for stage in block.get("stages", {}).values()
                       for r in stage]
        if not visited:
            raise SystemExit("%s carries no configurations for %s"
                             % (args.summary, name))

        seen, configs = set(), []
        for r in visited:
            mps = tuple(sorted(r["metapaths"]))
            if mps not in seen:
                seen.add(mps)
                configs.append(list(mps))
        print("== %s: %d configurations" % (name, len(configs)))

        rows = []
        if args.reuse_scores:
            # The search already trained every configuration and recorded both
            # scores. Training them again costs a GPU day and cannot change the
            # answer, so the recorded scores are used as they are.
            by_key = {}
            for r in visited:
                by_key.setdefault(tuple(sorted(r["metapaths"])), r)
            for mps in configs:
                r = by_key[tuple(mps)]
                row = {"metapaths": mps,
                       "val_macro_f1": float(r.get("val_macro_f1",
                                                   r.get("score", -1.0))),
                       "test_macro_f1": float(r.get("test_macro_f1", -1.0))}
                if r.get("metapaths_used"):
                    row["metapaths_used"] = list(r["metapaths_used"])
                # Size counts the graph that was realized, not the set that was
                # requested. A relation that yields no edge leaves a smaller
                # graph behind, and two requests can be one graph.
                row["size"] = len(set(_realized(row)))
                rows.append(row)
        else:
            import train as T
            data_dir = Path(args.data) / name
            for i, mps in enumerate(configs):
                print("[%d/%d] %s" % (i + 1, len(configs), "+".join(mps)))
                try:
                    m = T.run_han(data_dir, cfg, mps, args.seed)
                    used = list(m.get("relations", mps))
                    rows.append({"metapaths": mps, "metapaths_used": used,
                                 "size": len(set(used)),
                                 "val_macro_f1": float(m.get("best_val", -1.0)),
                                 "test_macro_f1": float(m.get("macro_f1", -1.0))})
                except Exception as e:
                    print("  [error] %s" % e)
                    rows.append({"metapaths": mps, "size": len(mps),
                                 "val_macro_f1": -1.0, "test_macro_f1": -1.0})

        by = {}
        for r in rows:
            by.setdefault(r["size"], []).append(r)
        print("=== stage-wise selection (val-top vs test-top) ===")
        for k in sorted(by):
            vt = max(by[k], key=lambda r: r["val_macro_f1"])
            tt = max(by[k], key=lambda r: r["test_macro_f1"])
            same = "SAME" if vt["metapaths"] == tt["metapaths"] else "DIFF"
            print("k=%d  val-top %s (val %.3f / test %.3f)  %s" %
                  (k, "+".join(vt["metapaths"]), vt["val_macro_f1"],
                   vt["test_macro_f1"], same))

        summary_out = summarize(name, rows, by)
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(d / "rows.json", "w"), indent=2)
        json.dump(summary_out, open(d / (name + ".json"), "w"), indent=2)
        print("[saved] %s" % (d / (name + ".json")))
        print("  selected %s (%d tied at the ceiling of %d searched)"
              % ("+".join(summary_out["selected"]),
                 summary_out["n_tied_at_ceiling"],
                 summary_out["n_configurations_searched"]))


if __name__ == "__main__":
    main()
