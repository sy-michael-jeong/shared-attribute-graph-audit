# Build metapath edges over extracted flows and save everything under
# data/processed_<tag>/<dataset>/ for training.
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import MISSING_TOKENS, RELATION_COLUMN


def timebin(ts_col, bin_seconds):
    out = np.zeros(len(ts_col), dtype=np.int64)
    for i, t in enumerate(ts_col):
        try:
            v = float(t)
        except Exception:
            try:
                v = pd.Timestamp(t).timestamp()
            except Exception:
                v = 0.0
        out[i] = int(v // bin_seconds)
    return out


def value_rng(seed, value):
    """A stream that depends on the value, never on where its rows sit.

    Seeding from the value keeps the graph identical under any reordering of
    the flow table, and keeps the groups independent of each other. Python's
    built-in hash is randomised per process, so the digest is taken here.
    """
    h = hashlib.blake2b(value.encode("utf-8", "replace"), digest_size=4)
    mixed = int.from_bytes(h.digest(), "big") ^ (int(seed) & 0xFFFFFFFF)
    return np.random.RandomState(mixed & 0xFFFFFFFF)


def row_keys(meta):
    """A per-flow identity taken from the flow tuple, not from its position.

    The extraction step returns the same flows in an order that can vary
    between runs. Ordering a group by this key instead of by row number makes
    the graph a function of the flows alone.
    """
    cols = [c for c in ("src_ip", "sport", "dst_ip", "dport", "ts")
            if c in meta.columns]
    if len(cols) < 3:
        return None
    joined = meta[cols].astype(str).agg("|".join, axis=1).tolist()
    return np.array([int.from_bytes(
        hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(),
        "big") for s in joined], dtype=np.uint64)


def edges_for_values(values, seed, degree, keys=None):
    """Connect rows that share a value, bounding the neighbourhood per flow.

    Sharing a value is a symmetric statement, so pairs are drawn without order
    and both directions are emitted. Within a value, each flow keeps at most
    `degree` neighbours, drawn along a random ring over the group, so the
    neighbourhood a flow receives does not depend on how common its value is.
    Groups small enough to stay under the budget become complete.

    The ring reaches the same distance in both directions, so the budget is
    even and an odd one would not be the degree the caller asked for.
    """
    if degree < 2 or degree % 2:
        raise ValueError(
            "degree must be an even number of at least 2, got %r" % (degree,))

    groups = {}
    for i, v in enumerate(values):
        s = str(v)
        if s.strip() in MISSING_TOKENS:
            continue
        groups.setdefault(s, []).append(i)

    src, dst = [], []
    for val, idxs in groups.items():
        n = len(idxs)
        if n < 2:
            continue
        arr = np.array(idxs, dtype=np.int64)
        if n - 1 <= degree:
            u, w = np.triu_indices(n, k=1)
            a, b = arr[u], arr[w]
        else:
            # Put the group in flow-identity order, shuffle it, then join each
            # flow to the flows that follow it on the ring. Every flow gets the
            # same number of neighbours, no position in the table is
            # privileged, and reordering the table changes nothing.
            if keys is not None:
                arr = arr[np.argsort(keys[arr], kind="stable")]
            ring = arr[value_rng(seed, val).permutation(n)]
            half = degree // 2
            a = np.concatenate([ring] * half)
            b = np.concatenate([np.roll(ring, -off) for off in range(1, half + 1)])
        src.append(np.concatenate([a, b]))
        dst.append(np.concatenate([b, a]))

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    s = np.concatenate(src); d = np.concatenate(dst)
    keep = s != d
    s, d = s[keep], d[keep]
    if s.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    _, uniq = np.unique(s.astype(np.int64) * (10 ** 9) + d, return_index=True)
    uniq.sort()
    return np.stack([s[uniq], d[uniq]], axis=0).astype(np.int64)


def build_edges(meta, relations, seed, degree, timebin_seconds):
    edges, used, skipped = {}, [], []
    keys = row_keys(meta)
    for rel in relations:
        col = RELATION_COLUMN.get(rel)
        if col is None or col not in meta.columns:
            skipped.append((rel, "column missing"))
            continue
        raw = meta[col].values
        vals = timebin(raw, timebin_seconds).astype(str) if rel == "via_timebin" \
            else np.array([str(v) if v is not None else "" for v in raw])
        ed = edges_for_values(vals, seed, degree, keys)
        if ed.shape[1] == 0:
            skipped.append((rel, "no shared values"))
            continue
        edges[rel] = ed
        used.append(rel)
    return edges, used, skipped


def materialize(name, raw_root, out_root, cfg, task, split_mode):
    # Imported here so that the edge builder above can be exercised, and its
    # invariants checked, without the training stack installed.
    from datasets import LOADERS
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    split = LOADERS[name](raw_root / name, task=task,
                          test_size=cfg["data"]["test_size"],
                          val_size=cfg["data"]["val_size"],
                          seed=cfg["seed"], split_mode=split_mode)
    for part in ("train", "val", "test"):
        np.save(out_dir / ("X_%s.npy" % part), split["X_%s" % part])
        np.save(out_dir / ("y_%s.npy" % part), split["y_%s" % part])
        split["meta_%s" % part].to_csv(out_dir / ("meta_%s.csv" % part), index=False)
    json.dump(split["feature_names"], open(out_dir / "feature_names.json", "w"), indent=2)
    json.dump(split["label_map"], open(out_dir / "label_map.json", "w"), indent=2)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(split["scaler"], f)


def requested_relations(cfg, name):
    """Relations requested for this dataset.

    Uses `metapaths_by_dataset` when it has an entry, else `metapaths`. BCCC-DoH
    and CIC-AndMal request the TLS relations only: in those two corpora the host
    and temporal relations read the capture setup rather than the traffic
    (Section 3.2 of the paper).

    Without this function the fact that the request differs by dataset would
    vanish from the code, and a relation absent from `skipped` in
    `hin_summary.json` could not be told apart as "not realized" vs "not requested".
    """
    per = cfg["hin"].get("metapaths_by_dataset") or {}
    return list(per.get(name, cfg["hin"]["metapaths"]))


def build(name, out_root, cfg, config_path=""):
    degree = cfg["hin"]["max_degree_per_value"]
    out_dir = out_root / name
    meta = pd.concat([pd.read_csv(out_dir / ("meta_%s.csv" % p))
                      for p in ("train", "val", "test")], ignore_index=True)
    requested = requested_relations(cfg, name)
    pool = list(cfg["hin"]["metapaths"])
    not_requested = [r for r in pool if r not in requested]
    edges, used, skipped = build_edges(
        meta, requested, cfg["seed"],
        degree, cfg["hin"]["timebin_seconds"])
    for rel, ed in edges.items():
        np.save(out_dir / ("hin_edges_%s.npy" % rel), ed)
        print("  %s: %d edges" % (rel, ed.shape[1]))
    if not_requested:
        print("  not requested for %s: %s" % (name, not_requested))
    json.dump({"config": str(config_path),
               "pool": pool,
               "not_requested": not_requested,
               "requested": requested,
               "used": used,
               "skipped": [{"relation": r, "reason": why} for r, why in skipped],
               "max_degree_per_value": degree,
               "timebin_seconds": cfg["hin"]["timebin_seconds"],
               "seed": cfg["seed"],
               "n_flows": int(len(meta)),
               "edges": {r: int(edges[r].shape[1]) for r in used}},
              open(out_dir / "hin_summary.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="dataset names as defined in datasets.py")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--config", default="config.yaml")
    # The sensitivity analyses need data roots that differ only in the budget or
    # the time-bin width. Taking both on the command line (and recording them in
    # hin_summary.json) keeps the provenance in the result files instead of in
    # hand-edited copies of config.yaml.
    ap.add_argument("--max-degree", type=int, default=None,
                    help="override hin.max_degree_per_value from config")
    ap.add_argument("--timebin-seconds", type=float, default=None,
                    help="override hin.timebin_seconds from config")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--split-mode", default="random",
                    choices=["random", "time", "time_stratified",
                             "sni_disjoint", "ja3_disjoint", "dst_ip_disjoint"])
    ap.add_argument("--skip-materialize", action="store_true",
                    help="reuse existing meta_*.csv and only rebuild edges")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    overrides = {}
    if args.max_degree is not None:
        cfg["hin"]["max_degree_per_value"] = args.max_degree
        overrides["max_degree_per_value"] = args.max_degree
    if args.timebin_seconds is not None:
        cfg["hin"]["timebin_seconds"] = args.timebin_seconds
        overrides["timebin_seconds"] = args.timebin_seconds
    if overrides:
        print("  config override: %s" % overrides)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print("== %s (split=%s)" % (name, args.split_mode))
        if not args.skip_materialize:
            materialize(name, Path(args.raw), out_root, cfg, args.task, args.split_mode)
        build(name, out_root, cfg, args.config)


if __name__ == "__main__":
    main()
