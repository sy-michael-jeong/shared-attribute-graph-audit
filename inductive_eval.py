# -*- coding: utf-8 -*-
"""Deployment-style inductive evaluation (Sec. 6.2).

Every other run in this artifact is transductive. The graph is built once over
the union of the partitions, so a test flow is a node during training and can
be linked to other test flows. Deployment is not like that. A new flow arrives
after training has finished, and it cannot see the other new flows arriving
with it.

The cross-partition masking control closes half of this gap. It removes the
edges that join training to test, but it keeps the edges among test flows, and
it also removes edges that touch the validation partition, which degrades early
stopping. This script closes both halves at once.

--- The deployment rule ---

  training    the graph is built from the training flows alone; validation and
              test flows do not exist
  validation  validation flows are attached to that graph and scored; used only
              for early stopping
  test        test flows are attached to that graph and scored

  attachment
    - a new flow joins **training flows** that carry its value, and no others
    - it receives at most the per-value budget used during training
    - it is isolated in a relation when the training partition holds no such
      value
    - **edges run from the training flow to the new flow only**

The last rule is what separates this from the transductive graph. Messages pass
from source to target, so an edge in the other direction would let a training
node aggregate the features of a flow that does not yet exist. With two layers
it would also open a new -> training -> new path, joining two test flows through
a shared neighbour even though no edge joins them directly.

--- What this control answers ---

  answers        what a relation is left carrying once both the training-test
                 and the test-test channels are gone, which is what deployment
                 can expect
  does not       whether the training graph itself encodes the capture
                 environment; edges among training flows remain

--- Verification ---

A run that finishes is not a run that is correct. Six invariants are asserted
while the graph is built, and the first three are written into the output.

  new_new_edges              edges between two new flows                 0
  new_as_source_edges        edges whose source is a new flow            0
  train_indegree_unchanged   training neighbourhoods survive attachment  true
  (index range, same value on both ends, attach degree <= budget)

The last three raise rather than report, so a completed run is the evidence.
The first two alone are not enough: with two layers a new-new path can exist
without a new-new edge, and `train_indegree_unchanged` is what rules that out.

Usage:
    python inductive_eval.py --datasets bccc_dohbrw \
        --data data/processed_deg2 --sets via_cert_validity \
        --out results/inductive
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from build_graph import edges_for_values, row_keys
from common import (RELATION_COLUMN, SPLITS, TIMEBIN_SECONDS, is_valid,
                    timebin_values)

# `train` and `models` pull in torch and PyG. Building the graph and checking
# the attachment rule need neither, so they are imported where they are used.
# This is what lets the invariant test and `--dry-run` run on a machine with
# no GPU, before any device is taken.


def relation_values(meta: pd.DataFrame, rel: str, bin_seconds: float):
    """Per-flow value for one relation, or None when the column is absent."""
    col = RELATION_COLUMN.get(rel)
    if col is None or col not in meta.columns:
        return None
    raw = meta[col].values
    if rel == "via_timebin":
        return timebin_values(raw, bin_seconds).astype(str)
    return np.array([str(v) if v is not None else "" for v in raw])


def train_graph(values, n_train: int, seed: int, budget: int, keys):
    """Edges among the training flows, under the rule used everywhere else."""
    v = np.array(values[:n_train], dtype=object)
    k = None if keys is None else keys[:n_train]
    return edges_for_values(v, seed, budget, k)


def _pick_without_replacement(rng, pool_size: int, k: int, m: int) -> np.ndarray:
    """For each of m rows, k distinct draws from 0..pool_size-1.

    Calling `rng.choice(n, k, replace=False)` once per row is not an option.
    numpy permutes all n on every call, so attaching 150k new flows to a value
    group of 150k training flows costs 2e10 operations.

    Instead the j-th draw is taken uniformly from the remaining pool_size - j
    and shifted past the draws already made. The distribution is the same as
    sampling without replacement, and every row is handled at once.
    """
    out = np.empty((m, k), dtype=np.int64)
    for j in range(k):
        r = rng.randint(0, pool_size - j, size=m)
        if j:
            prev = np.sort(out[:, :j], axis=1)
            for t in range(j):
                r = r + (r >= prev[:, t])
        out[:, j] = r
    return out


def attach_new(values, n_train: int, budget: int, seed: int) -> np.ndarray:
    """Attach new flows to training flows. New flows are never joined.

    A new flow is linked to at most `budget` training flows carrying its value,
    and is isolated when the training partition holds no such value. The random
    state is drawn from the value, so value groups are independent of one
    another and the graph does not depend on the row order of the table.

    **Edges are one-directional: training -> new.** The transductive builder
    emits symmetric edges, which makes a symmetric attachment the natural thing
    to write, and that breaks the deployment rule in two places at once. See
    the module docstring.
    """
    vals = np.asarray(values, dtype=object)
    valid = np.array([is_valid(str(v)) for v in vals])

    train_by_value: Dict[str, List[int]] = {}
    for i in np.flatnonzero(valid[:n_train]):
        train_by_value.setdefault(str(vals[i]), []).append(int(i))
    new_by_value: Dict[str, List[int]] = {}
    for j in np.flatnonzero(valid[n_train:]) + n_train:
        s = str(vals[j])
        if s in train_by_value:
            new_by_value.setdefault(s, []).append(int(j))

    src_parts, dst_parts = [], []
    for s, new_idx in new_by_value.items():
        pool = np.asarray(train_by_value[s], dtype=np.int64)
        new = np.asarray(new_idx, dtype=np.int64)
        h = hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=4)
        rng = np.random.RandomState(
            (int.from_bytes(h.digest(), "big") ^ (int(seed) & 0xFFFFFFFF))
            & 0xFFFFFFFF)
        if len(pool) <= budget:
            a = np.repeat(new, len(pool))
            b = np.tile(pool, len(new))
        else:
            pick = _pick_without_replacement(rng, len(pool), budget, len(new))
            a = np.repeat(new, budget)
            b = pool[pick].reshape(-1)
        # training (b) -> new (a). The reverse direction breaks the rule.
        src_parts.append(b)
        dst_parts.append(a)

    if not src_parts:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_parts),
                     np.concatenate(dst_parts)], axis=0)


def source_fingerprint(d: Path) -> dict:
    """A fingerprint of the split files this run read.

    A path string alone cannot tell two generations of a directory apart. Row
    counts and a digest of the metadata table can, so the directory can be
    measured again later and compared.
    """
    out = {}
    for s in SPLITS:
        y = np.load(d / ("y_%s.npy" % s))
        h = hashlib.blake2b(digest_size=8)
        with open(d / ("meta_%s.csv" % s), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[s] = {"n": int(y.shape[0]), "n_positive": int((y != 0).sum()),
                  "meta_blake2b": h.hexdigest()}
    return out


def build(meta, ns, relations, seed, budget, timebin_seconds):
    """Edges per relation, with the evidence that the rule was kept."""
    n_train, n_all = ns[0], sum(ns)
    keys = row_keys(meta)
    edges, report = {}, {}
    for rel in relations:
        v = relation_values(meta, rel, timebin_seconds)
        if v is None:
            # An unknown name stops the run. Skipping it would quietly build a
            # smaller set of relations than the one asked for, and the output
            # would not say so. CIC-AndMal carries seven, so a table built from
            # six still looks plausible.
            raise SystemExit(
                "%s: unknown relation. Known: %s"
                % (rel, ", ".join(sorted(RELATION_COLUMN))))
        tr = train_graph(v, n_train, seed, budget, keys)
        at = attach_new(v, n_train, budget, seed)
        e = np.concatenate([tr, at], axis=1) if at.shape[1] else tr
        if e.shape[1] == 0:
            continue

        # An out-of-range index dies inside CUDA as a device-side assert, and
        # that message does not say which relation caused it.
        if int(e.max()) >= n_all or int(e.min()) < 0:
            raise SystemExit("%s: edge index outside [0, %d)" % (rel, n_all))
        # A shared-attribute graph joins flows that carry the same value. The
        # time relation is no exception; its value is the bucket identifier.
        if at.shape[1]:
            same = np.asarray(v, dtype=object)[at[0]] == \
                np.asarray(v, dtype=object)[at[1]]
            if not bool(same.all()):
                raise SystemExit(
                    "%s: %d attachment edges join different values"
                    % (rel, int((~same).sum())))

        src_new = e[0] >= n_train
        n_new_new = int((src_new & (e[1] >= n_train)).sum())
        n_new_src = int(src_new.sum())
        if n_new_new:
            raise SystemExit("%s: %d edges join two new flows" % (rel, n_new_new))
        if n_new_src:
            raise SystemExit(
                "%s: %d edges take a new flow as source, so a training node "
                "would aggregate it" % (rel, n_new_src))

        indeg = np.bincount(e[1], minlength=n_all)
        indeg_train_only = np.bincount(tr[1], minlength=n_all)[:n_train]
        indeg_attach = np.bincount(at[1], minlength=n_all) if at.shape[1] \
            else np.zeros(n_all, dtype=np.int64)
        if int(indeg_attach.max()) > budget:
            raise SystemExit(
                "%s: attachment degree %d exceeds the budget of %d"
                % (rel, int(indeg_attach.max()), budget))

        edges[rel] = e
        report[rel] = {
            "n_train_edges": int(tr.shape[1]),
            "n_attach_edges": int(at.shape[1]),
            "new_new_edges": n_new_new,
            "new_as_source_edges": n_new_src,
            "attach_rate": round(float((indeg[n_train:] > 0).mean()), 4),
            "attach_rate_test": round(
                float((indeg[n_train + ns[1]:] > 0).mean()), 4),
            "isolated_test": int((indeg[n_train + ns[1]:] == 0).sum()),
            "train_indegree_max_before": int(indeg_train_only.max()),
            "train_indegree_max_after": int(indeg[:n_train].max()),
            "train_indegree_unchanged": bool(
                np.array_equal(indeg_train_only, indeg[:n_train])),
        }
        if not report[rel]["train_indegree_unchanged"]:
            raise SystemExit(
                "%s: attachment changed a training neighbourhood" % rel)
    return edges, report


def run_sets(name, data_root, cfg_all, relations, seeds, edges):
    """Train on the inductive graph through the reported code path.

    `run_han` reads its edges through `load_edges`. Replacing that one function
    keeps the model, the loss, the early stopping and the seed handling
    identical to the runs reported elsewhere. Writing a separate training loop
    would give up that identity.
    """
    import train
    from models import agg

    real = train.load_edges
    train.load_edges = lambda data_dir, rels: {
        r: e for r, e in edges.items() if r in rels}
    try:
        per = []
        for s in seeds:
            cfg = copy.deepcopy(cfg_all)
            cfg["seed"] = s
            print("  [inductive] seed=%d" % s)
            # One failing seed must not discard the seeds already finished.
            try:
                m = train.run_han(data_root / name, cfg, list(relations), s)
                per.append({"seed": s,
                            "macro_f1": float(m.get("macro_f1", -1.0)),
                            "minority_f1": float(m.get("minority_f1", -1.0)),
                            "weighted_f1": float(m.get("weighted_f1", -1.0))})
            except Exception as exc:  # noqa: BLE001
                print("  [failed] seed=%d %s: %s" % (s, type(exc).__name__, exc))
                per.append({"seed": s, "macro_f1": -1.0, "minority_f1": -1.0,
                            "weighted_f1": -1.0,
                            "error": "%s: %s" % (type(exc).__name__, exc)})
    finally:
        train.load_edges = real

    def col(k):
        return agg([p[k] for p in per if p["macro_f1"] >= 0])

    return {"metapaths": list(relations), "per_seed": per,
            "macro_f1": col("macro_f1"), "minority_f1": col("minority_f1"),
            "weighted_f1": col("weighted_f1")}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", required=True,
                    help='"rel1+rel2"; the reported relation set of the dataset')
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--dry-run", action="store_true",
                    help="build the graph and report the invariants only")
    args = ap.parse_args()

    # The reported relation set differs by dataset, and the five metadata
    # tables carry the same columns, so one --sets applied to several datasets
    # would run without error and report the wrong relation for four of them.
    if len(args.datasets) > 1:
        raise SystemExit(
            "--datasets takes one name. --sets differs by dataset, so a list "
            "would apply the same relation to all of them.")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    budget = cfg["hin"]["max_degree_per_value"]
    tb = cfg["hin"].get("timebin_seconds", TIMEBIN_SECONDS)
    relations = [m.strip() for m in args.sets.split("+") if m.strip()]
    root, out_root = Path(args.data), Path(args.out)

    for name in args.datasets:
        d = root / name
        metas = [pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
                 for s in SPLITS]
        ns = [len(m) for m in metas]
        # Edge indices are row numbers in the metadata table while training
        # runs on X. A mismatch points the edges at other flows and still runs.
        for s, n in zip(SPLITS, ns):
            n_y = int(np.load(d / ("y_%s.npy" % s)).shape[0])
            if n_y != n:
                raise SystemExit("%s/%s: metadata has %d rows, y has %d"
                                 % (name, s, n, n_y))
        meta = pd.concat(metas, ignore_index=True)
        print("== %s  train %d / val %d / test %d" % (name, ns[0], ns[1], ns[2]))

        edges, report = build(meta, ns, relations, cfg["seed"], budget, tb)
        if not edges:
            raise SystemExit("%s: no relation produced an edge" % name)

        for rel, r in report.items():
            print("    %-24s train %9d  attached %9d  new-new %d  new-as-src %d"
                  "  test attach %.3f  isolated %7d  train nbhd kept %s"
                  % (rel, r["n_train_edges"], r["n_attach_edges"],
                     r["new_new_edges"], r["new_as_source_edges"],
                     r["attach_rate_test"], r["isolated_test"],
                     r["train_indegree_unchanged"]))

        block = {name: {
            "data_root": str(root), "protocol": "inductive",
            "attach_rule": "new flows join training flows only, never each other",
            "graph_seed": cfg["seed"], "config": args.config,
            "sets_arg": args.sets, "seeds": list(args.seeds), "budget": budget,
            "n_train": ns[0], "n_val": ns[1], "n_test": ns[2],
            "source_fingerprint": source_fingerprint(d),
            "graph": report, "sets": {}}}

        out = out_root / (name + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            json.dump(block, open(out, "w"), indent=2)
            print("  [saved] %s   (graph only, no training)" % out)
            continue

        block[name]["sets"]["manual_1"] = run_sets(
            name, root, cfg, relations, args.seeds, edges)
        json.dump(block, open(out, "w"), indent=2)
        r = block[name]["sets"]["manual_1"]
        print("  macro %s +- %s   minority %s +- %s"
              % (r["macro_f1"]["mean"], r["macro_f1"]["std"],
                 r["minority_f1"]["mean"], r["minority_f1"]["std"]))
        print("  [saved] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
