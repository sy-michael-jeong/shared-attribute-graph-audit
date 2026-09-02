# -*- coding: utf-8 -*-
"""HAN on a graph with no relation edges (Sec. 5.3, 6.1).

HAN scores above a feature-only MLP on every dataset, and that gap is read as
evidence that the relations carry signal. The gap also has a second possible
source, since HAN and the MLP differ in depth, attention, normalization and
parameter count as well as in what they read. This control separates the two.

Every relation keeps its name and its place in the metadata, so the model is
constructed identically, but each relation's edge list is replaced by self
loops on all nodes. Attention therefore has the same shape to work with and
nothing to aggregate except the node's own features.

This is an operational architecture control, not a pure edge ablation. Because
self-loops are assigned to every node, including flows that do not carry the
field, the control removes inter-node messages *and* the field-presence
structure that relation availability encodes. The difference against the
reported HAN score is therefore read in the paper as the combined contribution
of neighbors and field presence (Sections 5.3 and 7.2), which value permutation
then splits further.

The reported HAN configuration of each dataset is used, so the comparison is
against the number in the main table rather than against a different graph.

One file is written per dataset, at <out>/<dataset>/multiseed_summary.json, in
the same shape train.py writes. bootstrap_ci.py walks that layout, so a single
combined file would be skipped there without a warning.

Usage:
    CUDA_VISIBLE_DEVICES=0 python no_edge_han.py \
        --data data/processed_deg2 \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/no_edge
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SELECTED, graph_fingerprint

SEEDS = [41, 42, 43, 44, 45]


def self_loop_edges(n: int) -> np.ndarray:
    ar = np.arange(n, dtype=np.int64)
    return np.stack([ar, ar], axis=0)


def run_seed(data_dir: Path, cfg: dict, relations, seed: int) -> dict:
    """run_han of train.py with every edge list replaced by self loops."""
    import torch
    import train as T
    from models import HAN, seed_all, pick_device

    seed_all(seed)
    device = pick_device(cfg.get("device", "auto"))
    arrs, label_map = T.load_splits(data_dir)
    x, y, masks = T.stack(arrs)
    n = x.shape[0]

    # The relations that would have carried edges, kept so that the metadata,
    # the number of attention heads over relations and the parameter count are
    # the ones the reported model uses.
    present = [r for r in relations
               if (data_dir / ("hin_edges_%s.npy" % r)).exists()
               and np.load(data_dir / ("hin_edges_%s.npy" % r)).shape[1] > 0]
    if not present:
        raise RuntimeError("no realized relations for %s" % data_dir.name)

    metadata = (["flow"], [("flow", r, "flow") for r in present])
    loop = torch.from_numpy(self_loop_edges(n)).long().to(device)
    edge_index_dict = {("flow", r, "flow"): loop for r in present}

    x = x.to(device); y = y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    ccfg = cfg["classifier"]
    model = HAN(x.shape[1], ccfg["hidden"], len(label_map), metadata,
                num_layers=ccfg["num_layers"], heads=ccfg["heads"],
                dropout=ccfg["dropout"],
                residual=ccfg.get("residual", True)).to(device)
    forward = lambda: model({"flow": x}, edge_index_dict)
    m = T.fit_neural(model, forward, y, masks, ccfg, len(label_map), device)
    m["seed"] = seed
    m["relations"] = present
    return m


def main():
    import yaml
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--sets", default=None,
                    help="relations joined by '+'. Without it the reported "
                         "configuration of the dataset is used. The control "
                         "has to be able to follow a configuration that is not "
                         "the reported one, because Sec. 6.6 runs the same "
                         "decomposition on a configuration taken from the "
                         "literature, and a control that silently used the "
                         "reported relations instead would be comparing two "
                         "different models.")
    ap.add_argument("--out", required=True,
                    help="directory; one <dataset>/multiseed_summary.json each")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(args.data)
    out_root = Path(args.out)

    for ds in args.datasets:
        d = root / ds
        if not d.is_dir():
            raise RuntimeError("%s is missing under %s" % (ds, root))
        rels = ([r.strip() for r in args.sets.split("+") if r.strip()]
                if args.sets else SELECTED[ds])
        unknown = [r for r in rels if not r.startswith("via_")]
        if unknown:
            raise RuntimeError("relation names must start with via_: %s" % unknown)
        per = []
        for s in args.seeds:
            m = run_seed(d, cfg, rels, s)
            per.append({"seed": s,
                        "macro_f1": float(m["macro_f1"]),
                        "minority_f1": float(m.get("minority_f1", 0.0)),
                        "weighted_f1": float(m.get("weighted_f1", 0.0))})
            print("   %s seed %d  macro-F1 %.4f" % (ds, s, m["macro_f1"]))

        def agg(key):
            v = [p[key] for p in per]
            return {"mean": round(float(np.mean(v)), 4),
                    "std": round(float(np.std(v)), 4), "n": len(v),
                    "values": [round(x, 4) for x in v]}

        block = {ds: {"data_root": str(root), "seeds": list(args.seeds),
                      "control": "every relation edge list replaced by self loops",
                      "graph": graph_fingerprint(d),
                      "sets": {"manual_1": {
                          "set_tag": "manual_1", "metapaths": list(rels),
                          "per_seed": per,
                          "macro_f1": agg("macro_f1"),
                          "minority_f1": agg("minority_f1"),
                          "weighted_f1": agg("weighted_f1")}}}}
        out = out_root / ds / "multiseed_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(block, open(out, "w"), indent=2)
        print("== %s ==  self-loop HAN %.4f +- %.4f   [saved] %s"
              % (ds, np.mean([p["macro_f1"] for p in per]),
                 np.std([p["macro_f1"] for p in per]), out))


if __name__ == "__main__":
    main()
