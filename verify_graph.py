# -*- coding: utf-8 -*-
"""Check that this code rebuilds the shipped graphs.

Edge construction is fully deterministic: the permutation seed in
`edges_for_values` is `blake2b(value) XOR seed` (`build_graph.py`), never
Python's process-randomized `hash`. Rebuilding from the same metadata and the
same settings must therefore give **byte-identical arrays**.

Why this matters: the shipped results were produced by the server-side
`scripts/` tree (`preprocess_hin.py`, `multiseed_eval.py`, ...) and this
artifact's `build_graph.py` and `train.py` are a rewrite that merges them.
Neural scores vary by seed, so equal scores cannot show that the two code
paths do the same thing; **equal graphs can.** If the graphs match, the only
remaining difference is the seed variation the paper already reports.

Metadata is not regenerated: without `--skip-materialize` the original
captures would overwrite `meta_*.csv` and the comparison target would be lost.

    python verify_graph.py --data data/processed_deg2 \\
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \\
        --config config.yaml --work data/_verify_graph
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from build_graph import build, requested_relations


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True, help="shipped data root")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--work", default="data/_verify_graph")
    ap.add_argument("--out", default=None, help="verdict JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    src_root, work = Path(args.data), Path(args.work)
    report = {"data_root": str(src_root), "config": args.config, "datasets": {}}
    all_ok = True

    for name in args.datasets:
        src = src_root / name
        dst = work / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        # Copy the metadata only; the edges are rebuilt by this code.
        for p in ("train", "val", "test"):
            shutil.copy2(src / ("meta_%s.csv" % p), dst / ("meta_%s.csv" % p))
            shutil.copy2(src / ("y_%s.npy" % p), dst / ("y_%s.npy" % p))

        print("\n== %s" % name)
        build(name, work, cfg, config_path=args.config)

        req = requested_relations(cfg, name)
        block = {"requested": req, "relations": {}}
        ok_ds = True
        for rel in req:
            a, b = src / ("hin_edges_%s.npy" % rel), dst / ("hin_edges_%s.npy" % rel)
            if not a.exists() and not b.exists():
                block["relations"][rel] = {"verdict": "absent in both (skipped)"}
                continue
            if a.exists() != b.exists():
                block["relations"][rel] = {
                    "verdict": "present on one side only",
                    "shipped": a.exists(), "rebuilt": b.exists()}
                ok_ds = False
                continue
            same_bytes = sha(a) == sha(b)
            ea, eb = np.load(a), np.load(b)
            same_arr = ea.shape == eb.shape and bool((ea == eb).all())
            block["relations"][rel] = {
                "shipped_edges": int(ea.shape[1]),
                "rebuilt_edges": int(eb.shape[1]),
                "identical_bytes": same_bytes,
                "identical_values": same_arr,
                "verdict": "identical" if same_arr else "different"}
            if not same_arr:
                ok_ds = False
            print("   %-24s shipped %8d  rebuilt %8d  %s"
                  % (rel, ea.shape[1], eb.shape[1],
                     "identical" if same_arr else "!!! different"))

        block["all_identical"] = ok_ds
        report["datasets"][name] = block
        all_ok &= ok_ds
        print("   -> %s" % ("all identical" if ok_ds else "!!! mismatch"))

    report["verdict"] = ("this code rebuilds the shipped graphs exactly"
                         if all_ok else "some relations do not reproduce")
    print("\n%s" % report["verdict"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
        print("[saved] %s" % out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
