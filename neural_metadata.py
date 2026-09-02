# -*- coding: utf-8 -*-
"""The same-source metadata features inside the neural pipeline (Sec. 6.1, 8).

The tabular comparison of Table 7 uses gradient boosting. This script hands the
same features to an MLP, so that the metadata is held constant and the learner
is the only thing that changes.

The MLP shares the training protocol of HAN. It is fitted on the training
partition alone, stopped early on validation macro-F1, and run over the same
five seeds, so there is no fit-partition variant here.

Features and the relation pool come from the same-source baseline, including
the out-of-fold treatment of the two training-count features.

Usage:
    python neural_metadata.py \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_time_strat --out runs/mlp_same_info --config config.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import same_info_matched as SIM  # noqa: E402
from common import seed_all, resolve_device  # noqa: E402

from common import SPLITS


class MLP(torch.nn.Module):
    """One hidden layer with ELU, dropout on the input and the hidden layer."""

    def __init__(self, in_dim, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def standardize(tr, va, te):
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(a - mu) / sd for a in (tr, va, te)]


def train_once(Xs, ys, cfg, seed, device):
    from sklearn.metrics import f1_score
    seed_all(seed)
    n_classes = int(max(y.max() for y in ys)) + 1
    X = [torch.from_numpy(a.astype(np.float32)).to(device) for a in Xs]
    y = [torch.from_numpy(a.astype(np.int64)).to(device) for a in ys]

    c = cfg["classifier"]
    model = MLP(X[0].shape[1], c.get("hidden", 128), n_classes,
                dropout=c.get("dropout", 0.5)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"],
                           weight_decay=c["weight_decay"])
    cw = np.bincount(ys[0], minlength=n_classes).astype(np.float32)
    cw = (cw.sum() / np.maximum(cw, 1)) / n_classes
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, device=device, dtype=torch.float32))

    best_val, bad, best_state = -1.0, 0, None
    for _ in range(1, c["epochs"] + 1):
        model.train(); opt.zero_grad()
        crit(model(X[0]), y[0]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(X[1]).argmax(-1).cpu().numpy()
        f = f1_score(ys[1], vp, average="macro", zero_division=0)
        if f > best_val:
            best_val, bad = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c.get("patience", 15):
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = model(X[2]).argmax(-1).cpu().numpy()
    return (f1_score(ys[2], tp, average="macro", zero_division=0),
            float(min(f1_score(ys[2], tp, average=None, zero_division=0))))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed_time_strat")
    ap.add_argument("--out", default="results/neural_metadata")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--timebin-seconds", type=float, default=300.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = resolve_device(cfg.get("device", "auto"))
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in args.datasets:
        p = root / name
        if not p.is_dir():
            print("[skip] %s" % p); continue
        Xs = [np.load(p / ("X_%s.npy" % s)) for s in SPLITS]
        ys = [np.load(p / ("y_%s.npy" % s)) for s in SPLITS]
        metas = [pd.read_csv(p / ("meta_%s.csv" % s), low_memory=False) for s in SPLITS]
        allm = pd.concat(metas, ignore_index=True)
        n_tr, n_va = len(Xs[0]), len(Xs[1])

        pool = sorted(f.stem.replace("hin_edges_", "") for f in p.glob("hin_edges_*.npy"))
        pool = [r for r in SIM.ALL_RELATIONS if r in pool] or SIM.ALL_RELATIONS
        sel = SIM.SELECTED[name]

        M_full, n_full, used_full = SIM.build_features(allm, pool, n_tr, args.timebin_seconds)
        M_sel, n_sel, used_sel = SIM.build_features(allm, sel, n_tr, args.timebin_seconds)

        def cut(M):
            return standardize(M[:n_tr], M[n_tr:n_tr + n_va], M[n_tr + n_va:])

        combos = [("flow_only", Xs)]
        for tag, M in (("selected", M_sel), ("full", M_full)):
            if M is None:
                continue
            Ms = cut(M)
            combos.append(("meta_%s" % tag, Ms))
            combos.append(("flow_plus_meta_%s" % tag,
                           [np.hstack([a, b]) for a, b in zip(Xs, Ms)]))

        res = {"relations_full": used_full, "relations_selected": used_sel, "runs": {}}
        print("\n=== %s  full=%d sel=%d" % (name, len(used_full), len(used_sel)))
        for tag, feats in combos:
            ms, mn = [], []
            for sd in args.seeds:
                a, b = train_once(feats, ys, cfg, sd, device)
                ms.append(a); mn.append(b)
            res["runs"][tag] = {"macro_f1_mean": round(float(np.mean(ms)), 4),
                                "macro_f1_std": round(float(np.std(ms)), 4),
                                "minority_f1_mean": round(float(np.mean(mn)), 4),
                                "per_seed": [round(x, 4) for x in ms]}
            print("    %-30s %.4f +- %.4f" % (tag, np.mean(ms), np.std(ms)))
        report[name] = res

    json.dump(report, open(out / "summary.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "summary.json"))


if __name__ == "__main__":
    main()
