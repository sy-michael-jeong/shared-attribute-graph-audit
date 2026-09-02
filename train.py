# Train one model family (han / gcn / mlp / hgb / egs) over multiple seeds and
# write a per-dataset summary json under --runs.
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from common import graph_fingerprint, is_valid, SELECTED
from models import HAN, GCN, MLP, EGraphSAGE, seed_all, pick_device, metrics, agg, class_weights

SEEDS = [41, 42, 43, 44, 45]


def load_splits(data_dir):
    arrs = {k: np.load(data_dir / (k + ".npy"))
            for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test")}
    label_map = json.load(open(data_dir / "label_map.json"))
    return arrs, label_map


def load_edges(data_dir, relations):
    edges = {}
    for rel in relations:
        f = data_dir / ("hin_edges_%s.npy" % rel)
        if f.exists():
            ed = np.load(f)
            if ed.shape[1] > 0:
                edges[rel] = ed
    return edges


def stack(arrs):
    X = np.concatenate([arrs["X_train"], arrs["X_val"], arrs["X_test"]], axis=0)
    y = np.concatenate([arrs["y_train"], arrs["y_val"], arrs["y_test"]], axis=0)
    n_tr, n_va = len(arrs["X_train"]), len(arrs["X_val"])
    masks = {}
    for key, lo, hi in (("train", 0, n_tr), ("val", n_tr, n_tr + n_va),
                        ("test", n_tr + n_va, len(X))):
        m = torch.zeros(len(X), dtype=torch.bool)
        m[lo:hi] = True
        masks[key] = m
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)), masks


def fit_neural(model, forward, y, masks, cfg, num_classes, device):
    # shared loop: class-weighted CE, Adam, early stopping on val macro-F1
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    weight = class_weights(y[masks["train"]].cpu().numpy(), num_classes, device) \
        if cfg.get("class_weight") == "balanced" else None
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    best_val, bad, best_state = -1.0, 0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); opt.zero_grad()
        loss = crit(forward()[masks["train"]], y[masks["train"]])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = forward()[masks["val"]].argmax(-1).cpu().numpy()
        val_f1 = f1_score(y[masks["val"]].cpu().numpy(), vp,
                          average="macro", zero_division=0)
        if val_f1 > best_val:
            best_val, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 15):
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yp = forward()[masks["test"]].argmax(-1).cpu().numpy()
    out = metrics(y[masks["test"]].cpu().numpy(), yp)
    out["best_val"] = float(best_val)
    return out


def run_han(data_dir, cfg, relations, seed):
    seed_all(seed)
    device = pick_device(cfg.get("device", "auto"))
    arrs, label_map = load_splits(data_dir)
    edges = load_edges(data_dir, relations)
    if not edges:
        raise RuntimeError("no edges found for %s" % relations)
    x, y, masks = stack(arrs)
    used = list(edges.keys())
    metadata = (["flow"], [("flow", r, "flow") for r in used])

    from torch_geometric.data import HeteroData
    data = HeteroData()
    data["flow"].x = x
    edge_index_dict = {("flow", r, "flow"): torch.from_numpy(e).long().to(device)
                       for r, e in edges.items()}
    x = x.to(device); y = y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    ccfg = cfg["classifier"]
    model = HAN(x.shape[1], ccfg["hidden"], len(label_map), metadata,
                num_layers=ccfg["num_layers"], heads=ccfg["heads"],
                dropout=ccfg["dropout"], residual=ccfg.get("residual", True)).to(device)
    forward = lambda: model({"flow": x}, edge_index_dict)
    m = fit_neural(model, forward, y, masks, ccfg, len(label_map), device)
    m["seed"] = seed
    m["relations"] = used
    return m


def run_gcn(data_dir, cfg, relations, seed):
    from torch_geometric.utils import to_undirected
    seed_all(seed)
    device = pick_device(cfg.get("device", "auto"))
    arrs, label_map = load_splits(data_dir)
    edges = load_edges(data_dir, relations)
    x, y, masks = stack(arrs)
    eds = [torch.from_numpy(e).long() for e in edges.values() if e.shape[1] > 0]
    if not eds:
        raise RuntimeError("no edges to union")
    edge_index = to_undirected(torch.cat(eds, dim=1), num_nodes=len(x)).to(device)
    x = x.to(device); y = y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    ccfg = cfg["classifier"]
    model = GCN(x.shape[1], ccfg["hidden"], len(label_map),
                dropout=ccfg["dropout"]).to(device)
    forward = lambda: model(x, edge_index)
    m = fit_neural(model, forward, y, masks, ccfg, len(label_map), device)
    m["seed"] = seed
    return m


def run_mlp(data_dir, cfg, seed):
    seed_all(seed)
    device = pick_device(cfg.get("device", "auto"))
    arrs, label_map = load_splits(data_dir)
    x, y, masks = stack(arrs)
    x = x.to(device); y = y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}
    ccfg = cfg["classifier"]
    model = MLP(x.shape[1], ccfg["hidden"], len(label_map),
                dropout=ccfg["dropout"]).to(device)
    forward = lambda: model(x)
    m = fit_neural(model, forward, y, masks, ccfg, len(label_map), device)
    m["seed"] = seed
    return m


def run_hgb(data_dir, seed, fit_train_only):
    from sklearn.ensemble import HistGradientBoostingClassifier
    arrs, label_map = load_splits(data_dir)
    if fit_train_only:
        X_fit, y_fit = arrs["X_train"], arrs["y_train"]
    else:
        X_fit = np.vstack([arrs["X_train"], arrs["X_val"]])
        y_fit = np.concatenate([arrs["y_train"], arrs["y_val"]])
    model = HistGradientBoostingClassifier(max_iter=400, random_state=seed,
                                           class_weight="balanced",
                                           early_stopping=False)
    model.fit(X_fit, y_fit)
    m = metrics(arrs["y_test"], model.predict(arrs["X_test"]))
    m["seed"] = seed
    return m


def endpoint_graph(data_dir):
    # nodes = src/dst endpoints, edges = flows (undirected message passing)
    import pandas as pd
    metas = [pd.read_csv(data_dir / f, usecols=lambda c: c in ("src_ip", "dst_ip"))
             for f in ("meta_train.csv", "meta_val.csv", "meta_test.csv")]
    meta = pd.concat(metas, ignore_index=True)
    nodes = {}
    src = np.zeros(len(meta), dtype=np.int64)
    dst = np.zeros(len(meta), dtype=np.int64)
    for i, (s, d) in enumerate(zip(meta["src_ip"].astype(str).values,
                                   meta["dst_ip"].astype(str).values)):
        # An absent address is kept on its own side of the graph rather than
        # merged with the other side's absent addresses. All five datasets
        # carry an address on every flow, so this branch does not fire on them.
        src[i] = nodes.setdefault(s if is_valid(s) else "s:" + s, len(nodes))
        dst[i] = nodes.setdefault(d if is_valid(d) else "d:" + d, len(nodes))
    return src, dst, len(nodes)


def run_egs(data_dir, cfg, seed, exclude_self=False):
    seed_all(seed)
    device = pick_device(cfg.get("device", "auto"))
    arrs, label_map = load_splits(data_dir)
    x, y, masks = stack(arrs)
    src, dst, n_nodes = endpoint_graph(data_dir)
    x = x.to(device); y = y.to(device)
    src = torch.from_numpy(src).to(device)
    dst = torch.from_numpy(dst).to(device)
    masks = {k: v.to(device) for k, v in masks.items()}
    ccfg = cfg["classifier"]
    model = EGraphSAGE(x.shape[1], ccfg["hidden"], len(label_map),
                       dropout=ccfg["dropout"],
                       exclude_self=exclude_self).to(device)
    forward = lambda: model(x, src, dst, n_nodes)
    m = fit_neural(model, forward, y, masks, ccfg, len(label_map), device)
    m["seed"] = seed
    m["exclude_self_edge"] = bool(exclude_self)
    return m


def han_sets(args, name):
    """Which relation sets to evaluate for HAN.

    Precedence: an explicit --sets string; otherwise the reported configuration
    in common.SELECTED (the set every table in the paper is computed with);
    otherwise, only when --summary names a beam-search summary, the best
    single relation / best combination / full pool read from that search.
    """
    if args.sets:
        return {"manual_%d" % (i + 1): [m.strip() for m in chunk.split("+") if m.strip()]
                for i, chunk in enumerate(args.sets.split(";"))}
    if not args.summary:
        if name not in SELECTED:
            raise SystemExit("%s has no entry in common.SELECTED; pass --sets or --summary" % name)
        return {"reported": list(SELECTED[name])}
    grand = json.load(open(args.summary))
    if name not in grand:
        raise SystemExit("%s not in %s" % (name, args.summary))
    v = grand[name]
    stage1 = sorted(v["stages"]["1"], key=lambda r: r["score"], reverse=True)
    best_single = stage1[0]["metapaths"]
    all_rows = [r for rows in v["stages"].values() for r in rows]
    best_combo = max(all_rows, key=lambda r: r["score"])["metapaths"]
    sets = {"best_single": best_single, "full_pool": list(v["pool"])}
    if sorted(best_combo) != sorted(v["pool"]):
        sets["best_combo"] = best_combo
    return sets


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", choices=["han", "gcn", "mlp", "hgb", "egs"], required=True)
    ap.add_argument("--egs-exclude-self", action="store_true",
                    help="E-GraphSAGE only: remove a flow's own edge from that flow's "
                         "prediction at the last layer. Earlier layers are untouched, so "
                         "the effect is a lower bound on the self-prediction bypass")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--sets", default=None,
                    help='han only: "rel1+rel2;rel3" (";" separates sets)')
    ap.add_argument("--summary", default=None,
                    help="han only: a beam-search summary (e.g. results/saturation/<dataset>/"
                         "combinatorial_grand_summary.json) to derive best-single / best-combo / "
                         "full-pool sets from. Without --sets and --summary the reported "
                         "configuration in common.SELECTED is used")
    ap.add_argument("--fit-train-only", action="store_true",
                    help="hgb only: fit on train instead of train+val")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_root, runs_root = Path(args.data), Path(args.runs)
    runs_root.mkdir(parents=True, exist_ok=True)

    fname = {"han": "multiseed_summary.json",
             "gcn": "homogeneous_gcn_summary.json",
             "mlp": "mlp_baseline_summary.json",
             "egs": "egraphsage_summary.json",
             "hgb": "tabular_baseline%s_summary.json"
                    % ("_trainonly" if args.fit_train_only else "")}[args.model]
    out_path = runs_root / fname
    grand = json.load(open(out_path)) if out_path.exists() else {}

    for name in args.datasets:
        data_dir = data_root / name
        print("== %s / %s" % (args.model, name))
        # `data_root` is the string this command was given, not the contents of
        # that directory. Directories get rebuilt and the string does not change
        # with them, which is how a table of numbers once came to name a graph
        # it had never been run on. The fingerprint is the contents.
        res = {"data_root": str(data_root), "seeds": args.seeds,
               "graph": graph_fingerprint(data_dir)}

        if args.model == "han":
            res["sets"] = {}
            for tag, rels in han_sets(args, name).items():
                rows = []
                for s in args.seeds:
                    print("  set=%s seed=%d" % (tag, s))
                    try:
                        rows.append(run_han(data_dir, cfg, rels, s))
                    except Exception as e:
                        print("  [error] %s" % e)
                        rows.append({"seed": s, "macro_f1": -1.0})
                entry = {"metapaths": rels, "per_seed": rows}
                for k in ("macro_f1", "minority_f1", "weighted_f1"):
                    entry[k] = agg([float(r.get(k, -1.0)) for r in rows])
                res["sets"][tag] = entry
        else:
            rows = []
            for s in args.seeds:
                print("  seed=%d" % s)
                try:
                    if args.model == "gcn":
                        m = run_gcn(data_dir, cfg, cfg["hin"]["metapaths"], s)
                    elif args.model == "mlp":
                        m = run_mlp(data_dir, cfg, s)
                    elif args.model == "egs":
                        m = run_egs(data_dir, cfg, s,
                                    exclude_self=args.egs_exclude_self)
                    else:
                        m = run_hgb(data_dir, s, args.fit_train_only)
                except Exception as e:
                    print("  [error] %s" % e)
                    m = {"seed": s, "macro_f1": -1.0}
                rows.append(m)
            res["per_seed"] = rows
            for k in ("macro_f1", "minority_f1", "weighted_f1"):
                res[k] = agg([float(r.get(k, -1.0)) for r in rows])

        grand[name] = res
        json.dump(grand, open(out_path, "w"), indent=2)
        print("saved %s" % out_path)


if __name__ == "__main__":
    main()
