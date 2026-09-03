# -*- coding: utf-8 -*-
"""Repeated split, cutoff and edge-sampling evaluation.

The paper claims that the split protocol changes scores and asks whether it
also changes the relation ranking. A single realisation of each protocol cannot
answer that, so this script varies one source of uncertainty at a time around
the reported configuration.

  A. random    split_mode=random, varying the split seed
  B. cutoff    split_mode=time_stratified, varying the test fraction
  Seeds are decoupled: a random-split variant redraws the split with its own
  seed and then rebuilds the edges with the fixed base edge seed, so only the
  edge-seed arm changes the edge sampling. (Earlier revisions of this script
  let build_graph.py sample the edges with the split seed; see README, "Random
  variants and edge seeds".)
  C. edgeseed  split fixed, varying only the edge-sampling seed

The edge seed is held fixed in A and the split is held fixed in C, so the three
sources do not mix.

Each variant is a full rebuild: `build_graph.py` writes the split and the edges,
then each model in turn is trained on it by `train.py`. Nothing here reuses a
graph across variants, because a variant that reused one would be measuring the
old split. The results are in results/repeated_splits/.

--- validation ---
Running a command is not treated as evidence that it worked. Each step checks
its own output and stops when it does not match.

  1) After materialisation the flow count is compared with the reference
     directory. BCCC-DoH and CIC-AndMal are read from the packet captures
     (the released CSVs hold a different corpus), so a count that differs
     from the reference means the adapter read a different source.
  2) Edge construction is confirmed through the used list of hin_summary.json.
     An empty list leaves the graph model with no metapaths and it will still
     exit cleanly. The seed recorded in hin_summary.json must equal the
     variant's edge seed; this is the check that the split seed did not leak
     into the edge sampling.

--- provenance ---
Every variant in repeat_summary.json carries a `protocol` block:
split_mode, split_seed, edge_seed, test_size, and edge_seed_recorded (the seed
read back from hin_summary.json after the build). --rebuild-summary restores
the block from the `_cfg/` files of the run tree; a random-split variant from a
run that predates the seed decoupling has no `_edge.yaml` and is marked
legacy_coupled_seed=true instead of being guessed.
  3) Every model summary is checked for existence and for a valid macro-F1.
     A script can exit with status zero and leave a failed result behind.

Run --check first to take one variant end to end through those three tests.

Usage:
    python repeat_splits.py --datasets iscx_vpn --data data/processed_deg2 \
        --check
    python repeat_splits.py --datasets iscx_vpn vnat \
        --data data/processed_deg2 --models full --rank --runs runs/repeat
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from common import ALL_RELATIONS, SELECTED

RELATIONS = list(ALL_RELATIONS)

MODEL_SETS = {"full": ["hgb", "mlp", "gcn", "han", "egs"],
              # Minimum set for a per-split decomposition. HGB is deterministic
              # and serves to confirm the split is identical; the other three
              # produce the decomposition.
              "decomp": ["hgb", "mlp", "noedge", "han"],
              "light": ["hgb", "han"]}

# HAN reported configuration (Table 4), shared by both split protocols. The sets live
# in common.py so that a change reaches every script that uses them.
SELECTED_SET = {k: "+".join(v) for k, v in SELECTED.items()}

# BCCC-DoH and CIC-AndMal are rebuilt from the packet captures rather than from
# the released CSV. The two disagree: the BCCC CSV holds 499,106 rows and the
# captures yield 505,040 flows. `build_graph.py` always reads the captures for
# these two; the extracted flow table is cached next to the captures
# (datasets.extract_pcaps), so only the first variant pays for the extraction.
NEEDS_PCAP = {"bccc_dohbrw", "cic_andmal"}

# Summary file each model script actually writes.
SUMMARY_NAME = {
    "hgb": "tabular_baseline_summary.json",
    "mlp": "mlp_baseline_summary.json",
    "gcn": "homogeneous_gcn_summary.json",
    "han": "multiseed_summary.json",
    "noedge": "%s/multiseed_summary.json",   # under <out>/<dataset>/
    "egs": "egraphsage_summary.json",
}

BASE_SEED, BASE_TEST, BASE_VAL = 42, 0.20, 0.10
DROPOUT = 0.5      # shared by the four neural models, Sec. 5.1


class StepFailed(RuntimeError):
    pass


def sh(cmd, dry):
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    rc = subprocess.call(cmd)
    if rc != 0:
        raise StepFailed("exit %d: %s" % (rc, " ".join(str(c) for c in cmd)))
    return rc


def canonical_counts(ref_root: Path, ds: str):
    d = ref_root / ds
    if not d.is_dir():
        return None
    try:
        return sum(len(np.load(d / ("y_%s.npy" % s)))
                   for s in ("train", "val", "test"))
    except Exception:
        return None


def check_materialized(vdir: Path, ds: str, expect_n, expect_edge_seed=None):
    """Verify the flow count, that edges were built, and the edge seed used."""
    d = vdir / ds
    got = sum(len(np.load(d / ("y_%s.npy" % s))) for s in ("train", "val", "test"))
    if expect_n is not None and got != expect_n:
        raise StepFailed(
            "%s: %d flows against %d in the reference. The adapter read a "
            "different source (BCCC-DoH and CIC-AndMal must come from the "
            "packet captures under --raw)." % (ds, got, expect_n))
    hs = d / "hin_summary.json"
    if not hs.is_file():
        raise StepFailed("%s: hin_summary.json missing" % ds)
    summ = json.load(open(hs))
    used = summ.get("used") or []
    if not used:
        raise StepFailed("%s: no edges were built (used=[]). The metadata "
                         "has no column for any requested relation." % ds)
    recorded = summ.get("seed")
    if expect_edge_seed is not None and int(recorded) != int(expect_edge_seed):
        raise StepFailed(
            "%s: hin_summary.json records edge seed %s but the variant's edge "
            "seed is %s. The split seed leaked into the edge sampling."
            % (ds, recorded, expect_edge_seed))
    print("    [ok] flows=%d, relations=%d %s, edge seed=%s"
          % (got, len(used), [r.replace('via_', '') for r in used], recorded))
    return got, used, recorded


def summary_path(rdir, model, dataset):
    """Path of the summary file a model actually writes.

    Only `no_edge_han.py` writes under `<out>/<dataset>/`; the others write
    directly under `--runs`. Keeping that difference in one place avoids a
    "file not found" stop when the self-loop result is looked up.
    """
    tail = SUMMARY_NAME[model]
    return rdir / model / (tail % dataset if "%s" in tail else tail)


def read_macro(path, dataset, required=True):
    p = Path(path)
    if not p.is_file():
        if required:
            raise StepFailed("result file missing: %s" % p)
        return None
    node = json.load(open(p))
    node = node.get(dataset, node)
    for key in ("per_seed", "runs", "sets"):
        if isinstance(node, dict) and key in node:
            node = node[key]
            break
    vals = []

    def walk(o):
        if isinstance(o, dict):
            v = o.get("macro_f1")
            if isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                for x in o.values():
                    walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(node)
    vals = [v for v in vals if v is not None]
    if not vals:
        raise StepFailed("no macro_f1 in result: %s" % p)
    if min(vals) < 0:
        raise StepFailed("result holds a failure value (macro_f1=%.1f): %s"
                         % (min(vals), p))
    # per_seed is kept at the precision the run produced. A mean taken over
    # values that have already been rounded to four places can land exactly on
    # a rounding boundary: the ten HIKARI random splits average to 0.70655,
    # where the fourth place is decided by the rounding rule rather than by the
    # measurement. Averaging the seeds instead removes the tie, and any later
    # aggregate is computed from this list.
    return {"mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4), "n": len(vals),
            "per_seed": [float(v) for v in vals]}


def protocol_from_tag(tag):
    """(split_mode, split_seed, test_size, edge_seed) implied by a variant tag.

    The tag is the only name a variant has, so it must be decodable; the
    inverse of the naming in `variants()`.
    """
    if tag.startswith("random_s"):
        s = int(tag[len("random_s"):])
        return "random", s, BASE_TEST, BASE_SEED
    if tag.startswith("cutoff_t"):
        digits = tag[len("cutoff_t"):]            # "015" -> 0.15, "0175" -> 0.175
        return "time_stratified", BASE_SEED, float("0." + digits[1:]), BASE_SEED
    if tag.startswith("edgeseed_s"):
        return "time_stratified", BASE_SEED, BASE_TEST, int(tag[len("edgeseed_s"):])
    raise StepFailed("unrecognised variant tag %s" % tag)


def protocol_record(mode, sseed, tsize, eseed, recorded, source):
    return {"split_mode": mode, "split_seed": int(sseed), "test_size": float(tsize),
            "edge_seed": int(eseed),
            "edge_seed_recorded": (None if recorded is None else int(recorded)),
            "edge_seed_source": source,
            "legacy_coupled_seed": False, "protocol_inferred": False}


def restore_protocol(runs: Path, ds: str, tag: str):
    """Protocol block for a variant of an existing run tree.

    Preference order: the `_cfg/<ds>_<tag>_protocol.json` written by this
    script at run time (exact); else the `_cfg/` yaml files. A random-split
    variant whose split seed differs from the base seed and whose tree holds no
    `_edge.yaml` was built before the seeds were decoupled, so its edges were
    sampled with the split seed: recorded as legacy_coupled_seed=true rather
    than guessed.
    """
    cfgdir = runs / "_cfg"
    exact = cfgdir / ("%s_%s_protocol.json" % (ds, tag))
    if exact.is_file():
        return json.load(open(exact))
    mode, sseed, tsize, eseed = protocol_from_tag(tag)
    rec = protocol_record(mode, sseed, tsize, eseed, None, "tag")
    rec["protocol_inferred"] = True
    edge_yaml = cfgdir / ("%s_%s_edge.yaml" % (ds, tag))
    if mode == "random" and int(sseed) != BASE_SEED:
        if edge_yaml.is_file():
            rec["edge_seed"] = int(yaml.safe_load(open(edge_yaml))["seed"])
            rec["edge_seed_source"] = "_cfg/%s" % edge_yaml.name
        else:
            rec["edge_seed"] = int(sseed)
            rec["edge_seed_source"] = "none: run predates the seed decoupling"
            rec["legacy_coupled_seed"] = True
    elif mode == "random":
        rec["edge_seed_source"] = "tag (split seed equals the base edge seed)"
    return rec


def rebuild_summary(runs: Path, datasets, models):
    """Rewrite repeat_summary.json from a run tree that already exists.

    The per-model summaries are written by train.py and stay on disk, so the
    aggregate can be rebuilt without training anything. This is how a summary
    written before per_seed was recorded is brought up to date, and it is also
    the check that the aggregate really is a function of the run outputs and
    not of anything that happened only once. The protocol block is restored
    from the `_cfg/` files (see restore_protocol).
    """
    report = {}
    for ds in datasets:
        dsdir = runs / ds
        if not dsdir.is_dir():
            raise StepFailed("no run directory for %s under %s" % (ds, runs))
        report[ds] = {}
        for vdir in sorted(p for p in dsdir.iterdir() if p.is_dir()):
            res = {"protocol": restore_protocol(runs, ds, vdir.name)}
            for m in models:
                f = summary_path(vdir, m, ds)
                if f.is_file():
                    res[m] = read_macro(f, ds)
            rank_dir = vdir / "rank"
            if rank_dir.is_dir():
                rank = {}
                for rel in RELATIONS:
                    f = rank_dir / rel / "multiseed_summary.json"
                    try:
                        rank[rel] = read_macro(f, ds) if f.is_file() else None
                    except StepFailed:
                        rank[rel] = None
                if any(v is not None for v in rank.values()):
                    res["rank"] = rank
            if len(res) > 1:
                report[ds][vdir.name] = res
                pr = res["protocol"]
                print("  %-12s %-18s %-22s split=%-5d edge=%-5d%s" % (
                    ds, vdir.name,
                    " ".join(sorted(k for k in res if k not in ("rank", "protocol"))),
                    pr["split_seed"], pr["edge_seed"],
                    "  LEGACY coupled seed" if pr["legacy_coupled_seed"] else ""))
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / "repeat_summary.json"
    json.dump(report, open(out, "w"), indent=2)
    print("[saved] %s" % out)
    return report


def variants(args):
    """(tag, split_mode, split_seed, test_size, edge_seed, materialize)"""
    out = []
    rng = np.random.RandomState(0)
    seeds = [BASE_SEED] + list(rng.randint(1, 10000, size=max(args.n_random - 1, 0)))
    for s in seeds:
        out.append(("random_s%d" % s, "random", int(s), BASE_TEST, BASE_SEED, True))
    for t in args.cutoffs:
        out.append(("cutoff_t%s" % str(t).replace(".", ""),
                    "time_stratified", BASE_SEED, float(t), BASE_SEED, True))
    eseeds = [BASE_SEED] + list(rng.randint(1, 10000, size=max(args.n_edge - 1, 0)))
    for i, s in enumerate(eseeds):
        out.append(("edgeseed_s%d" % s, "time_stratified", BASE_SEED,
                    BASE_TEST, int(s), i == 0))
    return out


def write_cfg(base_cfg, path, seed, test_size, metapaths=None):
    cfg = yaml.safe_load(open(base_cfg, encoding="utf-8"))
    cfg["seed"] = int(seed)
    cfg.setdefault("data", {})
    cfg["data"]["test_size"] = float(test_size)
    cfg["data"]["val_size"] = BASE_VAL
    if metapaths is not None:
        cfg["hin"]["metapaths"] = list(metapaths)
    yaml.safe_dump(cfg, open(path, "w", encoding="utf-8"), allow_unicode=True)
    return path


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--data", default="data/processed_deg2",
                    help="the canonical root, used as the reference the "
                         "flow count of each redrawn split must match")
    ap.add_argument("--work", default="data/_repeat")
    ap.add_argument("--runs", default="runs/repeat")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", choices=list(MODEL_SETS), default="full")
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-edge", type=int, default=5)
    ap.add_argument("--cutoffs", nargs="+", type=float,
                    default=[0.15, 0.175, 0.20, 0.225, 0.25])
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="rebuild the summary from scratch instead of resuming")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="take one variant end to end and stop")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="rebuild repeat_summary.json from a run tree that "
                         "already exists, training nothing")
    args = ap.parse_args()

    py = sys.executable
    work, out, ref = Path(args.work), Path(args.runs), Path(args.data)

    if args.rebuild_summary:
        rebuild_summary(out, args.datasets, MODEL_SETS[args.models])
        return

    out.mkdir(parents=True, exist_ok=True)
    tmpcfg = out / "_cfg"; tmpcfg.mkdir(exist_ok=True)

    vs = variants(args)
    if args.check:
        vs = vs[:1]
        args.datasets = args.datasets[:1]
    models = MODEL_SETS[args.models]

    print("variants=%d  datasets=%s  models=%s  rank=%s"
          % (len(vs), args.datasets, models, args.rank))
    for ds in args.datasets:
        n = canonical_counts(ref, ds)
        print("  reference %s: %s flows%s" % (ds, n,
              "  (rebuilt from the captures under --raw)" if ds in NEEDS_PCAP else ""))
    if args.dry_run:
        for t, m, s, ts, es, mat in vs:
            print("  %-16s split=%-16s seed=%-5d test=%.3f edge=%-5d mat=%s"
                  % (t, m, s, ts, es, mat))
        return

    # Resume. The summary is rewritten after every variant, so an interrupted
    # run leaves its finished variants on disk. The arguments cannot simply be
    # reduced to the remaining variants, because `variants()` draws from one
    # generator in order and changing `--n-random` changes the edge seeds drawn
    # after it. So the arguments stay and variants already in the summary are
    # skipped. For BCCC-DoH a variant rebuilds the split from 505,040 flows
    # (the capture extraction itself is cached, see datasets.extract_pcaps).
    sfile = out / "repeat_summary.json"
    report = json.load(open(sfile)) if (sfile.exists() and not args.fresh) else {}
    done = {d: set(v) for d, v in report.items()}
    if done:
        print("resuming: variants already done %s"
              % {d: len(v) for d, v in done.items()})
    try:
        for ds in args.datasets:
            expect_n = canonical_counts(ref, ds)
            report.setdefault(ds, {})
            for tag, mode, sseed, tsize, eseed, mat in vs:
                if tag in done.get(ds, ()):
                    print("\n== %s / %s  [skipped: already present]" % (ds, tag))
                    continue
                vdir = work / ("%s__%s" % (ds, tag))
                rdir = out / ds / tag
                rdir.mkdir(parents=True, exist_ok=True)
                print("\n== %s / %s" % (ds, tag), flush=True)

                if mat:
                    cfg_p = write_cfg(args.config,
                                      tmpcfg / ("%s_%s_split.yaml" % (ds, tag)),
                                      sseed, tsize)
                    # Re-draw the split from the source and build the edges;
                    # build_graph.py does both (materialize + build).
                    cmd = [py, "-u", "build_graph.py", "--datasets", ds,
                           "--raw", args.raw, "--out", str(vdir),
                           "--config", str(cfg_p), "--split-mode", mode]
                    sh(cmd, False)
                    if int(eseed) != int(sseed):
                        # build_graph.py uses one cfg["seed"] for both the split
                        # and the edge rings, so the pass above sampled the edges
                        # with the split seed. Rebuild the edges alone with the
                        # fixed edge seed so that a random-split variant changes
                        # the split and nothing else; the edge-seed variants
                        # below are the only arm that changes the sampling.
                        cfg_e = write_cfg(args.config,
                                          tmpcfg / ("%s_%s_edge.yaml" % (ds, tag)),
                                          eseed, tsize)
                        sh([py, "-u", "build_graph.py", "--datasets", ds,
                            "--out", str(vdir), "--config", str(cfg_e),
                            "--split-mode", mode, "--skip-materialize"], False)
                else:
                    src = work / ("%s__edgeseed_s%d" % (ds, BASE_SEED))
                    (vdir / ds).mkdir(parents=True, exist_ok=True)
                    for p in (src / ds).glob("*"):
                        if p.is_file() and not p.name.startswith("hin_edges_"):
                            shutil.copy2(p, vdir / ds / p.name)
                    cfg_p = write_cfg(args.config,
                                      tmpcfg / ("%s_%s_edge.yaml" % (ds, tag)),
                                      eseed, tsize)
                    # Keep the split and rebuild only the edges with another
                    # seed. The split files were copied above and are not
                    # regenerated; without --skip-materialize the split would be
                    # redrawn and the variant would no longer measure an
                    # edge-seed-only change.
                    sh([py, "-u", "build_graph.py", "--datasets", ds,
                        "--out", str(vdir), "--config", str(cfg_p),
                        "--split-mode", mode, "--skip-materialize"], False)

                _, _, recorded = check_materialized(vdir, ds, expect_n, eseed)
                # Provenance of this variant, written with the results and to
                # _cfg/ so that --rebuild-summary can restore it exactly.
                proto = protocol_record(mode, sseed, tsize, eseed, recorded,
                                        "hin_summary.json")
                json.dump(proto, open(tmpcfg / ("%s_%s_protocol.json" % (ds, tag)), "w"),
                          indent=2)

                cfg_p = write_cfg(args.config,
                                  tmpcfg / ("%s_%s_run.yaml" % (ds, tag)),
                                  eseed, tsize)
                seeds = [str(s) for s in args.seeds]
                # All five models go through train.py. Calling a different
                # script per model would hide one of them running with a
                # different configuration.
                def runner(model, extra=()):
                    return [py, "-u", "train.py", "--model", model,
                            "--datasets", ds, "--data", str(vdir),
                            "--runs", str(rdir / model),
                            "--config", str(cfg_p),
                            "--seeds"] + seeds + list(extra)

                runners = {
                    "hgb": runner("hgb"),
                    "mlp": runner("mlp"),
                    "gcn": runner("gcn"),
                    "han": runner("han", ("--sets", SELECTED_SET[ds])),
                    "egs": runner("egs"),
                    # Self-loop control: the only model that goes through a
                    # dedicated script rather than train.py; it uses the same
                    # config and the same relation list.
                    #
                    # It must be in the list for a per-split decomposition. HAN
                    # and MLP alone give the margin but not whether it came from
                    # the architecture or the edges, and that ratio is the paper's
                    # headline.
                    "noedge": [py, "-u", "no_edge_han.py",
                               "--datasets", ds, "--data", str(vdir),
                               "--out", str(rdir / "noedge"),
                               "--config", str(cfg_p),
                               "--sets", SELECTED_SET[ds],
                               "--seeds"] + seeds,
                }
                res = {"protocol": proto}
                for m in models:
                    sh(runners[m], False)
                    res[m] = read_macro(summary_path(rdir, m, ds), ds)
                    print("    [ok] %-4s %.4f +- %.4f" % (m, res[m]["mean"], res[m]["std"]))

                if args.rank:
                    rank = {}
                    for rel in RELATIONS:
                        rcfg = write_cfg(args.config,
                                         tmpcfg / ("%s_%s_%s.yaml" % (ds, tag, rel)),
                                         eseed, tsize, metapaths=[rel])
                        rrun = rdir / "rank" / rel
                        sh([py, "-u", "train.py", "--model", "han",
                            "--datasets", ds, "--data", str(vdir),
                            "--runs", str(rrun), "--config", str(rcfg),
                            "--sets", rel, "--seeds"] + seeds, False)
                        # a relation absent from this dataset is expected to fail
                        try:
                            rank[rel] = read_macro(rrun / "multiseed_summary.json", ds)
                        except StepFailed:
                            rank[rel] = None
                    res["rank"] = rank

                report[ds][tag] = res
                json.dump(report, open(out / "repeat_summary.json", "w"), indent=2)

                if not args.keep_data and tag != "edgeseed_s%d" % BASE_SEED:
                    shutil.rmtree(vdir, ignore_errors=True)

                if args.check:
                    print("\n[check] one variant passed every test; the full run can proceed.")
                    return
    except StepFailed as e:
        print("\n!! stopped: %s" % e)
        sys.exit(2)

    json.dump(report, open(out / "repeat_summary.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "repeat_summary.json"))


if __name__ == "__main__":
    main()
