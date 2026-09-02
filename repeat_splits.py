# -*- coding: utf-8 -*-
"""Repeated split, cutoff and edge-sampling evaluation.

The paper claims that the split protocol changes scores and asks whether it
also changes the relation ranking. A single realisation of each protocol cannot
answer that, so this script varies one source of uncertainty at a time around
the reported configuration.

  A. random    split_mode=random, varying the split seed
  B. cutoff    split_mode=time_stratified, varying the test fraction
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
     directory. The adapters of BCCC-DoH and CIC-AndMal default to the released
     CSV in the development pipeline, so rebuilding them without --from-pcap
     silently produces a different corpus from the one the paper reports.
  2) Edge construction is confirmed through the used list of hin_summary.json.
     An empty list leaves the graph model with no metapaths and it will still
     exit cleanly.
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
              # 분할마다 3분해를 내는 데 필요한 최소 집합. HGB 는 결정적이라
              # 분할이 같은지 확인하는 데 쓰이고, 나머지 셋이 분해를 만든다.
              "decomp": ["hgb", "mlp", "noedge", "han"],
              "light": ["hgb", "han"]}

# HAN selected set of Table 7, shared by both split protocols. The sets live
# in common.py so that a change reaches every script that uses them.
SELECTED_SET = {k: "+".join(v) for k, v in SELECTED.items()}

# BCCC-DoH and CIC-AndMal are rebuilt from the packet captures rather than from
# the released CSV. The two disagree: the BCCC CSV holds 499,106 rows and the
# captures yield 505,040 flows. `build_graph.py` always reads the captures for
# these two, so repeating a split for them costs an extraction each time.
NEEDS_PCAP = {"bccc_dohbrw", "cic_andmal"}

# Summary file each model script actually writes.
SUMMARY_NAME = {
    "hgb": "tabular_baseline_summary.json",
    "mlp": "mlp_baseline_summary.json",
    "gcn": "homogeneous_gcn_summary.json",
    "han": "multiseed_summary.json",
    "noedge": "%s/multiseed_summary.json",   # <out>/<dataset>/ 아래
    "egs": "egraphsage_summary.json",
}

BASE_SEED, BASE_TEST, BASE_VAL = 42, 0.20, 0.10
DROPOUT = 0.5      # shared by the four neural models, Table 6


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


def check_materialized(vdir: Path, ds: str, expect_n):
    """Verify the flow count and that edges were built."""
    d = vdir / ds
    got = sum(len(np.load(d / ("y_%s.npy" % s))) for s in ("train", "val", "test"))
    if expect_n is not None and got != expect_n:
        raise StepFailed(
            "%s: %d flows against %d in the reference. The adapter read a "
            "different source. Check whether this dataset needs --from-pcap."
            % (ds, got, expect_n))
    hs = d / "hin_summary.json"
    if not hs.is_file():
        raise StepFailed("%s: hin_summary.json missing" % ds)
    used = json.load(open(hs)).get("used") or []
    if not used:
        raise StepFailed("%s: no edges were built (used=[]). The metadata "
                         "has no column for any requested relation." % ds)
    print("    [ok] flows=%d, relations=%d %s" % (got, len(used),
                                                 [r.replace('via_', '') for r in used]))
    return got, used


def summary_path(rdir, model, dataset):
    """모델이 실제로 쓰는 요약 파일의 경로.

    `no_edge_han.py` 만 `<out>/<dataset>/` 아래에 쓴다. 나머지는 `--runs` 바로
    아래다. 이 차이를 한 곳에 모아 두지 않으면 자기루프 결과를 찾지 못한 채
    "파일 없음" 으로 멈춘다.
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


def rebuild_summary(runs: Path, datasets, models):
    """Rewrite repeat_summary.json from a run tree that already exists.

    The per-model summaries are written by train.py and stay on disk, so the
    aggregate can be rebuilt without training anything. This is how a summary
    written before per_seed was recorded is brought up to date, and it is also
    the check that the aggregate really is a function of the run outputs and
    not of anything that happened only once.
    """
    report = {}
    for ds in datasets:
        dsdir = runs / ds
        if not dsdir.is_dir():
            raise StepFailed("no run directory for %s under %s" % (ds, runs))
        report[ds] = {}
        for vdir in sorted(p for p in dsdir.iterdir() if p.is_dir()):
            res = {}
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
            if res:
                report[ds][vdir.name] = res
                print("  %-12s %-18s %s" % (ds, vdir.name,
                                            " ".join(sorted(k for k in res
                                                            if k != "rank"))))
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
                    help="이어받지 않고 요약을 처음부터 다시 만든다")
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
              "  (--from-pcap required)" if ds in NEEDS_PCAP else ""))
    if args.dry_run:
        for t, m, s, ts, es, mat in vs:
            print("  %-16s split=%-16s seed=%-5d test=%.3f edge=%-5d mat=%s"
                  % (t, m, s, ts, es, mat))
        return

    # 이어받기. 변형 하나가 끝날 때마다 요약을 다시 쓰므로, 중간에 끊긴
    # 실행의 결과는 디스크에 남아 있다. 그런데 인자를 줄여 남은 것만 돌릴 수는
    # 없다 — `variants()` 가 난수 생성기 하나를 순서대로 쓰기 때문에
    # `--n-random` 을 바꾸면 그 뒤에 뽑히는 엣지 시드 값 자체가 달라진다.
    # 그래서 인자는 그대로 두고, 이미 요약에 있는 변형만 건너뛴다.
    # BCCC-DoH 는 변형 하나가 캡처 13,754 개의 재추출을 요구하므로 이 차이가 크다.
    sfile = out / "repeat_summary.json"
    report = json.load(open(sfile)) if (sfile.exists() and not args.fresh) else {}
    done = {d: set(v) for d, v in report.items()}
    if done:
        print("이어받기: 이미 끝난 변형 %s"
              % {d: len(v) for d, v in done.items()})
    try:
        for ds in args.datasets:
            expect_n = canonical_counts(ref, ds)
            report.setdefault(ds, {})
            for tag, mode, sseed, tsize, eseed, mat in vs:
                if tag in done.get(ds, ()):
                    print("\n== %s / %s  [건너뜀 — 이미 있음]" % (ds, tag))
                    continue
                vdir = work / ("%s__%s" % (ds, tag))
                rdir = out / ds / tag
                rdir.mkdir(parents=True, exist_ok=True)
                print("\n== %s / %s" % (ds, tag), flush=True)

                if mat:
                    cfg_p = write_cfg(args.config,
                                      tmpcfg / ("%s_%s_split.yaml" % (ds, tag)),
                                      sseed, tsize)
                    # 원본에서 분할을 다시 뽑고 엣지를 만든다. build_graph.py
                    # 가 두 단계를 다 한다(materialize + build).
                    cmd = [py, "-u", "build_graph.py", "--datasets", ds,
                           "--raw", args.raw, "--out", str(vdir),
                           "--config", str(cfg_p), "--split-mode", mode]
                    sh(cmd, False)
                else:
                    src = work / ("%s__edgeseed_s%d" % (ds, BASE_SEED))
                    (vdir / ds).mkdir(parents=True, exist_ok=True)
                    for p in (src / ds).glob("*"):
                        if p.is_file() and not p.name.startswith("hin_edges_"):
                            shutil.copy2(p, vdir / ds / p.name)
                    cfg_p = write_cfg(args.config,
                                      tmpcfg / ("%s_%s_edge.yaml" % (ds, tag)),
                                      eseed, tsize)
                    # 분할은 그대로 두고 엣지만 다른 시드로 다시 만든다.
                    # 분할 파일은 위에서 복사했으므로 다시 만들지 않는다.
                    # --skip-materialize 를 빼면 분할이 새로 뽑혀 이 변형이
                    # 재는 것(엣지 시드만 바뀐 그래프)이 아니게 된다.
                    sh([py, "-u", "build_graph.py", "--datasets", ds,
                        "--out", str(vdir), "--config", str(cfg_p),
                        "--split-mode", mode, "--skip-materialize"], False)

                check_materialized(vdir, ds, expect_n)

                cfg_p = write_cfg(args.config,
                                  tmpcfg / ("%s_%s_run.yaml" % (ds, tag)),
                                  eseed, tsize)
                seeds = [str(s) for s in args.seeds]
                # 다섯 모델이 모두 train.py 를 지난다. 모델마다 다른
                # 스크립트를 부르면 어느 하나가 다른 설정으로 도는 것을
                # 알아채지 못한다.
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
                    # 자기루프 통제. `train.py` 가 아니라 전용 스크립트를 지나는
                    # 유일한 모델인데, 그 스크립트가 같은 config 와 같은 관계
                    # 목록을 쓰므로 다른 설정으로 도는 일은 없다.
                    #
                    # 이것이 목록에 있어야 분할마다 3분해를 낼 수 있다. HAN 과
                    # MLP 만 있으면 마진은 나오지만 그 마진이 아키텍처에서 왔는지
                    # 엣지에서 왔는지는 나오지 않는다 — 논문의 헤드라인이 바로
                    # 그 비율이다.
                    "noedge": [py, "-u", "no_edge_han.py",
                               "--datasets", ds, "--data", str(vdir),
                               "--out", str(rdir / "noedge"),
                               "--config", str(cfg_p),
                               "--sets", SELECTED_SET[ds],
                               "--seeds"] + seeds,
                }
                res = {}
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
