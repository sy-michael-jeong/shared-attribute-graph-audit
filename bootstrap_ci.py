# -*- coding: utf-8 -*-
"""Bootstrap intervals for the model comparisons (Sec. 5.3, 6.1).

With only five seeds the Wilcoxon signed-rank test cannot produce a p-value
below 0.0625, so effect sizes and intervals are reported instead of
null-hypothesis tests.

Two intervals are given.

  paired    resamples the seed-wise differences, treating the seeds as matched
            across the two models
  unpaired  resamples the two seed sets independently, assuming no matching

Sign consistency is the fraction of seeds on which the difference keeps the
same direction.

Usage:
    python bootstrap_ci.py --results results --out results/model_comparison
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

# Model comparisons.
PAIRS = [("han", "mlp"), ("han", "gcn"), ("gcn", "mlp"), ("han", "egraphsage")]

# Control comparisons. Each isolates one source of the HAN-over-MLP margin.
#   han - noedge    what the edges carry
#   noedge - mlp    what the architecture and the relation count carry
#   han - masked    what the edges that cross a partition boundary carry
#   gcn - noedge    whether a union graph beats having no edges at all
CONTROL_PAIRS = [("han", "noedge"), ("noedge", "mlp"),
                 ("han", "masked"), ("gcn", "noedge")]

DATASETS = ["bccc_dohbrw", "iscx_vpn", "hikari", "cic_andmal", "vnat"]
PERM_SEEDS = [41, 42, 43, 44, 45]

# Accuracy alone hides effects on the minority class, so both are reported.
METRICS = ["macro_f1", "minority_f1"]


def _clean(per_seed, metric):
    """Scores of the seeds that finished, in seed order.

    A seed that raised is written back as macro_f1 -1.0 rather than dropped,
    so that the seed list stays aligned with the requested seeds. Reading the
    raw list would average that sentinel into the interval. One failed seed
    out of five moves a mean of 0.915 to 0.532 and an edge share of 53% to
    160%, and nothing in the output says so.

    A seed is also dropped when it lacks the metric, which happens when the
    partition held one class and the minority score was never computed.
    """
    out = []
    for p in per_seed:
        if float(p.get("macro_f1", -1.0)) < 0:
            continue
        if metric not in p:
            continue
        out.append(float(p[metric]))
    return out


def seeds_from(path: Path, dataset: str, metric: str = "macro_f1"):
    """Seed-wise scores of one dataset from a summary file."""
    if not path.is_file():
        return None
    d = json.load(open(path))
    node = d.get(dataset)
    if node is None:
        return None
    if "sets" in node:
        # More than one set means the run was not the reported configuration.
        # Taking whichever comes first would silently compare a different
        # relation set, so the caller is told instead.
        if len(node["sets"]) > 1:
            print("  [warn] %s holds %d sets (%s). Skipped, since the reported "
                  "configuration is ambiguous."
                  % (path, len(node["sets"]), ", ".join(node["sets"])))
            return None
        for _, s in node["sets"].items():
            v = _clean(s["per_seed"], metric)
            return v or None
    if "per_seed" in node:
        v = _clean(node["per_seed"], metric)
        return v or None
    return None


def first_seeds(paths, dataset: str, metric: str):
    """First of several candidate paths that yields seed-wise scores.

    train.py writes one file per model holding every dataset, while the
    shipped results keep one directory per dataset. Both layouts are read here
    so that following the README end to end produces comparisons rather than
    an empty collection.
    """
    for p in paths:
        v = seeds_from(p, dataset, metric)
        if v:
            return v
    return None


def perm_seeds(paths, dataset: str, metric: str):
    """순열 통제의 관측치. 순열 시드 하나가 관측치 하나다.

    다른 모델은 그래프 하나 위에서 학습 시드 다섯을 돌린 것이라 관측치가 학습
    시드다. 순열은 그래프 자체가 순열 시드마다 다르므로, 학습 시드 다섯을 먼저
    평균 내 그 순열의 값을 만들고 순열 시드를 가로질러 다섯 개를 모은다. 그래야
    다른 모델과 짝을 지을 때 같은 것을 짝지은 것이 된다.

    순열 시드 다섯이 다 있는 데이터셋만 돌려준다. 넷만 있는 것을 다섯인 척
    쓰면 쌍체 비교가 조용히 어긋난다.
    """
    vals = []
    for p in paths:
        v = seeds_from(p, dataset, metric)
        if not v:
            return None
        vals.append(float(np.mean(v)))
    return vals if len(vals) == len(paths) else None


def collect(root: Path, metric: str = "macro_f1"):
    """Seed lists per model and per dataset, for one metric."""
    out = {m: {} for m in ("han", "gcn", "egraphsage", "mlp", "noedge",
                           "masked", "permuted")}
    for ds in DATASETS:
        candidates = {
            "han": [root / "main" / "han" / ds / "multiseed_summary.json",
                    root / "main" / "han" / "multiseed_summary.json"],
            "gcn": [root / "main" / "gcn" / ds / "homogeneous_gcn_summary.json",
                    root / "main" / "gcn" / "homogeneous_gcn_summary.json"],
            "noedge": [root / "no_edge" / ds / "multiseed_summary.json",
                       root / "no_edge" / "multiseed_summary.json"],
            "masked": [root / "masked" / "han" / ds / "multiseed_summary.json",
                       root / "masked" / "han" / "multiseed_summary.json"],
            "egraphsage": [root / "main" / "egraphsage" / "egraphsage_summary.json",
                           root / "main" / "egraphsage" / ds / "egraphsage_summary.json"],
            "mlp": [root / "main" / "mlp_summary.json",
                    root / "main" / "mlp" / "mlp_baseline_summary.json"],
            # 값 순열은 다섯 시드가 아니라 다섯 **순열 시드**로 돌았고, 시드마다
            # 그래프가 다르다. 한 순열의 다섯 학습 시드를 평균 내면 그래프
            # 하나에 대한 값이 되므로, 순열 시드를 가로질러 그 평균들을 모은다.
            # 그러면 관측치 다섯 개가 되어 다른 모델과 같은 방식으로 다룰 수 있다.
            "permuted": [root / "permutation" / ds / ("seed_%d" % s)
                         / "multiseed_summary.json" for s in PERM_SEEDS],
        }
        for m, paths in candidates.items():
            v = (perm_seeds(paths, ds, metric) if m == "permuted"
                 else first_seeds(paths, ds, metric))
            if v:
                out[m][ds] = v
    missing = [(m, ds) for m in out for ds in DATASETS if ds not in out[m]]
    if missing:
        print("  [note] no seed scores found for: "
              + ", ".join("%s/%s" % (m, ds) for m, ds in missing))
    return out


def ci_paired(a, b, n=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ci_unpaired(a, b, n=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a), np.asarray(b)
    ia = rng.integers(0, len(a), size=(n, len(a)))
    ib = rng.integers(0, len(b), size=(n, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/model_comparison")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    def run(pairs, metric, rng, tag=""):
        data = collect(root, metric)
        print("\n=== %s%s ===" % (metric, tag))
        print("%-13s %-18s %8s %20s %20s %6s" %
              ("dataset", "pair", "diff", "paired 95% CI", "unpaired 95% CI", "sign"))
        for ds in DATASETS:
            for x, y in pairs:
                a_, b_ = data[x].get(ds), data[y].get(ds)
                if not a_ or not b_ or len(a_) != len(b_):
                    continue
                m, lo, hi = ci_paired(a_, b_, args.resamples, rng)
                um, ulo, uhi = ci_unpaired(a_, b_, args.resamples, rng)
                d = np.asarray(a_) - np.asarray(b_)
                sign = float(max((d > 0).mean(), (d < 0).mean()))
                key = "%s__%s_minus_%s" % (ds, x, y)
                if metric != "macro_f1":
                    key += "__" + metric
                report[key] = {"metric": metric, "n_seeds": len(a_),
                               "mean_diff": round(m, 4),
                               "paired_ci": [round(lo, 4), round(hi, 4)],
                               "unpaired_ci": [round(ulo, 4), round(uhi, 4)],
                               "sign_consistency": round(sign, 2),
                               "paired_ci_includes_zero": bool(lo <= 0 <= hi),
                               "unpaired_ci_includes_zero": bool(ulo <= 0 <= uhi),
                               "per_seed_%s" % x: [round(v, 4) for v in a_],
                               "per_seed_%s" % y: [round(v, 4) for v in b_]}
                print("%-13s %-18s %8.4f  [%7.4f,%7.4f]  [%7.4f,%7.4f] %6.1f" %
                      (ds, "%s-%s" % (x, y), m, lo, hi, ulo, uhi, sign))

    # Each block draws from its own generator, so a block can be run or skipped
    # without shifting the intervals of the others.
    run(PAIRS, "macro_f1", np.random.default_rng(args.seed))
    run(CONTROL_PAIRS, "macro_f1", np.random.default_rng(args.seed + 1), " controls")
    for i, metric in enumerate(m for m in METRICS if m != "macro_f1"):
        run(PAIRS + CONTROL_PAIRS, metric, np.random.default_rng(args.seed + 2 + i))

    # HAN 이 MLP 를 넘는 마진을 세 항으로 가른다. 논문이 인용하는 것이 이것이다.
    #
    #   아키텍처    = 자기루프 − MLP     구조와 파라미터 수가 주는 몫
    #   가용성·연결 = 순열     − 자기루프  값이 무엇이든 상관없는 연결의 몫
    #   값 의미     = HAN      − 순열     값의 정체가 주는 몫
    #
    # 세 항의 합이 마진이다. 점추정만 내면 분모가 작을 때 몇 퍼센트라는 말이
    # 얼마나 단단한지 알 수 없다 — HIKARI 의 마진은 0.0081 이다. 그래서 각 항과
    # 각 비율에 부트스트랩 구간을 붙인다. 재표집 대상은 다른 비교와 같은
    # 관측치, 곧 시드다.
    shares = {}
    rng_share = np.random.default_rng(args.seed + 100)
    for metric in METRICS:
        data = collect(root, metric)
        for ds in DATASETS:
            h = data["han"].get(ds)
            sl = data["noedge"].get(ds)
            ml = data["mlp"].get(ds)
            pm = data["permuted"].get(ds)
            if not (h and sl and ml):
                continue
            mh, msl, mml = float(np.mean(h)), float(np.mean(sl)), float(np.mean(ml))
            total = mh - mml
            if abs(total) < 1e-9:
                continue
            block = {"han": round(mh, 4), "noedge": round(msl, 4),
                     "mlp": round(mml, 4), "total_margin": round(total, 4),
                     "architecture": round(msl - mml, 4),
                     "edges": round(mh - msl, 4),
                     "edge_share_pct": round(100 * (mh - msl) / total, 1),
                     "n_seeds": {"han": len(h), "noedge": len(sl),
                                 "mlp": len(ml)}}

            # 세 항으로 가르려면 순열이 있어야 한다. 없으면 두 항까지만 낸다.
            if pm and len(pm) == len(h) == len(sl):
                mpm = float(np.mean(pm))
                block.update({
                    "permuted": round(mpm, 4),
                    "availability_and_connection": round(mpm - msl, 4),
                    "value_identity": round(mh - mpm, 4),
                    "availability_share_pct": round(100 * (mpm - msl) / total, 1),
                    "value_share_pct": round(100 * (mh - mpm) / total, 1),
                    "three_way_sums_to_margin": bool(
                        abs((msl - mml) + (mpm - msl) + (mh - mpm) - total) < 1e-9)})
                block["n_seeds"]["permuted"] = len(pm)

                # 각 항과 각 비율의 구간.
                #
                # 쌍을 지을 수 있는 것과 없는 것을 구별해야 한다. MLP·자기루프·
                # HAN 의 관측치는 **학습 시드**이고 같은 시드끼리 짝이 된다.
                # 순열의 관측치는 **순열 시드**다 — 순열 시드 하나가 그래프
                # 하나이고, 그 안에서 학습 시드 다섯을 이미 평균 냈다. 순열
                # 시드 3 과 학습 시드 3 은 아무 관계도 없다.
                #
                # 그래서 학습 시드 셋은 인덱스 하나로 함께 뽑고(쌍체), 순열은
                # 독립적으로 뽑는다. 이 구별을 하지 않고 인덱스 하나를 넷에 다
                # 쓰면 짝이 아닌 것을 짝으로 세어 구간이 실제보다 좁아진다.
                a, b, c, d = (np.asarray(ml, float), np.asarray(sl, float),
                              np.asarray(pm, float), np.asarray(h, float))
                n, R = len(a), args.resamples
                idx = rng_share.integers(0, n, size=(R, n))
                idx_p = rng_share.integers(0, len(c), size=(R, len(c)))
                mA, mB = a[idx].mean(axis=1), b[idx].mean(axis=1)
                mC, mD = c[idx_p].mean(axis=1), d[idx].mean(axis=1)
                tot = mD - mA
                terms = {"architecture": mB - mA,
                         "availability_and_connection": mC - mB,
                         "value_identity": mD - mC}
                ci = {}
                for k, v in terms.items():
                    ci[k] = [round(float(np.percentile(v, 2.5)), 4),
                             round(float(np.percentile(v, 97.5)), 4)]
                    ok = np.abs(tot) > 1e-9
                    if ok.sum() >= 100:
                        pct = 100.0 * v[ok] / tot[ok]
                        ci[k + "_pct"] = [round(float(np.percentile(pct, 2.5)), 1),
                                          round(float(np.percentile(pct, 97.5)), 1)]
                block["ci"] = ci
                block["note"] = (
                    "terms over training seeds are paired; the permutation arm is "
                    "resampled independently because its observations are "
                    "permutation seeds, not training seeds. the percentage "
                    "interval is wide wherever the margin is "
                    "small; the denominator is resampled with the numerator")
            else:
                block["three_way"] = ("not available: the permutation control "
                                      "has no matching seed list for this "
                                      "dataset")
            shares["%s__%s" % (ds, metric)] = block
    report["_edge_share"] = shares

    json.dump(report, open(out / "bootstrap_ci.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "bootstrap_ci.json"))


if __name__ == "__main__":
    main()
