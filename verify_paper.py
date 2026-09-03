# -*- coding: utf-8 -*-
"""verify_paper.py -- recompute every number the paper reports from results/.

Each claim names a table or section of the manuscript, the value printed there,
and a function that recomputes that value from the shipped result files.
Run from the repository root:

    python verify_paper.py               # all claims
    python verify_paper.py --only "Table 6"

A claim passes when |stated - recomputed| <= tol (0.0006 for scores; wider for
percentages and counts, as set per claim). Exit status is non-zero if any claim
fails. Names follow the paper: VersionCipher is stored under the legacy column
name ``ja3`` (see README, "Internal naming note").
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "results"

DS = ["bccc_dohbrw", "iscx_vpn", "hikari", "cic_andmal", "vnat"]
NAME = dict(zip(DS, ["BCCC-DoH", "ISCX-VPN", "HIKARI", "CIC-AndMal", "VNAT"]))
# paper name -> artifact column
RENAMES = {"VersionCipher": "via_ja3", "CipherGroup": "via_tls_cipher_group"}


def seeds(path, metric="macro_f1", dataset=None):
    """Per-seed scores from a multiseed-format summary file."""
    p = ART / path
    if not p.exists():
        return None
    d = json.load(open(p))
    for ds, v in d.items():
        if dataset and ds != dataset:
            continue
        if "sets" in v:
            for _, s in v["sets"].items():
                return np.array([x[metric] for x in s["per_seed"]], float)
        if "per_seed" in v:
            return np.array([x[metric] for x in v["per_seed"]], float)
    return None


def mean(path, metric="macro_f1", dataset=None):
    v = seeds(path, metric, dataset)
    return None if v is None else float(v.mean())


def tabular(ds, run):
    d = json.load(open(ART / "main" / "tabular_summary.json"))
    return d[ds]["runs"][run]["macro_f1_mean"]


def mlp(ds, metric="macro_f1"):
    d = json.load(open(ART / "main" / "mlp_summary.json"))
    return float(np.mean([x[metric] for x in d[ds]["per_seed"]]))


def gcn(ds):
    return mean("main/gcn/%s/homogeneous_gcn_summary.json" % ds)


def egs(ds):
    d = json.load(open(ART / "main" / "egraphsage" / "egraphsage_summary.json"))
    return float(np.mean([x["macro_f1"] for x in d[ds]["per_seed"]]))


def han(ds, metric="macro_f1"):
    return mean("main/han/%s/multiseed_summary.json" % ds, metric)


def noedge(ds, metric="macro_f1"):
    return mean("no_edge/%s/multiseed_summary.json" % ds, metric)


def masked(ds, metric="macro_f1"):
    return mean("masked/han/%s/multiseed_summary.json" % ds, metric)


def boot(key, field):
    d = json.load(open(ART / "model_comparison" / "bootstrap_ci.json"))
    return d[key][field]


def cross_ratio(ds, rel):
    d = json.load(open(ART / "masked" / "edge_statistics.json"))
    return d[ds]["relations"][rel]["cross_ratio"]


def train_test_share(ds, rel):
    d = json.load(open(ART / "masked" / "edge_statistics.json"))
    bp = d[ds]["relations"][rel]["by_partition_pair"]
    tot = sum(bp.values())
    return (bp.get("tr-te", 0) + bp.get("te-tr", 0)) / tot


def ranks(ds):
    """Per-relation score vectors from a repeated-splits file, grouped by variant.

    Three variant kinds: random split, order-preserving cutoff, edge-sampling seed.
    Sec 6.5 includes edge-seed variants in the ranking aggregate but not in the two
    protocol columns; the tau values change if that distinction is dropped.
    """
    d = json.load(open(ART / "repeated_splits" / ("%s.json" % ds)))[ds]
    rels, R, C, E = None, [], [], []
    for k, v in d.items():
        if "rank" not in v:
            continue
        if rels is None:
            rels = sorted(v["rank"])
        vec = [v["rank"][r]["mean"] for r in rels]
        (R if k.startswith("random") else C if k.startswith("cutoff") else E).append(vec)
    return rels, R, C, E


def kendall(a, b):
    c = t = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        c += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        t += 1
    return c / t


def homophily(ds, rel, key="homophily", src="homophily/edge_homophily.json"):
    p = ART / src
    if not p.exists():
        return None
    d = json.load(open(p))
    return d[ds]["relations"][rel][key]


def build_claims():
    C = []

    def add(name, stated, fn, tol=0.0006, src=""):
        C.append((name, stated, fn, tol, src))

    # --- Table 4 (main table) ---
    T5 = {"bccc_dohbrw": (0.4149, 0.9964, 1.0000, 0.9385, 0.9436, 0.9998, 0.9897),
          "iscx_vpn":   (0.4914, 1.0000, 0.9531, 0.5329, 0.7359, 0.9552, 0.8725),
          "hikari":     (0.4824, 0.4570, 0.4652, 0.4427, 0.3787, 0.4508, 0.4496),
          "cic_andmal": (0.3550, 0.6583, 0.7028, 0.5581, 0.5582, 0.5762, 0.5422),
          "vnat":       (0.4883, 1.0000, 1.0000, 0.9525, 0.9001, 0.9949, 1.0000)}
    for ds, (maj, hgb, hgbm, ml, gc, ha, eg) in T5.items():
        n = NAME[ds]
        add("Table 4 %s HGB" % n, hgb, lambda d=ds: tabular(d, "flow_only__fit_trainval"),
            src="main/tabular_summary.json")
        add("Table 4 %s HGB+meta" % n, hgbm,
            lambda d=ds: tabular(d, "flow_plus_meta_full__fit_trainval"),
            src="main/tabular_summary.json")
        add("Table 4 %s MLP" % n, ml, lambda d=ds: mlp(d), src="main/mlp_summary.json")
        add("Table 4 %s GCN" % n, gc, lambda d=ds: gcn(d), src="main/gcn/")
        add("Table 4 %s HAN" % n, ha, lambda d=ds: han(d), src="main/han/")
        add("Table 4 %s E-GraphSAGE" % n, eg, lambda d=ds: egs(d), src="main/egraphsage/")
        add("Table 4 %s majority-class baseline" % n, maj,
            lambda d=ds: json.load(open(ART / "reference_lines" / "summary.json"))[d]["majority_macro_f1"],
            src="reference_lines/summary.json")

    # --- controls (Table 5, Sec 6.2) ---
    for ds, sl, mk in (("bccc_dohbrw", .9463, .9677), ("iscx_vpn", .5942, .6532),
                       ("hikari", .4503, .4510), ("cic_andmal", .5747, .5760),
                       ("vnat", .9898, .9949)):
        add("Table 5 %s self-loop" % NAME[ds], sl, lambda d=ds: noedge(d), src="no_edge/")
        add("Sec 6.2 %s masking" % NAME[ds], mk, lambda d=ds: masked(d), src="masked/han/")

    # --- relation contribution ratio (Sec 6.1) ---
    for ds, share in (("bccc_dohbrw", 87.3), ("iscx_vpn", 85.5), ("hikari", 5.9),
                      ("cic_andmal", 8.1), ("vnat", 12.1)):
        add("Sec 6.1 %s relation contribution ratio, macro %%" % NAME[ds], share,
            lambda d=ds: 100 * (han(d) - noedge(d)) / (han(d) - mlp(d)), tol=0.15,
            src="_edge_share")
    for ds, share in (("bccc_dohbrw", 87.7), ("iscx_vpn", 90.1), ("hikari", 3.0),
                      ("cic_andmal", 30.2), ("vnat", 12.2)):
        add("Sec 6.1 %s relation contribution ratio, minority %%" % NAME[ds], share,
            lambda d=ds: 100 * (han(d, "minority_f1") - noedge(d, "minority_f1"))
            / (han(d, "minority_f1") - mlp(d, "minority_f1")), tol=0.15,
            src="_edge_share")

    # --- variants that are easy to confuse ---
    for ds, v in (("bccc_dohbrw", 1.0000), ("iscx_vpn", 0.9601),
                  ("hikari", 0.4638), ("cic_andmal", 0.7028), ("vnat", 1.0000)):
        add("Sec 6.1 %s HGB+meta (selected set)" % NAME[ds], v,
            lambda d=ds: tabular(d, "flow_plus_meta_selected__fit_trainval"),
            src="main/tabular_summary.json (§5.1 'the set the graph receives')")

    # --- intervals quoted in the paper ---
    add("Sec 6.1 CIC HAN-self-loop unpaired CI lower", -0.008,
        lambda: boot("cic_andmal__han_minus_noedge", "unpaired_ci")[0], tol=0.0006,
        src="bootstrap_ci.json unpaired_ci (Sec 5.5 specifies unpaired)")
    add("Sec 6.1 CIC HAN-self-loop unpaired CI upper", 0.010,
        lambda: boot("cic_andmal__han_minus_noedge", "unpaired_ci")[1], tol=0.0006,
        src="bootstrap_ci.json unpaired_ci")

    # --- identifier-disjoint split, relation families, matched random, budget, E-GraphSAGE ---
    def dd(f):
        d = json.load(open(ART / "dst_disjoint" / f))["iscx_vpn"]
        b = list(d["sets"].values())[0] if "sets" in d else d
        return float(np.mean([x["macro_f1"] for x in b["per_seed"]]))
    for lab, f, v in (("HGB", "hgb.json", 0.5119), ("MLP", "mlp.json", 0.7955),
                      ("self-loop", "noedge.json", 0.8306), ("HAN", "han.json", 0.9729)):
        add("Table 10 ISCX destination-disjoint %s" % lab, v, lambda g=f: dd(g),
            src="dst_disjoint/%s" % f)

    def fam(kind, ds):
        d = json.load(open(ART / "relation_family" / ("%s_%s.json" % (kind, ds))))[ds]
        b = list(d["sets"].values())[0]
        return float(np.mean([x["macro_f1"] for x in b["per_seed"]]))
    for ds, t, h in (("iscx_vpn", 0.7205, 0.8081), ("vnat", 0.8448, 0.9924),
                     ("bccc_dohbrw", 0.9997, 0.9994)):
        add("Table 13 %s 7 TLS relations" % NAME[ds], t, lambda d=ds: fam("tls", d),
            src="relation_family/tls_%s.json" % ds)
        add("Table 13 %s 2 host relations" % NAME[ds], h, lambda d=ds: fam("host", d),
            src="relation_family/host_%s.json" % ds)

    add("Table 7 BCCC density-matched random", 0.9804,
        lambda: mean("random_match/bccc_dohbrw.json"),
        src="random_match/bccc_dohbrw.json")

    for ds, v in (("hikari", 0.4519), ("vnat", 0.9974)):
        add("Sec 6.6 %s neighbor budget 4" % NAME[ds], v,
            lambda d=ds: mean("sensitivity_weak/%s.json" % d, dataset=d),
            src="sensitivity_weak/")

    def egs_ns(ds):
        d = json.load(open(ART / "egs_no_self" / "egraphsage_summary.json"))[ds]
        return float(np.mean([x["macro_f1"] for x in d["per_seed"]]))
    for ds, v in (("bccc_dohbrw", 0.9908), ("iscx_vpn", 0.9374), ("hikari", 0.4515),
                  ("cic_andmal", 0.5368), ("vnat", 1.0000)):
        add("Sec 6.1 %s E-GraphSAGE self-edge excluded" % NAME[ds], v, lambda d=ds: egs_ns(d),
            src="egs_no_self/")

    add("Sec 6.5 HIKARI full-pool HAN", 0.4496,
        lambda: mean("typing_vs_pruning/han_full/hikari.json", dataset="hikari"),
        src="typing_vs_pruning/han_full/hikari.json")

    # --- repeated splits (Table 8) ---
    # Random-split variants: the shipped summaries for ISCX-VPN, VNAT, HIKARI and
    # CIC-AndMal are to be regenerated with the decoupled edge seed (see README,
    # "Random variants and edge seeds"). BCCC-DoH is not rerun: its reported
    # relation has six values, so every edge seed yields the same graph and the
    # edge-seed arm shows zero variation (Table 9).
    def repeat_arm(ds, prefix, what):
        d = json.load(open(ART / "repeated_splits" / ("%s.json" % ds)))[ds]
        v = [float(np.mean(d[k]["han"]["per_seed"]))
             for k in d if k.startswith(prefix)]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "n": len(v)}[what]

    T63 = {"iscx_vpn":   (0.9461, 0.0460, 10, 0.9290, 0.0520, 5,  0.0171),
           "vnat":       (0.9843, 0.0172, 10, 0.9869, 0.0183, 5, -0.0026),
           "hikari":     (0.7066, 0.0011, 10, 0.4549, 0.0034, 5,  0.2517)}
    for ds, (rm, rs, rn, cm, cs, cn, drop) in T63.items():
        n = NAME[ds]
        add("Table 8 %s random-split mean" % n, rm,
            lambda d=ds: repeat_arm(d, "random_", "mean"),
            src="repeated_splits/%s.json  per_seed" % ds)
        add("Table 8 %s random-split std" % n, rs,
            lambda d=ds: repeat_arm(d, "random_", "std"),
            src="repeated_splits/%s.json  per_seed" % ds)
        add("Table 8 %s random variants" % n, rn,
            lambda d=ds: repeat_arm(d, "random_", "n"), tol=0.5,
            src="repeated_splits/%s.json" % ds)
        add("Table 8 %s cutoff mean" % n, cm,
            lambda d=ds: repeat_arm(d, "cutoff_", "mean"),
            src="repeated_splits/%s.json  per_seed" % ds)
        add("Table 8 %s cutoff std" % n, cs,
            lambda d=ds: repeat_arm(d, "cutoff_", "std"),
            src="repeated_splits/%s.json  per_seed" % ds)
        add("Table 8 %s cutoff variants" % n, cn,
            lambda d=ds: repeat_arm(d, "cutoff_", "n"), tol=0.5,
            src="repeated_splits/%s.json" % ds)
        add("Table 8 %s drop" % n, drop,
            lambda d=ds: repeat_arm(d, "random_", "mean")
            - repeat_arm(d, "cutoff_", "mean"),
            src="repeated_splits/%s.json  difference of the two means" % ds)

    add("Sec 6.3 HIKARI drop / random-split std (236x)", 236,
        lambda: (repeat_arm("hikari", "random_", "mean")
                 - repeat_arm("hikari", "cutoff_", "mean"))
        / repeat_arm("hikari", "random_", "std"), tol=1.0,
        src="repeated_splits/hikari.json")

    # --- relation ranking stability (Sec 6.5) ---
    def tau_consensus(ds):
        _, R, C, _ = ranks(ds)
        return kendall(np.mean(R, axis=0), np.mean(C, axis=0))

    add("Sec 6.5 ISCX consensus tau random vs cutoff", 0.47, lambda: tau_consensus("iscx_vpn"), tol=0.006,
        src="repeated_splits, random 10 / cutoff 5, edge-seed variants excluded")
    add("Sec 6.5 VNAT consensus tau random vs cutoff", 0.24, lambda: tau_consensus("vnat"), tol=0.006,
        src="repeated_splits, random 10 / cutoff 5, edge-seed variants excluded")

    def tau_pair(ds, kind):
        _, R, C, _ = ranks(ds)
        if kind == "rr":
            return np.mean([kendall(a, b) for a, b in itertools.combinations(R, 2)])
        return np.mean([kendall(a, b) for a, b in itertools.product(R, C)])

    add("Sec 6.5 ISCX pairwise tau among random splits", 0.54, lambda: tau_pair("iscx_vpn", "rr"), tol=0.006,
        src="repeated_splits")
    add("Sec 6.5 ISCX pairwise tau random vs cutoff", 0.37, lambda: tau_pair("iscx_vpn", "rc"),
        tol=0.006, src="repeated_splits")

    def top1_count(ds, rel):
        rels, R, C, E = ranks(ds)
        first = [rels[int(np.argmax(v))] for v in R + C + E]
        return first.count(rel)

    add("Sec 6.5 SrcHost first-place count (20 variants)", 17,
        lambda: top1_count("iscx_vpn", "via_src_host"), tol=0.5,
        src="repeated_splits, all 20 variants")

    def n_first(ds):
        rels, R, C, E = ranks(ds)
        return len({rels[int(np.argmax(v))] for v in R + C + E})

    add("Sec 6.5 VNAT relations ever ranked first", 8, lambda: n_first("vnat"), tol=0.5,
        src="repeated_splits")

    # --- boundary crossing (Sec 4.3, Sec 6.2) ---
    add("Sec 4.3 BCCC CertSubject boundary-crossing %", 52,
        lambda: 100 * cross_ratio("bccc_dohbrw", "via_cert_subject"), tol=0.6,
        src="masked/edge_statistics.json")
    add("Sec 4.3 ISCX SrcHost boundary-crossing %", 10.2,
        lambda: 100 * cross_ratio("iscx_vpn", "via_src_host"), tol=0.15,
        src="masked/edge_statistics.json")
    add("Sec 4.3 ISCX VersionCipher boundary-crossing %", 30.8,
        lambda: 100 * cross_ratio("iscx_vpn", "via_ja3"), tol=0.15,
        src="masked/edge_statistics.json")
    add("Sec 4.3 VNAT TimeBin train-test crossing", 0.0,
        lambda: train_test_share("vnat", "via_timebin"), tol=1e-9,
        src="masked/edge_statistics.json by_partition_pair")
    add("Sec 4.3 ISCX TimeBin train-test crossing", 0.0,
        lambda: train_test_share("iscx_vpn", "via_timebin"), tol=1e-9,
        src="masked/edge_statistics.json by_partition_pair")

    # --- edge homophily (Table 12, Sec 5.3) ---
    for ds, rel, v in (("bccc_dohbrw", "via_cert_validity", 0.9999),
                       ("bccc_dohbrw", "via_ja3", 0.8109),
                       ("bccc_dohbrw", "via_cert_subject", 0.9999),
                       ("cic_andmal", "via_sni", 0.5419)):
        add("Table 12 %s %s" % (NAME[ds], rel.replace("via_", "")), v,
            lambda d=ds, r=rel: homophily(d, r), src="homophily/edge_homophily.json")

    # --- literature approximation (Sec 6.6) ---
    add("Sec 6.6 literature approximation HAN", 0.9892, lambda: mean("reta_approx/han.json"),
        src="reta_approx/han.json")
    add("Sec 6.6 approximation self-loop", 0.9476, lambda: mean("reta_approx/no_edge.json"),
        src="reta_approx/no_edge.json")
    add("Sec 6.6 approximation masking", 0.9734, lambda: mean("reta_approx/masked.json"),
        src="reta_approx/masked.json")
    def reta_perm():
        return float(np.mean([mean("reta_approx/permutation/seed_%d.json" % p)
                              for p in (41, 42, 43, 44, 45)]))
    def reta_margin():
        return mean("reta_approx/han.json") - mlp("bccc_dohbrw")
    add("Sec 6.6 approximation permutation", 0.9793, reta_perm, src="reta_approx/permutation/")
    add("Sec 6.6 approximation edge share %", 81.9,
        lambda: 100 * (mean("reta_approx/han.json") - mean("reta_approx/no_edge.json"))
        / reta_margin(), tol=0.15,
        src="reta_approx/ + main/mlp_summary.json")
    add("Sec 6.6 approximation architecture %", 18.1,
        lambda: 100 * (mean("reta_approx/no_edge.json") - mlp("bccc_dohbrw")) / reta_margin(),
        tol=0.15, src="reta_approx/no_edge.json + main/mlp_summary.json")
    add("Sec 6.6 approximation availability/connectivity %", 62.3,
        lambda: 100 * (reta_perm() - mean("reta_approx/no_edge.json")) / reta_margin(),
        tol=0.15, src="reta_approx/permutation/ + no_edge.json")
    add("Sec 6.6 approximation value semantics %", 19.6,
        lambda: 100 * (mean("reta_approx/han.json") - reta_perm()) / reta_margin(),
        tol=0.15, src="reta_approx/han.json + permutation/")
    add("Sec 6.6 approximation masking drop % of margin", 31.1,
        lambda: 100 * (mean("reta_approx/han.json") - mean("reta_approx/masked.json"))
        / reta_margin(), tol=0.15, src="reta_approx/han.json + masked.json")
    add("Sec 6.6 approximation total margin", 0.0508,
        lambda: mean("reta_approx/han.json") - mlp("bccc_dohbrw"),
        src="reta_approx/han.json - main/mlp_summary.json")

    # --- relation contribution ratio, canonical ---
    SHARE = {("bccc_dohbrw", "macro_f1"): 87.3, ("iscx_vpn", "macro_f1"): 85.5,
             ("hikari", "macro_f1"): 5.9, ("cic_andmal", "macro_f1"): 8.1,
             ("vnat", "macro_f1"): 12.1,
             ("bccc_dohbrw", "minority_f1"): 87.7, ("iscx_vpn", "minority_f1"): 90.1,
             ("hikari", "minority_f1"): 3.0, ("cic_andmal", "minority_f1"): 30.2,
             ("vnat", "minority_f1"): 12.2}
    for (d_, m_), pct in SHARE.items():
        add("Relation contribution ratio %s %s" % (NAME[d_], m_), pct,
            (lambda d_=d_, m_=m_: 100 * (han(d_, m_) - noedge(d_, m_))
             / (han(d_, m_) - mlp(d_, m_))), tol=0.06,
            src="main/han + no_edge + main/mlp_summary.json")
        add("Relation contribution ratio %s %s (file value)" % (NAME[d_], m_), pct,
            (lambda d_=d_, m_=m_: json.load(open(ART / "model_comparison" /
             "bootstrap_ci.json"))["_edge_share"]["%s__%s" % (d_, m_)]["edge_share_pct"]),
            tol=0.001, src="bootstrap_ci.json _edge_share")

    # --- homophily diagnostics and correlations (Sec 6.5) ---
    def _pairs31():
        rows = []

        def push(ds, rel, perf, hsrc):
            h = json.load(open(ART / hsrc))[ds]
            r = h["relations"][rel]
            n = h["n_flows"]
            rows.append((r["lift_over_chance"], 1 - r["isolated_nodes"] / n,
                         perf - noedge(ds)))
        b = json.load(open(ART / "single_relation" /
                           "bccc_dohbrw_with_host.json"))["bccc_dohbrw"]
        for s in b["sets"].values():
            push("bccc_dohbrw", s["metapaths"][0],
                 float(np.mean([p["macro_f1"] for p in s["per_seed"]])),
                 "homophily/edge_homophily_bccc10.json")
        c = json.load(open(ART / "single_relation" / "cic_andmal.json"))["cic_andmal"]
        for s in c["sets"].values():
            push("cic_andmal", s["metapaths"][0],
                 float(np.mean([p["macro_f1"] for p in s["per_seed"]])),
                 "homophily/edge_homophily.json")
        for ds in ("iscx_vpn", "vnat"):
            h = json.load(open(ART / "homophily" / "edge_homophily.json"))[ds]
            for rel in h["relations"]:
                p = ART / "split_protocol" / "relation_ranking" / "ts" / rel / ("%s.json" % ds)
                if not p.exists():
                    continue
                s = list(json.load(open(p))[ds]["sets"].values())[0]
                push(ds, rel, float(np.mean([q["macro_f1"] for q in s["per_seed"]])),
                     "homophily/edge_homophily.json")
        return rows

    def _spearman(x, y):
        def rk(a):
            a = np.asarray(a, float)
            r = np.empty(len(a), float)
            r[a.argsort()] = np.arange(len(a))
            for v in set(a.tolist()):
                mk = a == v
                if mk.sum() > 1:
                    r[mk] = r[mk].mean()
            return r
        return float(np.corrcoef(rk(x), rk(y))[0, 1])

    add("Sec 6.5 number of (dataset, relation) pairs", 31, lambda: len(_pairs31()), tol=0, src="31 (dataset, relation) pairs")
    add("Sec 6.5 rho(homophily lift, gain)", 0.248,
        lambda: _spearman([r[0] for r in _pairs31()], [r[2] for r in _pairs31()]),
        tol=0.0006, src="homophily + single_relation + no_edge")
    add("Sec 6.5 rho(reach, gain)", 0.206,
        lambda: _spearman([r[1] for r in _pairs31()], [r[2] for r in _pairs31()]),
        tol=0.0006, src="same as above")
    add("Sec 6.5 rho(reach x lift, gain)", 0.395,
        lambda: _spearman([r[0] * r[1] for r in _pairs31()],
                          [r[2] for r in _pairs31()]), tol=0.0006, src="same as above")

    def _norm_homophily(ds):
        src = ("homophily/edge_homophily_bccc10.json" if ds == "bccc_dohbrw"
               else "homophily/edge_homophily.json")
        h = json.load(open(ART / src))[ds]
        ch = h["chance_homophily"]
        mx = max(r["homophily"] for r in h["relations"].values())
        return (mx - ch) / (1 - ch)

    for ds, nh in (("bccc_dohbrw", 1.000), ("iscx_vpn", 1.000), ("vnat", 1.000),
                   ("hikari", 0.258), ("cic_andmal", 0.074)):
        add("Table 12 %s normalized homophily" % NAME[ds], nh,
            (lambda ds=ds: _norm_homophily(ds)), tol=0.0006, src="homophily/")
    # --- Table 9: relation contribution ratio per split protocol ---
    def _decomp(path, ds, family):
        p = ART / "repeat_decomp" / path
        if not p.exists():
            return None
        d = json.load(open(p))[ds]
        head = {"random": "random", "cutoff": "cutoff", "edgeseed": "edgeseed"}[family]
        vals = []
        for k, v in d.items():
            if not k.startswith(head):
                continue
            try:
                ha, ml, ne = v["han"]["mean"], v["mlp"]["mean"], v["noedge"]["mean"]
            except KeyError:
                continue
            if ha - ml <= 0:
                continue
            vals.append(100.0 * (ha - ne) / (ha - ml))
        return vals

    for dc_path, dc_ds, dc_fam, dc_n, dc_lo, dc_hi, dc_mean, dc_std in (
            ("iscx_vpn.json", "iscx_vpn", "cutoff", 5, 80.3, 87.6, 84.7, 2.7),
            ("iscx_vpn.json", "iscx_vpn", "edgeseed", 5, 82.0, 89.0, 85.8, 2.5),
            ("iscx_vpn.json", "iscx_vpn", "random", 10, 24.8, 74.3, 55.4, 15.8),
            ("cic_andmal.json", "cic_andmal", "cutoff", 2, 9.6, 17.6, 13.6, 4.0),
            ("cic_andmal.json", "cic_andmal", "edgeseed", 3, 4.0, 14.7, 10.1, 4.5),
            ("cic_andmal.json", "cic_andmal", "random", 5, 8.1, 26.0, 14.9, 6.2),
            ("bccc_dohbrw.json", "bccc_dohbrw", "cutoff", 3, 73.8, 84.8, 80.2, 4.7),
            ("bccc_dohbrw.json", "bccc_dohbrw", "edgeseed", 3, 84.5, 84.5, 84.5, 0.0),
            ("bccc_dohbrw.json", "bccc_dohbrw", "random", 5, 32.9, 34.4, 33.6, 0.7)):
        _t = "Table 9 %s %s" % (NAME[dc_ds], dc_fam)
        _s = "repeat_decomp/" + dc_path
        add(_t + " variants", dc_n,
            (lambda q=dc_path, d=dc_ds, f=dc_fam: len(_decomp(q, d, f))),
            tol=0, src=_s)
        add(_t + " min", dc_lo,
            (lambda q=dc_path, d=dc_ds, f=dc_fam: min(_decomp(q, d, f))),
            tol=0.06, src=_s)
        add(_t + " max", dc_hi,
            (lambda q=dc_path, d=dc_ds, f=dc_fam: max(_decomp(q, d, f))),
            tol=0.06, src=_s)
        add(_t + " mean", dc_mean,
            (lambda q=dc_path, d=dc_ds, f=dc_fam: float(np.mean(_decomp(q, d, f)))),
            tol=0.06, src=_s)
        add(_t + " std", dc_std,
            (lambda q=dc_path, d=dc_ds, f=dc_fam: float(np.std(_decomp(q, d, f)))),
            tol=0.06, src=_s)

    # --- Sec 7.4: HIKARI port columns ---
    def _idx(run):
        d = json.load(open(ART / "feature_audit" /
                           "hikari_index_column_random.json"))["runs"]
        return d[run]["macro_f1_mean"]

    add("Sec 7.4 HIKARI HGB after index removal", 0.7252,
        lambda: _idx("drop_index_col__fit_trainval"),
        src="feature_audit/hikari_index_column_random.json")
    add("Sec 7.4 HIKARI HGB after index+port removal", 0.7103,
        lambda: _idx("drop_index_and_ports__fit_trainval"),
        src="feature_audit/hikari_index_column_random.json")
    add("Sec 7.4 HIKARI port-column contribution", 0.0149,
        lambda: _idx("drop_index_col__fit_trainval")
                - _idx("drop_index_and_ports__fit_trainval"),
        src="feature_audit/hikari_index_column_random.json")

    # --- Table 12: max-homophily ties and their reach ---
    def _tie(ds):
        src = ("homophily/edge_homophily_bccc10.json" if ds == "bccc_dohbrw"
               else "homophily/edge_homophily.json")
        h = json.load(open(ART / src))[ds]
        ch = h["chance_homophily"]
        rs = {k: v for k, v in h["relations"].items() if v["n_edges"] > 0}
        norm = {k: (v["homophily"] - ch) / (1 - ch) for k, v in rs.items()}
        mx = max(round(v, 4) for v in norm.values())
        tie = [rs[k] for k, v in norm.items() if round(v, 4) == mx]
        return len(tie), min(r["reach"] for r in tie), max(r["reach"] for r in tie)

    for th_ds, th_n, th_lo, th_hi in (("bccc_dohbrw", 3, 0.33, 1.00),
                              ("iscx_vpn", 3, 0.48, 1.00),
                              ("vnat", 6, 0.31, 1.00),
                              ("hikari", 1, 0.99, 0.99),
                              ("cic_andmal", 1, 0.95, 0.95)):
        add("Table 12 %s max-homophily ties" % NAME[th_ds], th_n,
            (lambda d=th_ds: _tie(d)[0]), tol=0, src="homophily/")
        add("Table 12 %s reach min" % NAME[th_ds], th_lo,
            (lambda d=th_ds: _tie(d)[1]), tol=0.005, src="homophily/")
        add("Table 12 %s reach max" % NAME[th_ds], th_hi,
            (lambda d=th_ds: _tie(d)[2]), tol=0.005, src="homophily/")

    add("Sec 6.5 rho(normalized homophily, contribution) dataset level", 0.783,
        lambda: _spearman([_norm_homophily(d) for d in
                           ("bccc_dohbrw", "iscx_vpn", "vnat", "hikari", "cic_andmal")],
                          [100 * (han(d) - noedge(d)) / (han(d) - mlp(d)) for d in
                           ("bccc_dohbrw", "iscx_vpn", "vnat", "hikari", "cic_andmal")]),
        tol=0.0006, src="homophily/ + _edge_share recomputed")

    # --- Table 5/6: permutation and three-component decomposition ---
    def perm_mean(ds, metric="macro_f1"):
        return float(np.mean([
            mean("permutation/%s/seed_%d/multiseed_summary.json" % (ds, p), metric)
            for p in (41, 42, 43, 44, 45)]))

    PERM = {"bccc_dohbrw": 0.9997, "iscx_vpn": 0.6298, "vnat": 0.9766,
            "cic_andmal": 0.5744, "hikari": 0.4492}
    for d_, v in PERM.items():
        add("Table 5 %s permutation" % NAME[d_], v, (lambda d_=d_: perm_mean(d_)),
            src="permutation/%s/" % d_)

    THREE = {  # architecture, availability/connectivity, value semantics
        "bccc_dohbrw": (0.0078, 0.0534, 0.0001),
        "iscx_vpn":    (0.0613, 0.0356, 0.3254),
        "vnat":        (0.0373, -0.0132, 0.0183),
        "cic_andmal":  (0.0166, -0.0003, 0.0018),
        "hikari":      (0.0076, -0.0011, 0.0016),
    }
    for d_, (arch, avail, sem) in THREE.items():
        add("Table 6 %s architecture component" % NAME[d_], arch,
            (lambda d_=d_: noedge(d_) - mlp(d_)), src="no_edge + mlp")
        add("Table 6 %s availability/connectivity component" % NAME[d_], avail,
            (lambda d_=d_: perm_mean(d_) - noedge(d_)), src="permutation + no_edge")
        add("Table 6 %s value-semantics component" % NAME[d_], sem,
            (lambda d_=d_: han(d_) - perm_mean(d_)), src="main/han + permutation")
        add("Table 6 %s components sum to margin" % NAME[d_], 0.0,
            (lambda d_=d_: (noedge(d_) - mlp(d_)) + (perm_mean(d_) - noedge(d_))
             + (han(d_) - perm_mean(d_)) - (han(d_) - mlp(d_))),
            tol=1e-9, src="identity check")

    # --- Sec 6.1: value-semantics noise threshold (std of permutation-seed means + std of HAN seeds) ---
    def sem_threshold(ds):
        pm = [mean("permutation/%s/seed_%d/multiseed_summary.json" % (ds, p))
              for p in (41, 42, 43, 44, 45)]
        hs = seeds("main/han/%s/multiseed_summary.json" % ds)
        return float(np.std(pm) + np.std(hs))
    THR = {"iscx_vpn": 0.0653, "vnat": 0.0130, "cic_andmal": 0.0082,
           "hikari": 0.0018, "bccc_dohbrw": 0.0001}
    for d_, v in THR.items():
        add("Sec 6.1 %s value-semantics noise threshold" % NAME[d_], v, (lambda d_=d_: sem_threshold(d_)),
            src="permutation/%s/ + main/han/%s/" % (d_, d_))

    NFLOW = {"iscx_vpn": 891, "vnat": 2197, "cic_andmal": 400524, "hikari": 555278}
    for d_, col in (("iscx_vpn", "src_ip"), ("cic_andmal", "cert_subject"),
                    ("hikari", "dst_ip")):
        add("Permutation preserves presence %s %s" % (NAME[d_], col),
            0.0,
            (lambda d_=d_, col=col: json.load(open(
                ART / "permutation" / d_ / "seed_41" / "permutation_report.json"
            ))[d_]["permuted_fields"][col]
             - round(json.load(open(ART / "profiling" / "relation_profiling.json"))
                     ["datasets"][d_]["fields"][col]["coverage"] * NFLOW[d_])),
            tol=1, src="permutation_report + profiling")

    # --- synthetic-token control (Sec 6.2, Table 7) ---
    NOUNK = "single_relation/bccc_dohbrw_nounk.json"
    add("Sec 6.2 BCCC token-reverted macro-F1", 0.9805, lambda: mean(NOUNK, dataset="bccc_dohbrw"),
        src=NOUNK)
    add("Sec 6.2 BCCC token-reverted minority-F1", 0.9725,
        lambda: mean(NOUNK, "minority_f1", dataset="bccc_dohbrw"), src=NOUNK)
    add("Table 7 token-reverted edges", 292034,
        lambda: homophily("bccc_dohbrw", "via_cert_validity", "n_edges",
                          "homophily/edge_homophily_bccc10_nounk.json"),
        tol=0, src="homophily/edge_homophily_bccc10_nounk.json")
    add("Table 7 token-reverted isolated nodes", 359022,
        lambda: homophily("bccc_dohbrw", "via_cert_validity", "isolated_nodes",
                          "homophily/edge_homophily_bccc10_nounk.json"),
        tol=0, src="same as above")
    add("Table 7 token-retained edges", 716222,
        lambda: homophily("bccc_dohbrw", "via_cert_validity", "n_edges",
                          "homophily/edge_homophily_bccc10.json"),
        tol=0, src="homophily/edge_homophily_bccc10.json")
    add("Sec 6.2 token share of relation contribution, macro %", 36.1,
        lambda: 100 * (han("bccc_dohbrw") - mean(NOUNK, dataset="bccc_dohbrw"))
        / (han("bccc_dohbrw") - noedge("bccc_dohbrw")), tol=0.06,
        src="main/han + no_edge + " + NOUNK)
    add("Sec 6.2 token share of relation contribution, minority %", 36.8,
        lambda: 100 * (han("bccc_dohbrw", "minority_f1")
                       - mean(NOUNK, "minority_f1", dataset="bccc_dohbrw"))
        / (han("bccc_dohbrw", "minority_f1") - noedge("bccc_dohbrw", "minority_f1")),
        tol=0.06, src="same as above")

    # --- values corrected during drafting, pinned here ---
    add("Table 11 VNAT test flows", 439,
        lambda: json.load(open(ART / "reference_lines" / "summary.json"))["vnat"]["n_test"],
        tol=0, src="reference_lines/summary.json")
    add("Sec 3.1 VNAT VersionCipher duplicates", 1.0,
        lambda: json.load(open(ART / "profiling" / "relation_profiling.json"))
        ["datasets"]["vnat"]["fields"]["ja3"]["train_test_overlap"],
        src="profiling/relation_profiling.json")
    add("Sec 6.3 HIKARI HGB random split", 0.7165,
        lambda: json.load(open(ART / "repeated_splits" / "hikari.json"))
        ["hikari"]["random_s42"]["hgb"]["mean"], src="repeated_splits/hikari.json")
    add("Sec 6.2 BCCC random-edge minus self-loop", -0.0088,
        lambda: mean("random_control/bccc_dohbrw.json") - noedge("bccc_dohbrw"),
        src="random_control/ + no_edge/")
    add("Sec 6.5 VNAT full pool mean", 0.9796,
        lambda: float(np.mean([mean("split_protocol/relation_ranking/ts/%s/vnat.json" % r)
                               for r in ("via_sni", "via_ja3", "via_cert_subject",
                                         "via_alpn", "via_cert_issuer",
                                         "via_tls_cipher_group", "via_cert_validity",
                                         "via_src_host", "via_dst_host", "via_timebin")])),
        src="split_protocol/relation_ranking/ts/")
    # --- derived values quoted in the text ---
    add("Sec 6.3 VNAT cutoff HAN max", 0.9972,
        lambda: max(v["han"]["mean"] for k, v in
                    json.load(open(ART / "repeated_splits" / "vnat.json"))["vnat"].items()
                    if k.startswith("cutoff")), src="repeated_splits/vnat.json")

    add("Sec 6.5 ISCX single relation VersionCipher", 0.6686,
        lambda: mean("split_protocol/relation_ranking/ts/via_ja3/iscx_vpn.json"),
        src="relation_ranking/")
    add("Sec 6.5 VNAT full-pool HAN", 0.9088,
        lambda: mean("typing_vs_pruning/han_full/vnat.json"), src="typing_vs_pruning/")
    add("Sec 6.3 VNAT cutoff GCN min", 0.7452,
        lambda: min(v["gcn"]["mean"] for k, v in
                    json.load(open(ART / "repeated_splits" / "vnat.json"))["vnat"].items()
                    if k.startswith("cutoff")), src="repeated_splits/vnat.json")
    add("Sec 6.3 VNAT cutoff GCN max", 0.9345,
        lambda: max(v["gcn"]["mean"] for k, v in
                    json.load(open(ART / "repeated_splits" / "vnat.json"))["vnat"].items()
                    if k.startswith("cutoff")), src="repeated_splits/vnat.json")
    add("Sec 6.3 VNAT cutoff HAN min", 0.9504,
        lambda: min(v["han"]["mean"] for k, v in
                    json.load(open(ART / "repeated_splits" / "vnat.json"))["vnat"].items()
                    if k.startswith("cutoff")), src="repeated_splits/vnat.json")
    add("Sec 6.2 ISCX SrcHost edge count", 1780,
        lambda: json.load(open(ART / "masked" / "edge_statistics.json"))
        ["iscx_vpn"]["relations"]["via_src_host"]["raw"], tol=0,
        src="masked/edge_statistics.json")
    add("Sec 6.2 ISCX SrcHost edges removed by masking", 182,
        lambda: (lambda r: r["raw"] - r["kept"])(
            json.load(open(ART / "masked" / "edge_statistics.json"))
            ["iscx_vpn"]["relations"]["via_src_host"]), tol=0,
        src="masked/edge_statistics.json")

    add("Sec 6.2 ISCX masking drop", 0.3020,
        lambda: han("iscx_vpn") - masked("iscx_vpn"),
        src="main/han + masked/han")
    add("Sec 6.3 HIKARI MLP random split", 0.6055,
        lambda: json.load(open(ART / "repeated_splits" / "hikari.json"))
        ["hikari"]["random_s42"]["mlp"]["mean"], src="repeated_splits/hikari.json")
    add("Sec 6.2 ISCX random-edge (bucket 100, 3 seeds)", 0.6132,
        lambda: mean("random_control/sweep/g100_s44.json", dataset="iscx_vpn"),
        src="random_control/sweep/")
    add("Sec 6.5 ISCX tau among cutoff rankings", 0.7156,
        lambda: float(np.mean([kendall(a, b) for a, b in
                               itertools.combinations(ranks("iscx_vpn")[2], 2)])),
        src="repeated_splits/iscx_vpn.json")

    add("Sec 6.1 CIC unpaired CI lower", -0.0075, lambda: boot("cic_andmal__han_minus_noedge",
                                                    "unpaired_ci")[0],
        src="bootstrap_ci.json")
    add("Sec 6.1 CIC unpaired CI upper", 0.0106, lambda: boot("cic_andmal__han_minus_noedge",
                                                   "unpaired_ci")[1],
        src="bootstrap_ci.json")

    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only claims whose name contains this string")
    args = ap.parse_args()
    ok, bad, skipped = 0, [], []
    print("%-64s %10s %12s  %s" % ("claim", "stated", "recomputed", "status"))
    print("-" * 96)
    for name, stated, fn, tol, src in build_claims():
        if args.only and args.only not in name:
            continue
        try:
            got = fn()
        except Exception as e:
            skipped.append((name, "%s: %s" % (type(e).__name__, e), src)); continue
        if got is None:
            skipped.append((name, "source file missing", src)); continue
        good = abs(float(got) - float(stated)) <= tol
        ok += good
        if not good: bad.append((name, stated, got, src))
        print("%-64s %10s %12.6g  %s" % (name[:64], stated, got, "PASS" if good else "FAIL"))
    print("-" * 96)
    print("passed %d, failed %d, skipped %d" % (ok, len(bad), len(skipped)))
    for n, s_, g, src in bad:
        print("  FAIL %-50s stated %-10s recomputed %-12.6g <- %s" % (n, s_, g, src))
    for n, why, src in skipped:
        print("  SKIP %-50s %s <- %s" % (n, why, src))
    if not bad and not skipped:
        print("ALL CLAIMS PASS")
    return 1 if (bad or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
