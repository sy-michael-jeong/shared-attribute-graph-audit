# -*- coding: utf-8 -*-
"""Move run outputs into the layout the paper cites.

The training scripts write one file per model under whatever `--runs` directory
they were given. The shipped `results/` tree is arranged by experiment instead,
and several files are renamed so that a reader can find them from a section
number. Those two facts used to be connected by nothing at all: the README
listed commands that wrote to `runs/`, the shipped files sat in `results/`, and
no code joined them. Of 178 shipped files, the documented commands produced 31
at the path where they were shipped.

This script is that join, written down. Every shipped file appears in `MAP`
with the run output it comes from, so a reader can follow

    training script  ->  runs/<tag>/<file>  ->  collect.py  ->  results/<path>

and an entry with no source is a file that cannot be produced, which is what
the shipping check refuses.

    python collect.py --runs runs --results results          # copy
    python collect.py --runs runs --results results --check  # report only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DS = ["bccc_dohbrw", "iscx_vpn", "hikari", "cic_andmal", "vnat"]
SEEDS = [41, 42, 43, 44, 45]
RELATIONS = ["via_sni", "via_ja3", "via_cert_subject", "via_alpn",
             "via_cert_issuer", "via_tls_cipher_group", "via_cert_validity",
             "via_src_host", "via_dst_host", "via_timebin"]

# File name each training script writes per model.
MODEL_FILE = {
    "han": "multiseed_summary.json",
    "gcn": "homogeneous_gcn_summary.json",
    "mlp": "mlp_baseline_summary.json",
    "egs": "egraphsage_summary.json",
    "hgb": "tabular_baseline_summary.json",
}


# Files that scripts write directly into the result tree via `--out results/...`.
# Nothing to move, so they are not in the mapping, but they are listed here so
# the check does not flag them as "no source". The producing commands are in the
# README.
DIRECT = [
    "composition/composition_order_preserving.json",   # dataset_composition.py
    "composition/hikari_drift.json",                   # audit.py drift
    "dataset_suitability/ctu13_tls_coverage.json",     # ctu13_coverage.py
    "feature_audit/hikari_index_column_random.json",   # check_index_feature.py
    "feature_audit/feature_matrix_audit.json",         # audit.py features
    "group_split/group_split_feasibility.json",        # group_split_feasibility.py
    "homophily/edge_homophily.json",                   # edge_homophily.py
    "homophily/edge_homophily_bccc10.json",            # edge_homophily.py
    "homophily/edge_homophily_bccc10_nounk.json",      # edge_homophily.py
    "homophily/edge_homophily_random_control.json",    # edge_homophily.py
    "masked/edge_statistics.json",                     # mask_edges.py --stats
    "model_comparison/bootstrap_ci.json",              # bootstrap_ci.py
    "neural_metadata/summary.json",                    # neural_metadata.py
    "profiling/README.txt",                            # hand-written
    "profiling/relation_profiling.json",               # profile_relations.py
    "rank_stability/rank_stability.json",              # rank_stability.py
    "reproducibility/graph_rebuild.json",              # verify_graph.py
    "structure/verify_graph.json",                     # verify_graph.py --out
    "reference_lines/token_census.json",                # token_census.py
    "reference_lines/token_census_all.json",            # token_census.py
    "token_reversal/cic_andmal_hin_summary.json",       # build_graph.py
    "sensitivity_weak/hikari_hin_summary.json",         # build_graph.py --max-degree
    "sensitivity_weak/cic_andmal_hin_summary.json",     # build_graph.py --max-degree
    "sensitivity_weak/vnat_hin_summary.json",           # build_graph.py --max-degree
    "split_protocol/overlap/overlap_dst_disjoint.json", # split_overlap_audit.py
    "reference_lines/availability_null.json",          # availability_null.py
    "reference_lines/availability_null_nounk.json",    # availability_null.py
    "reference_lines/availability_rule.json",          # availability_rule.py
    "reference_lines/availability_rule_nounk.json",    # availability_rule.py
    "reference_lines/summary.json",                    # baselines.py
    "split_protocol/overlap/overlap_order_preserving.json",  # split_overlap_audit.py
    "split_protocol/overlap/overlap_random.json",      # split_overlap_audit.py
    "split_protocol/overlap/overlap_random_cic.json",  # split_overlap_audit.py
    "structure/summary.json",                          # graph_stats.py
]


def build_map():
    """(run output path, shipped path) pairs. Every shipped file must appear here."""
    m = []

    def add(src, dst):
        m.append((src, dst))

    # -- main table (Table 4, Sec. 6.1) ---------------------------------------
    for ds in DS:
        add("main/han/%s/multiseed_summary.json" % ds,
            "main/han/%s/multiseed_summary.json" % ds)
        add("main/gcn/%s/homogeneous_gcn_summary.json" % ds,
            "main/gcn/%s/homogeneous_gcn_summary.json" % ds)
    add("main/mlp/mlp_baseline_summary.json", "main/mlp_summary.json")
    add("main/egraphsage/egraphsage_summary.json",
        "main/egraphsage/egraphsage_summary.json")
    add("main/tabular/tabular_summary.json", "main/tabular_summary.json")

    # -- controls -----------------------------------------------------------
    for ds in DS:
        add("no_edge/%s/multiseed_summary.json" % ds,
            "no_edge/%s/multiseed_summary.json" % ds)
        add("masked/han/%s/multiseed_summary.json" % ds,
            "masked/han/%s/multiseed_summary.json" % ds)
        for s in SEEDS:
            add("permutation/%s/seed_%d/multiseed_summary.json" % (ds, s),
                "permutation/%s/seed_%d/multiseed_summary.json" % (ds, s))
            add("permutation/%s/seed_%d/permutation_report.json" % (ds, s),
                "permutation/%s/seed_%d/permutation_report.json" % (ds, s))
        add("inductive/%s.json" % ds, "inductive/%s.json" % ds)

    # Random edges: one file per dataset. ISCX and VNAT were run together, so
    # the combined file name is kept.
    for ds in ["bccc_dohbrw", "cic_andmal", "hikari"]:
        add("random_control/%s/multiseed_summary.json" % ds,
            "random_control/%s.json" % ds)
    add("random_control/iscx_vpn_vnat/multiseed_summary.json",
        "random_control/iscx_vpn_vnat.json")
    for g in (10, 100, 1000):
        for s in (42, 43, 44):
            add("random_control/sweep/g%d_s%d/multiseed_summary.json" % (g, s),
                "random_control/sweep/g%d_s%d.json" % (g, s))

    # -- search and selection (Sec. 5.4, 6.5) --------------------------------
    for ds in ["bccc_dohbrw", "iscx_vpn", "hikari", "vnat"]:
        add("saturation/%s/combinatorial_grand_summary.json" % ds,
            "saturation/%s/combinatorial_grand_summary.json" % ds)
        add("selection/%s/%s.json" % (ds, ds), "selection/%s/%s.json" % (ds, ds))
        add("selection/%s/rows.json" % ds, "selection/%s/rows.json" % ds)
    for ds in ["bccc_dohbrw", "iscx_vpn", "vnat"]:
        add("typing_vs_pruning/han_full/%s/multiseed_summary.json" % ds,
            "typing_vs_pruning/han_full/%s.json" % ds)

    # -- split protocols (Sec. 6.3) -----------------------------------------
    for ds in ["bccc_dohbrw", "iscx_vpn", "hikari", "vnat"]:
        add("repeat/%s/repeat_summary.json" % ds, "repeated_splits/%s.json" % ds)
    add("split_protocol/random/han/bccc_dohbrw/multiseed_summary.json",
        "split_protocol/random/han/bccc_dohbrw/multiseed_summary.json")
    for rel in RELATIONS:
        for ds in ["iscx_vpn", "vnat"]:
            add("relation_ranking/ts/%s/%s/multiseed_summary.json" % (rel, ds),
                "split_protocol/relation_ranking/ts/%s/%s.json" % (rel, ds))

    # -- single relations and sensitivity ----------------------------------
    add("single_relation/bccc_dohbrw_nounk/multiseed_summary.json",
        "single_relation/bccc_dohbrw_nounk.json")
    add("single_relation/bccc_dohbrw_with_host/multiseed_summary.json",
        "single_relation/bccc_dohbrw_with_host.json")
    add("single_relation/cic_andmal/multiseed_summary.json",
        "single_relation/cic_andmal.json")
    for tag in ["degree4_bccc", "degree4_iscx", "tb60", "tb600"]:
        add("sensitivity/%s/multiseed_summary.json" % tag,
            "sensitivity/%s.json" % tag)

    # -- literature approximation (Sec. 6.6) --------------------------------
    # E-GraphSAGE with self-edges excluded, and CIC synthetic-token reversion.
    add("egs_no_self/egraphsage_summary.json",
        "egs_no_self/egraphsage_summary.json")
    add("nounk_cic/multiseed_summary.json", "token_reversal/cic_andmal.json")

    # Budget 4 on the three datasets with small relation contribution. CIC does
    # not fit in 24 GB; all five seeds are recorded as failures, and that failure
    # is itself a result.
    for ds in ["hikari", "cic_andmal", "vnat"]:
        add("degree4_weak/%s/multiseed_summary.json" % ds,
            "sensitivity_weak/%s.json" % ds)

    # The permutation of the literature approximation shuffles the two relations that configuration uses.
    for s_ in SEEDS:
        add("reta_perm_ja3/seed_%d/multiseed_summary.json" % s_,
            "reta_approx/permutation/seed_%d.json" % s_)
    add("reta_noedge/bccc_dohbrw/multiseed_summary.json", "reta_approx/no_edge.json")

    # Decomposition per split; the self-loop arm is needed to measure how the attribution depends on the split.
    add("repeat_decomp_iscx/repeat_summary.json", "repeat_decomp/iscx_vpn.json")
    add("_check_bccc/repeat_summary.json", "repeat_decomp/bccc_dohbrw_check.json")
    add("repeat_decomp_bccc/repeat_summary.json", "repeat_decomp/bccc_dohbrw.json")
    add("repeat_decomp_cic/repeat_summary.json", "repeat_decomp/cic_andmal.json")

    # The only split that actually separates an identifier axis; the tabular baseline collapses here.
    add("dstdisj/hgb/tabular_baseline_summary.json", "dst_disjoint/hgb.json")
    add("dstdisj/mlp/mlp_baseline_summary.json", "dst_disjoint/mlp.json")
    add("dstdisj/noedge/iscx_vpn/multiseed_summary.json", "dst_disjoint/noedge.json")
    add("dstdisj/han/multiseed_summary.json", "dst_disjoint/han.json")

    # Density-matched random control; coverage is borrowed from the real relation.
    add("random_match_bccc/multiseed_summary.json", "random_match/bccc_dohbrw.json")
    # The same run is also shipped under a flat name. Both stay in the mapping;
    # a file without a source in the result tree is exactly what this check forbids.
    add("random_match_bccc/multiseed_summary.json", "random_match_bccc.json")

    # TLS family vs host family; possible only on the three datasets where both are realized.
    for fam in ("tls", "host"):
        for ds in ("iscx_vpn", "vnat", "bccc_dohbrw"):
            add("family/%s/%s/multiseed_summary.json" % (fam, ds),
                "relation_family/%s_%s.json" % (fam, ds))


    # Effect of relation typing with the number of relations held equal.
    add("han_full_hikari/multiseed_summary.json",
        "typing_vs_pruning/han_full/hikari.json")
    add("noedge_bccc10/bccc_dohbrw/multiseed_summary.json",
        "no_edge/bccc_dohbrw_bccc10.json")

    # E-GraphSAGE with self-edges excluded.


    for tag in ["han", "masked", "no_edge"]:
        add("reta/%s/multiseed_summary.json" % tag, "reta_approx/%s.json" % tag)
    # runs/reta/permutation/ shuffled the reported configuration's field
    # (CertValidity) and is therefore not a control for the literature
    # configuration (ja3 + cert_subject). It is not shipped; the correct
    # permutation is reta_perm_ja3 -> reta_approx/permutation/ above.

    # -- graph construction records ----------------------------------------
    # build_graph.py writes into the data directory. The data is not
    # distributed, so only the records are moved here.
    for ds in DS:
        add("_data/processed_deg2/%s/hin_summary.json" % ds,
            "structure/hin_summary_%s.json" % ds)
        add("_data/processed_deg2/%s/feature_names.json" % ds,
            "feature_names/%s.json" % ds)
        add("_data/_random/%s/hin_summary.json" % ds,
            "random_control/structure/hin_summary_%s.json" % ds)
    add("_data/processed_deg2_bccc10/bccc_dohbrw/hin_summary.json",
        "single_relation/structure/hin_summary_bccc_dohbrw_10rel.json")
    return m


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    runs, res = Path(args.runs), Path(args.results)
    m = build_map()
    have = missing = 0
    for src, dst in m:
        s, d = runs / src, res / dst
        if s.is_file():
            have += 1
            if not args.check:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(s, d)
        else:
            missing += 1
            print("  missing %-56s -> %s" % (src, dst))

    print("\n%d mappings: %d present, %d missing%s"
          % (len(m), have, missing, "  (check only)" if args.check else ""))

    # Files in the shipped tree that are not in this table have no source.
    mapped = {d for _, d in m} | set(DIRECT)
    extra = [str(f.relative_to(res)).replace("\\", "/")
             for f in res.rglob("*") if f.is_file()
             and str(f.relative_to(res)).replace("\\", "/") not in mapped]
    if extra:
        print("\n[shipped files without a source] this list must be empty")
        for e in sorted(extra):
            print("  %s" % e)
    else:
        print("\nEvery shipped file is in the mapping (%d) or the direct-output list (%d)"
              % (len(m), len(DIRECT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
