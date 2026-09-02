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
`아티팩트_적격판정.py` refuses.

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

# 훈련 스크립트가 모델마다 쓰는 파일 이름.
MODEL_FILE = {
    "han": "multiseed_summary.json",
    "gcn": "homogeneous_gcn_summary.json",
    "mlp": "mlp_baseline_summary.json",
    "egs": "egraphsage_summary.json",
    "hgb": "tabular_baseline_summary.json",
}


# 스크립트가 `--out results/...` 로 결과 트리에 직접 쓰는 파일. 옮길 것이
# 없으므로 대응표에 넣지 않되, "출처 없음" 으로 잡히지 않도록 여기 적는다.
# 어느 명령이 만드는지는 README 의 재현 절에 있다.
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
    "profiling/README.txt",                            # 수기
    "profiling/relation_profiling.json",               # profile_relations.py
    "rank_stability/rank_stability.json",              # rank_stability.py
    "reproducibility/graph_rebuild.json",              # verify_graph.py
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
    """(run 출력 경로, 출하 경로) 목록. 출하되는 모든 파일이 여기 있어야 한다."""
    m = []

    def add(src, dst):
        m.append((src, dst))

    # -- 본 표 (Sec. 6.1) -------------------------------------------------
    for ds in DS:
        add("main/han/%s/multiseed_summary.json" % ds,
            "main/han/%s/multiseed_summary.json" % ds)
        add("main/gcn/%s/homogeneous_gcn_summary.json" % ds,
            "main/gcn/%s/homogeneous_gcn_summary.json" % ds)
    add("main/mlp/mlp_baseline_summary.json", "main/mlp_summary.json")
    add("main/egraphsage/egraphsage_summary.json",
        "main/egraphsage/egraphsage_summary.json")
    add("main/tabular/tabular_summary.json", "main/tabular_summary.json")

    # -- 통제 -------------------------------------------------------------
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

    # 무작위 엣지. 데이터셋 하나에 파일 하나로 모은다. ISCX 와 VNAT 은 한 번에
    # 돌렸으므로 합본 이름을 그대로 쓴다.
    for ds in ["bccc_dohbrw", "cic_andmal", "hikari"]:
        add("random_control/%s/multiseed_summary.json" % ds,
            "random_control/%s.json" % ds)
    add("random_control/iscx_vpn_vnat/multiseed_summary.json",
        "random_control/iscx_vpn_vnat.json")
    for g in (10, 100, 1000):
        for s in (42, 43, 44):
            add("random_control/sweep/g%d_s%d/multiseed_summary.json" % (g, s),
                "random_control/sweep/g%d_s%d.json" % (g, s))

    # -- 탐색과 선택 (Sec. 6.2) -------------------------------------------
    for ds in ["bccc_dohbrw", "iscx_vpn", "hikari", "vnat"]:
        add("saturation/%s/combinatorial_grand_summary.json" % ds,
            "saturation/%s/combinatorial_grand_summary.json" % ds)
        add("selection/%s/%s.json" % (ds, ds), "selection/%s/%s.json" % (ds, ds))
        add("selection/%s/rows.json" % ds, "selection/%s/rows.json" % ds)
    for ds in ["bccc_dohbrw", "iscx_vpn", "vnat"]:
        add("typing_vs_pruning/han_full/%s/multiseed_summary.json" % ds,
            "typing_vs_pruning/han_full/%s.json" % ds)

    # -- 분할 프로토콜 (Sec. 6.3) -----------------------------------------
    for ds in ["bccc_dohbrw", "iscx_vpn", "hikari", "vnat"]:
        add("repeat/%s/repeat_summary.json" % ds, "repeated_splits/%s.json" % ds)
    add("split_protocol/random/han/bccc_dohbrw/multiseed_summary.json",
        "split_protocol/random/han/bccc_dohbrw/multiseed_summary.json")
    for rel in RELATIONS:
        for ds in ["iscx_vpn", "vnat"]:
            add("relation_ranking/ts/%s/%s/multiseed_summary.json" % (rel, ds),
                "split_protocol/relation_ranking/ts/%s/%s.json" % (rel, ds))

    # -- 단일 관계와 민감도 -----------------------------------------------
    add("single_relation/bccc_dohbrw_nounk/multiseed_summary.json",
        "single_relation/bccc_dohbrw_nounk.json")
    add("single_relation/bccc_dohbrw_with_host/multiseed_summary.json",
        "single_relation/bccc_dohbrw_with_host.json")
    add("single_relation/cic_andmal/multiseed_summary.json",
        "single_relation/cic_andmal.json")
    for tag in ["degree4_bccc", "degree4_iscx", "tb60", "tb600"]:
        add("sensitivity/%s/multiseed_summary.json" % tag,
            "sensitivity/%s.json" % tag)

    # -- 문헌 근사 구성 ----------------------------------------------------
    # 자기 엣지를 뺀 E-GraphSAGE 통제, 그리고 CIC 의 합성 토큰 되돌림.
    add("egs_no_self/egraphsage_summary.json",
        "egs_no_self/egraphsage_summary.json")
    add("nounk_cic/multiseed_summary.json", "token_reversal/cic_andmal.json")

    # 예산 4 를 관계 몫이 작은 세 데이터셋에도. CIC 는 24GB 에 올라가지 않아
    # 다섯 시드가 모두 실패로 기록된다. 그 실패 자체가 결과다.
    for ds in ["hikari", "cic_andmal", "vnat"]:
        add("degree4_weak/%s/multiseed_summary.json" % ds,
            "sensitivity_weak/%s.json" % ds)

    # 문헌 근사 구성의 순열은 그 구성이 쓰는 두 관계를 섞어야 한다.
    for s_ in SEEDS:
        add("reta_perm_ja3/seed_%d/multiseed_summary.json" % s_,
            "reta_approx/permutation/seed_%d.json" % s_)
    add("reta_noedge/bccc_dohbrw/multiseed_summary.json", "reta_approx/no_edge.json")

    # 분할마다 3분해. 자기루프가 있어야 귀속이 분할에 얼마나 의존하는지 잰다.
    add("repeat_decomp_iscx/repeat_summary.json", "repeat_decomp/iscx_vpn.json")
    add("_check_bccc/repeat_summary.json", "repeat_decomp/bccc_dohbrw_check.json")
    add("repeat_decomp_bccc/repeat_summary.json", "repeat_decomp/bccc_dohbrw.json")
    add("repeat_decomp_cic/repeat_summary.json", "repeat_decomp/cic_andmal.json")

    # 식별자 축을 실제로 자른 유일한 분할. 표 형태 기준선이 여기서 무너진다.
    add("dstdisj/hgb/tabular_baseline_summary.json", "dst_disjoint/hgb.json")
    add("dstdisj/mlp/mlp_baseline_summary.json", "dst_disjoint/mlp.json")
    add("dstdisj/noedge/iscx_vpn/multiseed_summary.json", "dst_disjoint/noedge.json")
    add("dstdisj/han/multiseed_summary.json", "dst_disjoint/han.json")

    # 밀도까지 맞춘 무작위 통제. 커버리지를 실제 관계에서 빌려 온다.
    add("random_match_bccc/multiseed_summary.json", "random_match/bccc_dohbrw.json")
    # 같은 실행이 평평한 이름으로도 출하되어 있다. 둘 다 대응표에 둔다 —
    # 출처가 없는 파일이 결과 트리에 남아 있는 것이 이 검사기가 막는 것이다.
    add("random_match_bccc/multiseed_summary.json", "random_match_bccc.json")

    # TLS 계열 대 호스트 계열 직접 대결. 두 계열이 모두 실현되는 세 곳에서만 된다.
    for fam in ("tls", "host"):
        for ds in ("iscx_vpn", "vnat", "bccc_dohbrw"):
            add("family/%s/%s/multiseed_summary.json" % (fam, ds),
                "relation_family/%s_%s.json" % (fam, ds))


    # 관계 수가 같은 조건에서의 타입 구분 효과.
    add("han_full_hikari/multiseed_summary.json",
        "typing_vs_pruning/han_full/hikari.json")
    add("noedge_bccc10/bccc_dohbrw/multiseed_summary.json",
        "no_edge/bccc_dohbrw_bccc10.json")

    # 자기 엣지를 뺀 E-GraphSAGE.


    for tag in ["han", "masked", "no_edge"]:
        add("reta/%s/multiseed_summary.json" % tag, "reta_approx/%s.json" % tag)
    # runs/reta/permutation/ 은 보고 구성의 필드(CertValidity)를 섞은 것이라 문헌
    # 구성(ja3+cert_subject)의 통제가 아니다. 옮기지 않는다. 올바른 순열은 위의
    # reta_perm_ja3 → reta_approx/permutation/ 다. (2026-09-02 제거)

    # -- 그래프 구성 기록 --------------------------------------------------
    # build_graph.py 는 데이터 디렉터리에 쓴다. 데이터는 배포하지 않으므로
    # 기록만 여기로 옮긴다.
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
            print("  없음  %-56s -> %s" % (src, dst))

    print("\n대응 %d 개 — 있음 %d, 없음 %d%s"
          % (len(m), have, missing, "  (검사만)" if args.check else ""))

    # 출하 트리에 있는데 이 표에 없는 파일. 있으면 출처가 없는 것이다.
    mapped = {d for _, d in m} | set(DIRECT)
    extra = [str(f.relative_to(res)).replace("\\", "/")
             for f in res.rglob("*") if f.is_file()
             and str(f.relative_to(res)).replace("\\", "/") not in mapped]
    if extra:
        print("\n[출처 없는 출하 파일] 이 목록이 비어야 한다")
        for e in sorted(extra):
            print("  %s" % e)
    else:
        print("\n출하 트리의 모든 파일이 대응표 %d 또는 직접 출력 %d 에 있다"
              % (len(m), len(DIRECT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
