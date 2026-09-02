# -*- coding: utf-8 -*-
"""A relation built from nothing, at matched edge count and degree (Sec. 6.2).

The other three controls take edges away or change what they carry. None of
them puts edges in. That leaves one question open. A graph model may gain over
a feature-only baseline simply because each node now averages over a few
neighbours, whatever those neighbours are. Smoothing of that kind is a property
of the architecture and the sample size, not of the metadata.

This control supplies the missing arm. Flows are assigned to buckets that no
field determines, and the standard sampler joins each flow to two others inside
its bucket. Edge count and degree distribution match a relation whose field is
populated everywhere, and homophily sits at chance by construction, which
`edge_homophily.py` verifies afterwards. What separates this graph from a real
relation is only that its edges read no metadata.

A bucket holds about `--group-size` flows. The size changes the reach of the
relation without changing the number of edges, since the degree is fixed, so
sweeping it separates a local averaging effect from long-range propagation.

Assignment is a function of the flow's addresses and timestamp, so a rebuild
reproduces it and the row order does not matter. Where a dataset has no
timestamp, the addresses of many flows coincide; repeated keys are separated by
their occurrence index so that the buckets do not inherit the endpoint
structure they are meant to be free of.

Usage:
    python random_edge_control.py --data data/processed_time_strat_random \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --group-size 100 --seed 42

The directory given to --data must be a copy. The script writes a column into
meta_*.csv and an edge file beside it.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_graph import edges_for_values
from common import SPLITS, is_valid

KEY_COLS = ("src_ip", "sport", "dst_ip", "dport", "ts")
RELATION = "via_random"
COLUMN = "random_bucket"
MISSING = "-"   # 빌더의 결측 표기

# 다른 실험이 읽는 루트. 여기에 열을 쓰면 그 실험의 입력이 바뀐다.
PROTECTED = {
    "processed_deg2",
    "processed_deg2_bccc10",
    "processed_time_strat_random",
    "processed_degree4",
    "processed_time_strat_remap",
}


def refuse_if_source(root: Path) -> None:
    if root.name in PROTECTED:
        raise SystemExit(
            "'%s' is a root other experiments read. This script writes a column\n"
            "into meta_*.csv in place, so give it a copy:\n"
            "    cp -r data/%s data/_random" % (root.name, root.name))


def keys(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in KEY_COLS if c in df.columns]
    if len(cols) < 3:
        raise SystemExit("meta carries fewer than three address or time columns: "
                         + str(list(df.columns)))
    joined = df[cols].astype(str).agg("|".join, axis=1).tolist()
    seen: dict = {}
    out = np.empty(len(joined), dtype=np.uint64)
    for i, s in enumerate(joined):
        k = seen.get(s, 0)
        seen[s] = k + 1
        if k:
            s = "%s#%d" % (s, k)
        out[i] = int.from_bytes(
            hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "big")
    return out


def run(root: Path, name: str, group_size: int, seed: int, degree: int,
        match_column: str = "") -> None:
    if group_size < 4:
        raise SystemExit("--group-size must be at least 4 for a degree of two")
    d = root / name
    parts = [d / ("meta_%s.csv" % s) for s in SPLITS]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise SystemExit("missing: " + str(missing))

    frames = [pd.read_csv(p, low_memory=False) for p in parts]
    rows = pd.concat(frames, ignore_index=True)
    n = len(rows)

    salt = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    k = keys(rows) ^ salt

    # 커버리지를 맞출 열이 주어지면, 그 열이 값을 가진 flow 에만 버킷을 준다.
    #
    # 이것이 없으면 무작위 통제는 모든 flow 에 버킷을 주므로 고립 노드가 하나도
    # 없고, 그래서 실제 관계보다 촘촘한 그래프가 된다. 실제 관계는 필드가 없는
    # flow 를 고립시키는데, 엣지 수의 차이는 대부분 거기서 온다 — BCCC-DoH 의
    # CertValidity 는 토큰을 되돌리면 flow 의 71%가 고립되고 엣지가 59% 준다.
    # 버킷 크기만 바꿔서는 그 상태에 닿을 수 없다. 값이 있는 flow 의 집합을
    # 그대로 빌려 와야 "같은 밀도의 무의미한 그래프" 가 된다.
    #
    # 빌려 오는 것은 **어느 flow 가 값을 갖는가** 하나이고 값 자체는 아니다.
    # 그래서 이 통제는 여전히 값의 의미를 지운다.
    covered = np.ones(n, dtype=bool)
    if match_column:
        if match_column not in rows.columns:
            raise SystemExit("%s: no column %s to match coverage from"
                             % (name, match_column))
        covered = np.array([is_valid(str(v)) for v in rows[match_column].values])
        if not covered.any():
            raise SystemExit("%s: %s is empty on every flow" % (name, match_column))

    n_cov = int(covered.sum())
    n_buckets = max(1, n_cov // group_size)
    # A prefix is added because the builder treats a bare "0" as a missing
    # value, which would drop bucket zero entirely.
    bucket = np.full(n, MISSING, dtype=object)
    bucket[covered] = ["b%d" % x for x in (k[covered] % np.uint64(n_buckets))]
    bucket = np.asarray(bucket, dtype=object)

    written = 0
    for p, f in zip(parts, frames):
        f[COLUMN] = bucket[written:written + len(f)]
        f.to_csv(p, index=False)
        written += len(f)

    e = edges_for_values(bucket, seed, degree, keys(rows))
    np.save(d / ("hin_edges_%s.npy" % RELATION), e)

    sizes = pd.Series(bucket[covered]).value_counts()
    deg = np.bincount(e[0], minlength=n) if e.size else np.zeros(n, int)
    print("  [%s] %d flows, covered %d (%.4f), %d buckets, median size %d"
          % (name, n, n_cov, n_cov / n, n_buckets, int(sizes.median())))
    print("      edges %d, isolated %d" % (e.shape[1], int((deg == 0).sum())))
    small = int((sizes < 4).sum())
    if small:
        print("      %d buckets hold fewer than four flows, so their members "
              "receive a degree below two" % small)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True, help="a copy, not the original")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--group-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--match-coverage-from", default="",
                    help="이 메타 열이 값을 가진 flow 에만 버킷을 준다. "
                         "실제 관계와 같은 고립 구조를 만들 때 쓴다")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit("no such path: " + str(root))
    refuse_if_source(root)
    print("[random edge control] group size %d, seed %d, degree %d%s"
          % (args.group_size, args.seed, args.degree,
             ", coverage matched to %s" % args.match_coverage_from
             if args.match_coverage_from else ""))
    for name in args.datasets:
        run(root, name, args.group_size, args.seed, args.degree,
            args.match_coverage_from)
    print("\nTrain HAN on this directory with the relation set '%s'." % RELATION)
    print("Then run edge_homophily.py on it. Homophily should sit at chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
