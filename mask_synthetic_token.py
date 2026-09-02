# -*- coding: utf-8 -*-
"""meta_*.csv 의 한 열에서 추출기가 채운 합성 토큰을 결측 표기로 바꾼다.

값을 읽지 못했을 때 추출기는 칸을 비우지 않고 'unknown' 같은 문자열을
채운다. 그래프 빌더의 MISSING_TOKENS 에 그 문자열이 없으므로, 값을 못 읽은
flow 끼리 "같은 값을 공유한다"고 판정되어 엣지가 생긴다. 그렇게 만들어진
관계는 TLS 값이 아니라 파싱 실패 패턴을 잇는다.

이 스크립트는 그 토큰을 빌더가 결측으로 보는 '-' 로 바꾼다. 관계가 실제로
읽힌 값에만 의존할 때 무엇이 남는지 보기 위한 통제다. 결과가 유지되면
발견이 토큰 선택에 기대지 않는다는 증거가 되고, 무너지면 그것이 결과다.

meta CSV 를 제자리에서 고쳐 쓰므로 반드시 복사본을 준다. 원본 이름으로는
실행을 거부한다.

Usage:
    cp -r data/processed_deg2_bccc10 data/processed_deg2_bccc10_nounk
    python -u mask_synthetic_token.py \
        --data data/processed_deg2_bccc10_nounk --datasets bccc_dohbrw \
        --column cert_validity_bucket --tokens unknown invalid_range
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

# 빌더가 결측으로 읽는 표기 가운데 하나. src/data/flow_graph_hin.py 의
# MISSING_TOKENS 와 같은 집합에서 골랐다.
MISSING = "-"

# 다른 실험이 읽는 루트. 여기에 쓰면 이미 나온 결과가 재현되지 않으므로
# 막는다. 복사본에만 실행한다.
PROTECTED = {
    "processed",
    "processed_time_strat",
    "processed_deg2",
    "processed_deg2_bccc10",
    "processed_dst_ip_disjoint",
    "processed_random_v2",
    "processed_degree4",
    "processed_time_strat_remap",
}

PARTS = ("train", "val", "test")


def refuse_if_source(root: Path) -> None:
    if root.name in PROTECTED:
        raise SystemExit(
            "'%s' 는 다른 실험이 읽는 루트다. meta CSV 를 제자리에서 고치므로\n"
            "복사본을 만들어 그 경로를 준다:\n"
            "    cp -r data/%s data/%s_nounk" % (root.name, root.name, root.name))


def run(root: Path, name: str, column: str, tokens: list) -> None:
    d = root / name
    paths = [d / ("meta_%s.csv" % s) for s in PARTS]
    absent = [str(p) for p in paths if not p.exists()]
    if absent:
        raise SystemExit("없는 파일: " + str(absent))

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    for p, f in zip(paths, frames):
        if column not in f.columns:
            raise SystemExit(
                "'%s' 에 열 '%s' 가 없다. 있는 열: %s"
                % (p, column, str(list(f.columns))))

    joined = pd.concat(frames, ignore_index=True)
    before = joined[column].astype(str).str.strip().value_counts()

    if MISSING in [str(t).strip() for t in tokens]:
        raise SystemExit("바꿀 토큰과 결측 표기가 같다")
    hit = {t: int(before.get(t, 0)) for t in tokens}
    if not any(hit.values()):
        raise SystemExit(
            "'%s' 열에 %s 가 하나도 없다. 상위 값: %s"
            % (column, str(tokens), str(before.head(8).to_dict())))

    written = 0
    for p, f in zip(paths, frames):
        col = f[column].astype(str).str.strip()
        f[column] = col.where(~col.isin(tokens), MISSING)
        f.to_csv(p, index=False)
        written += len(f)

    after = pd.concat(frames, ignore_index=True)[column].astype(str).str.strip()
    n_missing = int(after.eq(MISSING).sum())
    n_value = written - n_missing

    print("  [%s] %s, flow %s" % (name, column, f"{written:,}"))
    for t, c in hit.items():
        print("      '%s' -> '%s'  %s" % (t, MISSING, f"{c:,}"))
    print("      결측이 된 flow %s, 값이 남은 flow %s"
          % (f"{n_missing:,}", f"{n_value:,}"))
    print("      남은 값 종류 %d" % after[after != MISSING].nunique())


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True,
                    help="처리된 데이터 루트. 원본이 아니라 복사본을 준다")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--column", required=True,
                    help="meta CSV 의 열 이름. 예: cert_validity_bucket")
    ap.add_argument("--tokens", nargs="+", required=True,
                    help="결측으로 바꿀 문자열. 예: unknown invalid_range")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit("없는 경로: " + str(root))
    refuse_if_source(root)

    for name in args.datasets:
        run(root, name, args.column, list(args.tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
