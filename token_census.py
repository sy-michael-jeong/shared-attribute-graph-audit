# -*- coding: utf-8 -*-
"""추출기가 채워 넣은 문자열이 각 필드에 얼마나 있는가 (Sec. 3.3, 6.1).

`extract_pcap.py` 는 값을 읽지 못했을 때 필드를 비우지 않고 문자열을 채운다.
인증서 날짜를 파싱하지 못하면 `unknown`, 유효기간이 음수면 `invalid_range`,
암호군을 매핑하지 못하면 `Unknown`, 발급자를 정규화하지 못하면
`Private_or_Unknown` 이다. 어느 것도 `common.MISSING_TOKENS` 에 없으므로 값으로
세어지고, **같은 실패를 겪은 flow 끼리 서로 연결된다.**

다섯 번째 통제는 그 토큰을 결측으로 되돌린다. 그 통제가 어느 필드에서 의미가
있는지는 통제를 돌리기 전에 여기서 알 수 있다 — 토큰이 열아홉 개뿐인 필드를
되돌려 봐야 엣지가 서른여덟 개 줄 뿐이다.

카운트만 하므로 모델도 GPU 도 쓰지 않는다.

    python token_census.py --data data/processed_deg2_bccc10 \\
        --datasets bccc_dohbrw \\
        --out results/reference_lines/token_census.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import MISSING_TOKENS, RELATION_COLUMN, SPLITS, is_valid

# 추출기가 실패를 표시하려고 넣는 문자열. 열 이름 -> 토큰들.
SYNTHETIC = {
    "cert_validity_bucket": ["unknown", "invalid_range"],
    "tls_cipher_group": ["Unknown"],
    "cert_issuer_org": ["Private_or_Unknown", "PublicCA_Other"],
}

FIELDS = [c for r, c in RELATION_COLUMN.items() if r != "via_timebin"]


def load(d: Path):
    metas = [pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
             for s in SPLITS]
    ys = [np.load(d / ("y_%s.npy" % s)) for s in SPLITS]
    return pd.concat(metas, ignore_index=True), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    report = {
        "data_root": str(root),
        "question": "how much of what is counted as a value is a string the "
                    "extractor wrote because it could not read one",
        "missing_tokens": list(MISSING_TOKENS),
        "synthetic_tokens": SYNTHETIC,
        "datasets": {},
    }

    for name in args.datasets:
        meta, y = load(root / name)
        n = len(meta)
        block = {"n_flows": int(n), "fields": {}}
        for col in FIELDS:
            if col not in meta.columns:
                continue
            v = meta[col].astype(str).str.strip()
            valid = np.array([is_valid(x) for x in v.values])
            n_valid = int(valid.sum())
            toks = {}
            for t in SYNTHETIC.get(col, []):
                c = int((v == t).sum())
                toks[t] = {"n_flows": c,
                           "share_of_valid": round(c / n_valid, 4) if n_valid else 0.0,
                           "n_attack": int(((v == t) & (y != 0)).sum())}
            synth = sum(x["n_flows"] for x in toks.values())
            block["fields"][col] = {
                "coverage": round(n_valid / n, 4),
                "n_valid": n_valid,
                "n_missing": int(n - n_valid),
                "synthetic_tokens": toks,
                "n_synthetic": synth,
                # 채움 문자열을 결측으로 되돌렸을 때 남는 커버리지.
                "coverage_after_reversal": round((n_valid - synth) / n, 4),
                "share_of_valid_that_is_synthetic":
                    round(synth / n_valid, 4) if n_valid else 0.0}
        report["datasets"][name] = block

        print("== %s  (%d flow)" % (name, n))
        for col, f in sorted(block["fields"].items(),
                             key=lambda kv: -kv[1]["n_synthetic"]):
            if not f["synthetic_tokens"]:
                continue
            print("   %-22s 커버리지 %.4f -> %.4f   채움 %d (유효의 %.1f%%)"
                  % (col, f["coverage"], f["coverage_after_reversal"],
                     f["n_synthetic"],
                     100 * f["share_of_valid_that_is_synthetic"]))
            for t, x in f["synthetic_tokens"].items():
                print("        %-22s %8d   그중 공격 %d" %
                      (t, x["n_flows"], x["n_attack"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n[saved] %s" % out)


if __name__ == "__main__":
    main()
