# -*- coding: utf-8 -*-
"""출하된 그래프가 이 코드로 다시 만들어지는지 확인한다.

엣지 구축은 완전히 결정적이다. `edges_for_values` 의 순열 시드는
`blake2b(value) XOR seed` 이고 (`build_graph.py:30-39`), 파이썬 내장 `hash` 를
쓰지 않으므로 프로세스마다 달라지지 않는다. 그러므로 같은 메타데이터와 같은
설정으로 다시 지으면 **바이트 단위로 같은 배열**이 나와야 한다.

이것이 왜 중요한가. 출하된 결과는 서버의 `scripts/` 트리(`preprocess_hin.py`,
`multiseed_eval.py`, ...)가 만들었고, 이 아티팩트의 `build_graph.py` 와
`train.py` 는 그것들을 하나로 합쳐 새로 쓴 것이다. 신경망 점수는 시드마다
달라지므로 같은 값이 나오는지로 두 코드가 같은 일을 하는지 판정할 수 없다.
**그래프는 판정할 수 있다.** 그래프가 같으면 남는 차이는 이미 논문이 보고하는
시드 변동뿐이다.

메타데이터는 다시 만들지 않는다. `--skip-materialize` 없이 돌리면 원본 캡처에서
`meta_*.csv` 를 다시 써버리므로 비교 대상이 사라진다.

    python verify_graph.py --data data/processed_deg2 \\
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \\
        --config config.yaml --work data/_verify_graph
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from build_graph import build, requested_relations


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True, help="출하된 데이터 루트")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--work", default="data/_verify_graph")
    ap.add_argument("--out", default=None, help="판정 결과 JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    src_root, work = Path(args.data), Path(args.work)
    report = {"data_root": str(src_root), "config": args.config, "datasets": {}}
    all_ok = True

    for name in args.datasets:
        src = src_root / name
        dst = work / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        # 메타데이터만 복사한다. 엣지는 이 코드가 새로 만든다.
        for p in ("train", "val", "test"):
            shutil.copy2(src / ("meta_%s.csv" % p), dst / ("meta_%s.csv" % p))
            shutil.copy2(src / ("y_%s.npy" % p), dst / ("y_%s.npy" % p))

        print("\n== %s" % name)
        build(name, work, cfg, config_path=args.config)

        req = requested_relations(cfg, name)
        block = {"requested": req, "relations": {}}
        ok_ds = True
        for rel in req:
            a, b = src / ("hin_edges_%s.npy" % rel), dst / ("hin_edges_%s.npy" % rel)
            if not a.exists() and not b.exists():
                block["relations"][rel] = {"verdict": "둘 다 없음 (건너뜀)"}
                continue
            if a.exists() != b.exists():
                block["relations"][rel] = {
                    "verdict": "한쪽에만 있다",
                    "shipped": a.exists(), "rebuilt": b.exists()}
                ok_ds = False
                continue
            same_bytes = sha(a) == sha(b)
            ea, eb = np.load(a), np.load(b)
            same_arr = ea.shape == eb.shape and bool((ea == eb).all())
            block["relations"][rel] = {
                "shipped_edges": int(ea.shape[1]),
                "rebuilt_edges": int(eb.shape[1]),
                "identical_bytes": same_bytes,
                "identical_values": same_arr,
                "verdict": "일치" if same_arr else "다르다"}
            if not same_arr:
                ok_ds = False
            print("   %-24s 출하 %8d  재생성 %8d  %s"
                  % (rel, ea.shape[1], eb.shape[1],
                     "일치" if same_arr else "!!! 다르다"))

        block["all_identical"] = ok_ds
        report["datasets"][name] = block
        all_ok &= ok_ds
        print("   -> %s" % ("전부 일치" if ok_ds else "!!! 불일치 있음"))

    report["verdict"] = ("이 코드가 출하된 그래프를 그대로 다시 만든다"
                         if all_ok else "재현되지 않는 관계가 있다")
    print("\n%s" % report["verdict"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
        print("[saved] %s" % out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
