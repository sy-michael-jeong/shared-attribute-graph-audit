# -*- coding: utf-8 -*-
"""귀납 붙이기 규칙이 지켜야 하는 불변량. 데이터도 GPU 도 필요 없다.

배포 규칙은 코드 한 줄이 어긋나도 조용히 깨진다. 새 flow 끼리 엣지가 하나만
생겨도 그 그래프는 전이 학습으로 되돌아가고, 결과는 정상으로 보인다. 그래서
규칙을 문장이 아니라 검사로 적어 둔다.

    python -u test_inductive_attach.py
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

from inductive_eval import attach_new, build

FAIL = []


def check(label: str, ok: bool) -> None:
    print("  %-54s %s" % (label, "ok" if ok else "FAIL"))
    if not ok:
        FAIL.append(label)


def main() -> int:
    print("새 flow 끼리 잇지 않는다")
    v = np.array(["A"] * 12)
    e = attach_new(v, 6, 2, 42)
    new = e >= 6
    check("모든 값이 같아도 새-새 엣지가 0", int((new[0] & new[1]).sum()) == 0)

    print("\n예산을 지킨다")
    # 메시지는 source -> target 으로 간다. 새 노드가 받는 이웃 수는
    # 들어오는 차수다. 나가는 차수로 세면 0 이 나온다.
    indeg = np.bincount(e[1], minlength=12)
    check("새 flow 의 들어오는 차수가 전부 예산과 같다",
          set(indeg[6:].tolist()) == {2})
    check("학습 flow 는 부착으로 아무것도 받지 않는다",
          int(indeg[:6].sum()) == 0)

    print("\n학습에 없는 값은 고립된다")
    v = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "A", "A", "Z", "Z"])
    e = attach_new(v, 6, 2, 42)
    indeg = np.bincount(e[1], minlength=12) if e.shape[1] else np.zeros(12, int)
    check("학습에 없던 값은 차수 0", indeg[6] == 0 and indeg[7] == 0
          and indeg[10] == 0 and indeg[11] == 0)
    check("학습에 있던 값은 차수 > 0", indeg[8] > 0 and indeg[9] > 0)

    print("\n결측은 잇지 않는다")
    v = np.array(["-", "-", "-", "A", "A", "A", "-", "-", "A"])
    e = attach_new(v, 6, 2, 42)
    indeg = np.bincount(e[1], minlength=9) if e.shape[1] else np.zeros(9, int)
    check("결측 토큰을 가진 새 flow 는 차수 0", indeg[6] == 0 and indeg[7] == 0)
    check("유효값을 가진 새 flow 는 차수 > 0", indeg[8] > 0)

    print("\n값 그룹을 넘지 않는다")
    v = np.array(["A"] * 4 + ["B"] * 4 + ["A", "B", "A", "B"])
    e = attach_new(v, 8, 2, 42)
    check("모든 엣지가 같은 값끼리",
          all(v[s] == v[t] for s, t in zip(e[0], e[1])))

    print("\n구성이 결정론이다")
    check("같은 입력이 같은 배열", np.array_equal(attach_new(v, 8, 2, 42),
                                            attach_new(v, 8, 2, 42)))

    print("\n부착 엣지는 한 방향이다")
    check("자기 루프 없음", all(a != b for a, b in zip(e[0], e[1])))
    check("새 flow 가 source 인 엣지가 없다", int((e[0] >= 8).sum()) == 0)
    check("모든 target 이 새 flow", bool((e[1] >= 8).all()))

    print("\n두 층 경로로도 새-새가 이어지지 않는다")
    # 새_i -> 학습_k -> 새_j 경로. 새가 source 가 아니면 구조상 0 이다.
    ins = {}
    outs = {}
    for a, b in zip(e[0].tolist(), e[1].tolist()):
        if a >= 8:
            ins.setdefault(b, []).append(a)
        if b >= 8:
            outs.setdefault(a, []).append(b)
    paths = sum(len(ins.get(k, [])) * len(v) for k, v in outs.items())
    check("2홉 새-새 경로 0", paths == 0)

    print("\n규칙이 깨지면 잡아낸다")
    # build 가 새-새 엣지를 발견하면 종료해야 한다. 일부러 만들어 확인한다.
    meta = pd.DataFrame({"src_ip": ["a"] * 6, "sport": range(6),
                         "dst_ip": ["b"] * 6, "dport": range(6),
                         "ts": range(6), "sni": ["A"] * 6})
    caught = False
    try:
        import inductive_eval as IE
        real = IE.attach_new
        # 새 flow 를 source 로 두는 역방향 엣지. 예전 판이 이렇게 냈다.
        IE.attach_new = lambda *a, **k: np.array([[4, 0], [0, 4]], dtype=np.int64)
        try:
            build(meta, [3, 1, 2], ["via_sni"], 42, 2, 300.0)
        finally:
            IE.attach_new = real
    except SystemExit:
        caught = True
    check("역방향 엣지를 넣으면 종료한다", caught)

    print()
    if FAIL:
        print("FAILED: %s" % FAIL)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
