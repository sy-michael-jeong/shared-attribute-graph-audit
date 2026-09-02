# -*- coding: utf-8 -*-
"""Invariants the inductive attachment rule must satisfy. Needs no data and no GPU.

A deployment rule breaks silently when one line of code is off: a single
new-new edge turns the graph back into transductive training and the scores
still look normal. So the rule is written as checks, not prose.

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
    print("new flows are never connected to each other")
    v = np.array(["A"] * 12)
    e = attach_new(v, 6, 2, 42)
    new = e >= 6
    check("no new-new edge even when every value is identical", int((new[0] & new[1]).sum()) == 0)

    print("\nbudget is respected")
    # Messages travel source -> target. A new node's neighbour count is its
    # in-degree; counting out-degree would give 0.
    indeg = np.bincount(e[1], minlength=12)
    check("every new flow has in-degree equal to the budget",
          set(indeg[6:].tolist()) == {2})
    check("training flows receive nothing from attachment",
          int(indeg[:6].sum()) == 0)

    print("\nvalues absent from training are isolated")
    v = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "A", "A", "Z", "Z"])
    e = attach_new(v, 6, 2, 42)
    indeg = np.bincount(e[1], minlength=12) if e.shape[1] else np.zeros(12, int)
    check("value unseen in training -> degree 0", indeg[6] == 0 and indeg[7] == 0
          and indeg[10] == 0 and indeg[11] == 0)
    check("value seen in training -> degree > 0", indeg[8] > 0 and indeg[9] > 0)

    print("\nmissing values are never connected")
    v = np.array(["-", "-", "-", "A", "A", "A", "-", "-", "A"])
    e = attach_new(v, 6, 2, 42)
    indeg = np.bincount(e[1], minlength=9) if e.shape[1] else np.zeros(9, int)
    check("new flow with a missing token -> degree 0", indeg[6] == 0 and indeg[7] == 0)
    check("new flow with a valid value -> degree > 0", indeg[8] > 0)

    print("\nedges stay inside value groups")
    v = np.array(["A"] * 4 + ["B"] * 4 + ["A", "B", "A", "B"])
    e = attach_new(v, 8, 2, 42)
    check("every edge joins equal values",
          all(v[s] == v[t] for s, t in zip(e[0], e[1])))

    print("\nconstruction is deterministic")
    check("same input -> same array", np.array_equal(attach_new(v, 8, 2, 42),
                                            attach_new(v, 8, 2, 42)))

    print("\nattachment edges are one-directional")
    check("no self-loops", all(a != b for a, b in zip(e[0], e[1])))
    check("no edge has a new flow as source", int((e[0] >= 8).sum()) == 0)
    check("every target is a new flow", bool((e[1] >= 8).all()))

    print("\nno new-new path through two layers")
    # new_i -> train_k -> new_j paths; structurally 0 when no new flow is a source.
    ins = {}
    outs = {}
    for a, b in zip(e[0].tolist(), e[1].tolist()):
        if a >= 8:
            ins.setdefault(b, []).append(a)
        if b >= 8:
            outs.setdefault(a, []).append(b)
    paths = sum(len(ins.get(k, [])) * len(v) for k, v in outs.items())
    check("2-hop new-new paths == 0", paths == 0)

    print("\nviolations are caught")
    # The build must stop on a new-new edge; create one on purpose to check.
    meta = pd.DataFrame({"src_ip": ["a"] * 6, "sport": range(6),
                         "dst_ip": ["b"] * 6, "dport": range(6),
                         "ts": range(6), "sni": ["A"] * 6})
    caught = False
    try:
        import inductive_eval as IE
        real = IE.attach_new
        # A reversed edge with a new flow as source, as an earlier version produced.
        IE.attach_new = lambda *a, **k: np.array([[4, 0], [0, 4]], dtype=np.int64)
        try:
            build(meta, [3, 1, 2], ["via_sni"], 42, 2, 300.0)
        finally:
            IE.attach_new = real
    except SystemExit:
        caught = True
    check("a reversed edge stops the build", caught)

    print()
    if FAIL:
        print("FAILED: %s" % FAIL)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
