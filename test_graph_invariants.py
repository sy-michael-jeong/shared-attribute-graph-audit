# -*- coding: utf-8 -*-
"""Invariants the metapath graph builder has to hold.

A relation edge states that two flows share a value. That statement is
symmetric, carries no multiplicity, and does not depend on where the two flows
happen to sit in the table. Each check below turns one of those properties
into a test, so a change to the builder that breaks one of them fails here
instead of moving a number in a results table.

Usage:
    python test_graph_invariants.py
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

from build_graph import edges_for_values, row_keys

DEGREES = (2, 4, 6)
GROUP_SIZES = (147, 131, 103, 35, 12, 3)


def sample_table():
    """A table whose values repeat at very different rates."""
    values = []
    for i, n in enumerate(GROUP_SIZES):
        values += ["v%d" % i] * n
    n = len(values)
    return pd.DataFrame({
        "src_ip": ["10.0.0.%d" % (i % 250) for i in range(n)],
        "sport": [1024 + i for i in range(n)],
        "dst_ip": ["93.184.%d.%d" % (i // 250, i % 250) for i in range(n)],
        "dport": [443] * n,
        "ts": [1429042300.0 + i * 0.37 for i in range(n)],
        "value": values,
    })


def edge_set(e):
    return set(zip(e[0].tolist(), e[1].tolist()))


def check(name, ok):
    print("  %-46s %s" % (name, "ok" if ok else "FAILED"))
    return ok


def main():
    meta = sample_table()
    vals = meta["value"].values
    keys = row_keys(meta)
    passed = True

    for degree in DEGREES:
        print("degree = %d" % degree)
        e = edges_for_values(vals, 42, degree, keys)
        s = edge_set(e)

        passed &= check("no duplicate edge", e.shape[1] == len(s))
        passed &= check("no self loop", not any(a == b for a, b in s))
        passed &= check("every edge has its reverse",
                        all((b, a) in s for a, b in s))

        deg = {}
        for a, _ in s:
            deg[a] = deg.get(a, 0) + 1
        off, uniform = 0, True
        for n in GROUP_SIZES:
            d = [deg.get(j, 0) for j in range(off, off + n)]
            off += n
            uniform &= (min(d) == max(d) == min(degree, n - 1))
        passed &= check("degree is the budget at every group size", uniform)

        again = edges_for_values(vals, 42, degree, keys)
        passed &= check("same input gives the same array",
                        again.shape == e.shape and bool((again == e).all()))

        perm = np.random.RandomState(7).permutation(len(meta))
        shuffled = meta.iloc[perm].reset_index(drop=True)
        e2 = edges_for_values(shuffled["value"].values, 42, degree,
                              row_keys(shuffled))
        back = {(perm[a], perm[b]) for a, b in zip(*e2)}
        passed &= check("row order does not change the graph", back == s)

        other = edges_for_values(vals, 43, degree, keys)
        passed &= check("a different seed gives a different graph",
                        edge_set(other) != s)
        print()

    print("odd and sub-minimal budgets")
    for bad in (1, 3, 5, 0, -2):
        try:
            edges_for_values(vals, 42, bad, keys)
            passed &= check("degree %d is refused" % bad, False)
        except ValueError:
            passed &= check("degree %d is refused" % bad, True)

    print()
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
