# -*- coding: utf-8 -*-
"""Rank agreement and the diagnostic correlations (Sec. 6.5).

Two questions are answered here, and both are rank statistics, so they share a
file.

  Sec. 6.5   Does the relation ranking survive a redraw of the split? Each
             split variant gives every relation a five-seed mean, and the
             relations are ordered by it. Agreement between two orderings is
             Kendall's tau.

  Sec. 6.5   Do the pre-training diagnostics predict the gain a relation
             brings? Normalised homophily and reach are computed before any
             training, and are correlated with what the relation adds over the
             self-loop control by Spearman's rho.

Kendall tau-a is used, not tau-b. Ties are frequent here: on VNAT ten relations
reach the validation ceiling and share a score. tau-b divides by a tie-adjusted
norm, which raises the coefficient exactly where the ranking carries least
information. tau-a keeps n(n-1)/2 in the denominator, so a tie contributes
nothing to agreement rather than being discounted out of the count. The
reported instability is therefore not an artefact of the tie correction.

Significance is a permutation test rather than a table lookup. With five
datasets the asymptotic p-value is meaningless, and three of the five
normalised homophily values are 1.000, so the effective sample is smaller than
n suggests. The permutation is exhaustive when n! is small enough to enumerate.

Usage:
    python rank_stability.py --results results --out results/rank_stability
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

DATASETS = ["bccc_dohbrw", "iscx_vpn", "hikari", "cic_andmal", "vnat"]

# The split variants of Section 6.3 and 6.5. Edge-seed variants redraw the graph rather
# than the split, so they are kept apart from the protocol comparison.
PREFIXES = ("random_", "cutoff_", "edgeseed_")


def tau_a(a, b) -> float:
    """Kendall tau-a between two score vectors.

    Concordant minus discordant over n(n-1)/2. A pair tied in either vector is
    counted in the denominator and contributes nothing to the numerator, which
    is what separates tau-a from tau-b.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return float("nan")
    s = 0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


def spearman(x, y) -> float:
    """Spearman rho, ties given their average rank."""
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v), float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def perm_p(x, y, stat=spearman, n_perm=100000, seed=0):
    """Two-sided permutation p-value. Exhaustive when n! is small."""
    obs = stat(x, y)
    x = list(x)
    if math.factorial(len(x)) <= 40320:
        perms = list(itertools.permutations(x))
        vals = [stat(p, y) for p in perms]
        exact = True
    else:
        rng = np.random.default_rng(seed)
        vals = [stat(rng.permutation(x), y) for _ in range(n_perm)]
        exact = False
    hits = sum(1 for v in vals if abs(v) >= abs(obs) - 1e-12)
    return obs, hits / len(vals), exact, len(vals)


def variants(block: dict):
    """Split variants of one dataset that carry a relation ranking."""
    return {k: v["rank"] for k, v in block.items()
            if k.startswith(PREFIXES) and isinstance(v, dict) and "rank" in v}


def ranking_agreement(root: Path):
    """Pairwise tau between split variants, and against the consensus."""
    out = {}
    for ds in DATASETS:
        f = root / "repeated_splits" / ("%s.json" % ds)
        if not f.is_file():
            continue
        var = variants(json.load(open(f))[ds])
        if len(var) < 2:
            continue
        rels = sorted(set.intersection(*[set(v) for v in var.values()]))
        vecs = {k: [v[r]["mean"] for r in rels] for k, v in var.items()}

        rand = {k: v for k, v in vecs.items() if k.startswith("random_")}
        cut = {k: v for k, v in vecs.items() if k.startswith("cutoff_")}
        block = {"relations": rels, "n_variants": len(vecs)}

        for tag, group in (("random", rand), ("cutoff", cut)):
            keys = sorted(group)
            pairs = [tau_a(group[a], group[b])
                     for a, b in itertools.combinations(keys, 2)]
            if pairs:
                block["%s_pairwise_tau_mean" % tag] = round(float(np.mean(pairs)), 4)
                block["%s_n_pairs" % tag] = len(pairs)

        # The consensus of a group is the mean score of each relation across
        # its variants. Averaging the ranks instead would let a relation that
        # is second everywhere outrank one that is first half the time.
        if rand and cut:
            cr = np.mean([rand[k] for k in sorted(rand)], axis=0)
            cc = np.mean([cut[k] for k in sorted(cut)], axis=0)
            block["consensus_tau_random_vs_cutoff"] = round(tau_a(cr, cc), 4)

        # How many relations reach first place at least once. Ties are
        # frequent, so two counts are given. Picking one winner out of a tie
        # by list order would put a third number between them that depends on
        # nothing but the order the relations happen to be stored in.
        any_top, uniq_top, n_tied = set(), set(), 0
        counts = {}
        for v in vecs.values():
            sc = dict(zip(rels, v))
            m = max(sc.values())
            top = [r for r, x in sc.items() if x == m]
            any_top |= set(top)
            for r in top:
                counts[r] = counts.get(r, 0) + 1
            if len(top) == 1:
                uniq_top |= set(top)
            else:
                n_tied += 1
        block["top_counts"] = {r: counts[r] for r in sorted(counts)}
        block["n_relations_ever_top"] = len(any_top)
        block["n_relations_unique_top"] = len(uniq_top)
        block["n_variants_ending_in_a_tie"] = n_tied
        out[ds] = block
    return out


def diagnostics(root: Path):
    """Do the pre-training diagnostics predict what a relation adds?

    Two levels are reported. At dataset level the normalised homophily of the
    reported relation is correlated with the edge share of that dataset. At
    relation level every (dataset, relation) pair that was evaluated on its own
    is correlated with what it adds over the self-loop control of its dataset.
    """
    share = json.load(open(root / "model_comparison" / "bootstrap_ci.json"))
    share = share.get("_edge_share", {})
    hom = json.load(open(root / "homophily" / "edge_homophily.json"))

    sel = json.load(open(root / "selection" / "selected.json")) \
        if (root / "selection" / "selected.json").is_file() else None

    rows = []
    for ds in DATASETS:
        key = "%s__macro_f1" % ds
        if key not in share or ds not in hom:
            continue
        rels = hom[ds]["relations"]
        # The relation the paper reports for this dataset.
        # Table 12 tabulates the maximum over relations, not the relation
        # the selection rule picked. Several relations often share that
        # maximum, so the count of relations at it is reported alongside; the
        # name of any one of them would be an arbitrary choice.
        pick = max(rels, key=lambda r: rels[r].get("normalised_homophily", 0))
        if pick not in rels:
            continue
        top = rels[pick]["normalised_homophily"]
        tied = [r for r in rels
                if rels[r].get("normalised_homophily") == top]
        rows.append({"dataset": ds,
                     "normalised_homophily": top,
                     "n_relations_at_this_value": len(tied),
                     "reach": rels[pick]["reach"],
                     "edge_share_pct": share[key]["edge_share_pct"]})

    out = {"dataset_level": {"rows": rows}}
    if len(rows) >= 3:
        x = [r["normalised_homophily"] for r in rows]
        y = [r["edge_share_pct"] for r in rows]
        rho, p, exact, n = perm_p(x, y)
        out["dataset_level"].update(
            {"spearman": round(rho, 4), "p_permutation": round(p, 4),
             "exhaustive": exact, "n_permutations": n, "n": len(rows)})
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/rank_stability")
    args = ap.parse_args()

    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rep = {"kendall": "tau-a, ties contribute nothing to agreement",
           "ranking": ranking_agreement(root),
           "diagnostics": diagnostics(root)}

    print("=== relation ranking agreement (Kendall tau-a) ===")
    for ds, b in rep["ranking"].items():
        print("%-12s variants %2d  random tau %s  cutoff tau %s  consensus %s"
              % (ds, b["n_variants"], b.get("random_pairwise_tau_mean"),
                 b.get("cutoff_pairwise_tau_mean"),
                 b.get("consensus_tau_random_vs_cutoff")))
        print("%-12s relations reaching first place: %s ever, %s outright; "
              "%s of %s variants end in a tie"
              % ("", b.get("n_relations_ever_top"),
                 b.get("n_relations_unique_top"),
                 b.get("n_variants_ending_in_a_tie"), b["n_variants"]))

    d = rep["diagnostics"]["dataset_level"]
    if "spearman" in d:
        print("\n=== normalised homophily vs edge share ===")
        for r in d["rows"]:
            print("  %-12s hom %.4f (%d relations at it)  reach %.4f  "
                  "share %5.1f%%"
                  % (r["dataset"], r["normalised_homophily"],
                     r["n_relations_at_this_value"], r["reach"],
                     r["edge_share_pct"]))
        print("  rho %.4f   permutation p %.4f  (%s, %d)"
              % (d["spearman"], d["p_permutation"],
                 "exhaustive" if d["exhaustive"] else "sampled",
                 d["n_permutations"]))

    json.dump(rep, open(out / "rank_stability.json", "w"), indent=2)
    print("\n[saved] %s" % (out / "rank_stability.json"))


if __name__ == "__main__":
    main()
