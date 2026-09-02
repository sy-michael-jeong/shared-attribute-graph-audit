# -*- coding: utf-8 -*-
"""Build Figure 1, the metapath saturation curves.

The horizontal axis counts the relations that actually enter the graph. A
configuration file can list a relation that produces no edges in a given
dataset, and a configuration naming such a relation falls back to the smaller
graph. Aggregation therefore de-duplicates on metapaths_used and counts the
true size.

In BCCC-DoH the configuration lists seven TLS relations, but SNI produces no
edges, so six is the maximum.

The curve is read from the shipped summaries every time. An earlier version
carried a copy of the aggregate in the source, which drifted away from the
summaries it was supposed to reflect: it put BCCC-DoH between 0.968 and 0.977
where the summaries give 0.9998 at one relation and the validation ceiling
from two onward. Reading the files is the only way the figure and the results
directory can be made to agree.

Usage:
    python make_fig1.py --summaries results/saturation --out fig1_saturation.pdf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"bccc_dohbrw": "BCCC-DoH", "iscx_vpn": "ISCX-VPN",
         "hikari": "HIKARI", "vnat": "VNAT"}
STYLE = {"BCCC-DoH": ("o", "-"), "ISCX-VPN": ("s", "--"),
         "HIKARI": ("^", "-."), "VNAT": ("d", ":")}


def from_summaries(root: Path):
    """Best validation macro-F1 per true relation count, from the summaries."""
    out = {}
    for f in sorted(root.glob("*/combinatorial_grand_summary.json")):
        d = json.load(open(f))
        for ds, v in d.items():
            if ds.startswith("_") or ds not in LABEL:
                continue
            best = {}
            for _, items in v.get("stages", {}).items():
                for it in items:
                    used = it.get("metapaths_used") or it.get("metapaths") or []
                    sc = it.get("val_macro_f1", it.get("score"))
                    if not used or sc is None:
                        continue
                    n = len(set(used))
                    if n not in best or sc > best[n]:
                        best[n] = float(sc)
            if best:
                out[LABEL[ds]] = best
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--summaries", default="results/saturation",
                    help="directory holding the combinatorial summaries")
    ap.add_argument("--out", default="fig1_saturation.pdf")
    args = ap.parse_args()

    data = from_summaries(Path(args.summaries))
    if not data:
        raise SystemExit(
            "%s holds no readable summary. The figure is drawn from the "
            "summaries alone, so there is nothing to fall back on."
            % args.summaries)
    print("[from summaries] " + ", ".join(
        "%s %d points" % (k, len(v)) for k, v in data.items()))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for name in ("BCCC-DoH", "ISCX-VPN", "HIKARI", "VNAT"):
        if name not in data:
            continue
        pts = sorted(data[name].items())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        m, ls = STYLE[name]
        ax.plot(xs, ys, marker=m, linestyle=ls, markersize=5.5,
                alpha=0.85, markerfacecolor="none", markeredgewidth=1.2,
                linewidth=1.3, label=name)

    ax.set_xlabel("Number of relations in the graph")
    ax.set_ylabel("Best validation macro-F1")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.45, 1.03)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="center left", fontsize=8, frameon=False, ncol=2,
              bbox_to_anchor=(0.02, 0.55))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out)
    print("[saved] %s" % args.out)


if __name__ == "__main__":
    main()
