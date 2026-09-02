# -*- coding: utf-8 -*-
"""Can CTU-13 support a TLS relation graph at all? (Sec. 3.3)

A relation exists only where the field is populated on both sides of the label.
A dataset whose attack traffic carries no TLS handshake therefore cannot be used
for this construction, and that can be decided before any model is trained.

CTU-13 was a sixth candidate for this study. Each scenario ships a Zeek
`ssl.log` alongside the capture, and each scenario README lists the infected and
the normal addresses, which is how the dataset authors assign labels. Reading
those two files answers the question directly, with no pcap parsing.

The answer is that three of the five scenarios contain no botnet TLS session at
all. Of the two that do, one carries two server names that belong to a benign
client running on the infected host, and the other carries none. So the attack
class has no TLS metadata to build a relation from.

A CTU-13 capture holds three kinds of traffic, not two. The README labels a
small set of addresses infected and another small set normal; everything else is
the rest of the university network and the authors label it background. This
script counts all three separately. Calling every address that is not infected
"normal" would be reporting the background of a university router as the benign
class, and the benign counts would be an order of magnitude larger than the
dataset defines them to be.

The scenario logs are downloaded on demand. Pass `--logs` to point at a local
copy instead, laid out as `<dir>/Botnet-42/ssl.log`.

    python ctu13_coverage.py \
        --out results/dataset_suitability/ctu13_tls_coverage.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://mcfp.felk.cvut.cz/publicDatasets"

# The same six addresses are listed as normal in all five scenario READMEs.
NORMAL = ["147.32.84.170",   # Normal-V42-Stribrek
          "147.32.84.134",   # Normal-V42-Jist
          "147.32.84.164",   # Normal-V42-Grill
          "147.32.87.36",    # CVUT-WebServer
          "147.32.80.9",     # CVUT-DNS-Server
          "147.32.87.11"]    # MatLab-Server

# Infected addresses, taken from the `IP Addresses` block of each scenario
# README. Four scenarios infect one virtual machine; scenario 9 infects ten.
SCENARIOS = {
    "Botnet-42": {"dir": "CTU-Malware-Capture-Botnet-42",
                  "ctu13_scenario": 1,
                  "malware": "Neris",
                  "infected": ["147.32.84.165"]},
    "Botnet-44": {"dir": "CTU-Malware-Capture-Botnet-44",
                  "ctu13_scenario": 3,
                  "malware": "Rbot",
                  "infected": ["147.32.84.165"]},
    "Botnet-45": {"dir": "CTU-Malware-Capture-Botnet-45",
                  "ctu13_scenario": 4,
                  "malware": "Rbot",
                  "infected": ["147.32.84.165"]},
    "Botnet-49": {"dir": "CTU-Malware-Capture-Botnet-49",
                  "ctu13_scenario": 8,
                  "malware": "Murlo",
                  "infected": ["147.32.84.165"]},
    "Botnet-50": {"dir": "CTU-Malware-Capture-Botnet-50",
                  "ctu13_scenario": 9,
                  "malware": "Neris",
                  "infected": ["147.32.84.165", "147.32.84.191",
                               "147.32.84.192", "147.32.84.193",
                               "147.32.84.204", "147.32.84.205",
                               "147.32.84.206", "147.32.84.207",
                               "147.32.84.208", "147.32.84.209"]},
}


def read_ssl_log(text: str):
    """Zeek's tab-separated log. The header names the columns."""
    fields, rows = None, []
    for line in text.splitlines():
        if line.startswith("#fields"):
            fields = line.split("\t")[1:]
            continue
        if line.startswith("#") or not line.strip():
            continue
        if fields is None:
            continue
        parts = line.split("\t")
        if len(parts) < len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def fetch(name: str, spec: dict, logs: Path | None) -> str:
    if logs is not None:
        p = logs / name / "ssl.log"
        return p.read_text(encoding="utf-8", errors="replace")
    url = "%s/%s/bro/ssl.log" % (BASE, spec["dir"])
    print("  fetching %s" % url)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def survey(delay: float = 0.2):
    """Which other captures on the same server carry a TLS log at all.

    CTU-13 is not the only public capture collection from this lab. The rest of
    the Malware Capture Facility is single-malware machines: one infected host
    per capture, with no normal host beside it. Pairing such a capture with a
    separate normal capture would make the capture source itself predict the
    label, which is the failure this paper measures. So the size of that
    collection is worth recording, and so is the fact that none of it places
    infected and normal hosts on one network.

    One request for the index, then one per directory. Deliberately slow.

    The index gives sizes rounded to a unit, and that unit is binary: what it
    prints as `138M` is 138 MiB, not 138 MB. The rounded figure is only used to
    rank, and the size that is reported comes from a `Content-Length` header on
    the one file that ranks first, in bytes, so it carries no unit to confuse.
    """
    print("  surveying %s" % BASE)
    with urllib.request.urlopen(BASE + "/", timeout=120) as r:
        index = r.read().decode("utf-8", errors="replace")
    dirs = sorted(set(re.findall(r'href="([^"/?]+)/"', index)))
    dirs = [d for d in dirs if not d.startswith(".")]

    with_ssl, largest, largest_rank = [], None, 0.0
    for i, d in enumerate(dirs):
        url = "%s/%s/bro/" % (BASE, d)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                page = r.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001  a directory with no bro/ folder
            time.sleep(delay)
            continue
        if 'href="ssl.log"' in page:
            with_ssl.append(d)
            m = re.search(r'href="ssl\.log"[^\n]*?(\d+(?:\.\d+)?)([KMG])\b',
                          page)
            if m:
                kib = float(m.group(1)) * {"K": 1.0, "M": 1024.0,
                                           "G": 1048576.0}[m.group(2)]
                if kib > largest_rank:
                    largest_rank, largest = kib, d
        if (i + 1) % 50 == 0:
            print("    %d/%d  ssl.log so far %d" % (i + 1, len(dirs),
                                                    len(with_ssl)))
        time.sleep(delay)

    largest_bytes = None
    if largest is not None:
        req = urllib.request.Request("%s/%s/bro/ssl.log" % (BASE, largest),
                                     method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
        largest_bytes = int(n) if n else None

    return {"directories_listed": len(dirs),
            "with_ssl_log": len(with_ssl),
            "largest": largest,
            "largest_ssl_log_bytes": largest_bytes,
            "note": "The large captures are single-malware machines. Pairing "
                    "them with a separate normal capture would make the "
                    "capture source itself predict the label, which is the "
                    "failure this paper measures. CTU-13 is the only "
                    "collection that places infected and normal hosts on one "
                    "network."}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=None,
                    help="local copy laid out as <dir>/Botnet-42/ssl.log")
    ap.add_argument("--survey", action="store_true",
                    help="also count how many other captures on the same "
                         "server carry a TLS log (one request per directory, "
                         "a few minutes)")
    args = ap.parse_args()
    logs = Path(args.logs) if args.logs else None

    report = {
        "purpose": "Whether a public benchmark can support a TLS relation "
                   "graph at all. A relation exists only where the field is "
                   "populated on both sides of the label, so a dataset whose "
                   "attack traffic carries no TLS handshake cannot be used for "
                   "this construction.",
        "method": "Zeek ssl.log published with each CTU-13 scenario was read "
                  "directly. Rows were assigned to the botnet, the normal or "
                  "the background class by matching id.orig_h against the "
                  "infected and normal addresses listed in the scenario "
                  "README, which is how the dataset authors assign labels. An "
                  "address in neither list is background, as it is in the "
                  "published netflow labels. No pcap parsing was needed.",
        "source": "%s/CTU-Malware-Capture-Botnet-{42,44,45,49,50}/bro/ssl.log"
                  % BASE,
        "label_source": "%s/CTU-Malware-Capture-Botnet-<N>/README.html"
                        % BASE,
        "normal_hosts": NORMAL,
        "measured": date.today().isoformat(),
        "scenarios": {},
    }

    n_bot_total = 0
    zero = []
    for name, spec in SCENARIOS.items():
        rows = read_ssl_log(fetch(name, spec, logs))
        infected, normal_hosts = set(spec["infected"]), set(NORMAL)
        bot = [r for r in rows if r.get("id.orig_h") in infected]
        normal = [r for r in rows if r.get("id.orig_h") in normal_hosts]
        background = [r for r in rows if r.get("id.orig_h") not in infected
                      and r.get("id.orig_h") not in normal_hosts]

        def snis(rs):
            return {r.get("server_name", "") for r in rs
                    if r.get("server_name", "") not in ("", "-")}

        block = {"malware": spec["malware"],
                 "ctu13_scenario": spec["ctu13_scenario"],
                 "infected_hosts": spec["infected"],
                 "tls_sessions_total": len(rows),
                 "botnet": len(bot),
                 "botnet_distinct_sni": len(snis(bot)),
                 # Listed, not summarised. Where the count is small the values
                 # themselves are the evidence: on scenario 1 they are an AOL
                 # client's, which is why two names are not two bot names.
                 "botnet_sni_values": sorted(snis(bot)),
                 "normal": len(normal),
                 "normal_distinct_sni": len(snis(normal)),
                 "background": len(background)}
        report["scenarios"][name] = block
        n_bot_total += len(bot)
        if not bot:
            zero.append(name)
        print("  %-11s TLS %7d   botnet %5d (%d SNI)   normal %6d (%d SNI)"
              "   background %7d"
              % (name, len(rows), len(bot), len(snis(bot)), len(normal),
                 len(snis(normal)), len(background)))

    # A relation needs a value, not a session. So the finding is not that the
    # botnet TLS count is small but that the distinct server names it yields
    # is zero or near it, and the sentence says both.
    nonzero = [n for n in SCENARIOS if n not in zero]
    report["finding"] = (
        "%d of the %d scenarios contain no botnet TLS session. The %d that "
        "do yield %s distinct server names between them. The attack class "
        "therefore has no TLS metadata to build a relation from, and CTU-13 "
        "cannot support a TLS relation graph."
        % (len(zero), len(SCENARIOS), len(nonzero),
           " and ".join(str(report["scenarios"][n]["botnet_distinct_sni"])
                        for n in nonzero)))
    report["botnet_tls_sessions_total"] = n_bot_total
    report["scenarios_without_any_botnet_tls"] = zero

    if args.survey:
        report["mcfp_survey"] = survey()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print("\n%s" % report["finding"])
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
