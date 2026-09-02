# -*- coding: utf-8 -*-
"""Shared definitions for the artifact.

This module is the single source of truth. Every other script imports from
here instead of restating the same constants, so that a change to a missing
value token, a relation name or a selected set takes effect everywhere at once.
"""
from __future__ import annotations
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

# ------------------------------------------------------- missing-value tokens
# Zeek writes '-' for an absent field, and the dataset adapters can emit
# 'nan', '(empty)' or 'None'. Treating any of these as a real value would make
# flows that share nothing but an empty field connect to each other, so the
# relation would encode its own missingness pattern as edges.
MISSING_TOKENS = ("", "-", "nan", "None", "(empty)", "0")


def is_valid(v) -> bool:
    return str(v).strip() not in MISSING_TOKENS


# ------------------------------------------------------------ relation table
RELATION_COLUMN = {
    "via_sni": "sni",
    "via_ja3": "ja3",
    "via_cert_subject": "cert_subject",
    "via_alpn": "alpn",
    "via_cert_issuer": "cert_issuer_org",
    "via_tls_cipher_group": "tls_cipher_group",
    "via_cert_validity": "cert_validity_bucket",
    "via_timebin": "ts",
    "via_src_host": "src_ip",
    "via_dst_host": "dst_ip",
}
ALL_RELATIONS = list(RELATION_COLUMN)

# Relation families as the paper defines them (Section 3.2): the TLS family of
# seven relations and the auxiliary family of two host relations plus one
# temporal relation. The family comparison of Section 6.5 runs TLS against the
# two host relations. `via_ja3` is the legacy column name of the paper's
# VersionCipher relation (see README, "Internal naming note").
TLS_RELATIONS = ["via_sni", "via_ja3", "via_cert_subject", "via_alpn",
                 "via_cert_issuer", "via_tls_cipher_group", "via_cert_validity"]
HOST_RELATIONS = ["via_src_host", "via_dst_host"]
TEMPORAL_RELATIONS = ["via_timebin"]
AUX_RELATIONS = HOST_RELATIONS + TEMPORAL_RELATIONS

# HAN reported configuration (Table 4 and every control). The same set is used
# under both split protocols.
# Every entry is an explicit list. CIC-AndMal was written as None here at one
# point, meaning "no subset was selected", and the three scripts that read this
# table resolved that None differently: one to the seven TLS relations, one to
# all ten, one to whatever the graph happened to carry. The reported results
# use the seven TLS relations, so that is what the entry says.
SELECTED = {
    "bccc_dohbrw": ["via_cert_validity"],
    "iscx_vpn": ["via_src_host"],
    "hikari": ["via_dst_host"],
    "vnat": ["via_timebin"],
    "cic_andmal": list(TLS_RELATIONS),
}

# Columns that travel with a flow but are never model inputs. The graph
# builder reads the relation fields from here, and dataset_composition.py
# reads the fine-grained label.
META_COLUMNS = (
    "src_ip", "dst_ip", "sport", "dport", "ts",
    "sni", "ja3", "cert_subject",
    "alpn", "cert_issuer_org", "tls_cipher_group", "cert_validity_bucket",
    "traffic_category", "attack_category", "app",
)

SPLITS = ("train", "val", "test")
TIMEBIN_SECONDS = 300.0


def graph_fingerprint(data_dir) -> Dict:
    """Which graph a run actually saw, recorded next to its scores.

    A result file that names only its data directory does not say what was in
    that directory at the time. Directories get rebuilt; the recorded path
    does not change with them. Edge counts per relation and node counts per
    partition identify the graph itself, so a result can be traced without
    the directory still being around.
    """
    from pathlib import Path
    d = Path(data_dir)
    out: Dict = {}
    for split in SPLITS:
        p = d / ("y_%s.npy" % split)
        if p.exists():
            out["n_" + split] = int(len(np.load(p)))
    out["edges"] = {f.stem.replace("hin_edges_", ""): int(np.load(f).shape[1])
                    for f in sorted(d.glob("hin_edges_*.npy"))}
    return out


def timebin_values(ts_col, bin_seconds: float = TIMEBIN_SECONDS) -> np.ndarray:
    """The same bucket identifier the relation uses, not the raw timestamp."""
    out = np.zeros(len(ts_col), dtype=np.int64)
    for i, t in enumerate(ts_col):
        try:
            v = float(t)
        except Exception:
            try:
                v = pd.Timestamp(t).timestamp()
            except Exception:
                v = 0.0
        out[i] = int(v // bin_seconds)
    return out


def relation_values(allm: pd.DataFrame, rel: str,
                    bin_seconds: float = TIMEBIN_SECONDS):
    """Value series for one relation, or None when the column is absent."""
    col = RELATION_COLUMN[rel]
    if col not in allm.columns:
        return None
    raw = allm[col].values
    if rel == "via_timebin":
        return pd.Series(timebin_values(raw, bin_seconds).astype(str))
    return pd.Series([str(v) if v is not None else "" for v in raw])


# ------------------------------------------------------------------- metrics
def macro_f1(y_true, y_pred, n_classes: int = 2) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def minority_f1(y_true, y_pred) -> float:
    u, c = np.unique(y_true, return_counts=True)
    m = int(u[np.argmin(c)])
    tp = int(((y_pred == m) & (y_true == m)).sum())
    fp = int(((y_pred == m) & (y_true != m)).sum())
    fn = int(((y_pred != m) & (y_true == m)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "minority_f1": minority_f1(np.asarray(y_true), np.asarray(y_pred))}


# ------------------------------------------------------------- reproducibility
def seed_all(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> str:
    if name in ("cpu", "cuda"):
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------- validation
def check_relations(data_dir, dataset: str, expect_flows=None) -> List[str]:
    """Check that the flow count matches and that edges were actually built.

    A model script can exit with status zero and still leave a failed result
    behind, so the contents are verified rather than the exit code.
    """
    import json
    from pathlib import Path
    d = Path(data_dir) / dataset
    n = sum(len(np.load(d / ("y_%s.npy" % s))) for s in SPLITS)
    if expect_flows is not None and n != expect_flows:
        raise RuntimeError("%s: %d flows, expected %d. The adapter read a "
                           "different source." % (dataset, n, expect_flows))
    rels = sorted(f.stem.replace("hin_edges_", "")
                  for f in d.glob("hin_edges_*.npy"))
    if not rels:
        raise RuntimeError("%s: no edge files." % dataset)
    for r in rels:
        e = np.load(d / ("hin_edges_%s.npy" % r))
        col = RELATION_COLUMN.get(r)
        if col is None or r == "via_timebin":
            continue
        allm = pd.concat([pd.read_csv(d / ("meta_%s.csv" % s), low_memory=False)
                          for s in SPLITS], ignore_index=True)
        if col not in allm.columns:
            continue
        covered = int(allm[col].astype(str).apply(is_valid).sum())
        inc = int((np.bincount(np.concatenate([e[0], e[1]]), minlength=n) > 0).sum())
        if covered and inc > covered:
            raise RuntimeError(
                "%s/%s: %d incident nodes exceed %d covered flows, so missing "
                "values are producing edges." % (dataset, r, inc, covered))
    return rels
