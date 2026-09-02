# Dataset loading and split construction for the five datasets used in the paper.
import hashlib
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import META_COLUMNS, is_valid


SPLIT_MODES = ("random", "time", "time_stratified",
               "sni_disjoint", "ja3_disjoint", "dst_ip_disjoint")

HIKARI_URL = "https://zenodo.org/records/5199540/files/ALLFLOWMETER_HIKARI2021.csv.zip?download=1"
HIKARI_MD5 = "d7d9e277fe4a66cb00764d7f91a810dd"


def split_random(N, y, test_size, val_size, seed):
    classes, counts = np.unique(y, return_counts=True)
    strat = y if counts.min() >= 2 else None
    idx_all = np.arange(N)
    idx_tr, idx_te = train_test_split(idx_all, test_size=test_size,
                                      random_state=seed, stratify=strat)
    y_tr = y[idx_tr]
    strat2 = y_tr if np.unique(y_tr, return_counts=True)[1].min() >= 2 else None
    idx_tr, idx_va = train_test_split(idx_tr, test_size=val_size / (1 - test_size),
                                      random_state=seed, stratify=strat2)
    return idx_tr, idx_va, idx_te


def split_time(ts, test_size, val_size):
    ts_num = pd.to_numeric(pd.Series(ts), errors="coerce").fillna(0).values
    order = np.argsort(ts_num, kind="stable")
    N = len(order)
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_tr = N - n_test - n_val
    return order[:n_tr], order[n_tr:n_tr + n_val], order[n_tr + n_val:]


def split_time_stratified(ts, y, test_size, val_size):
    # per-class ordered split: sort each class by ts (record order when ts is
    # constant), take the first 70% for training, the next 10% for validation,
    # and the last 20% for test, then merge.
    ts_num = pd.to_numeric(pd.Series(ts), errors="coerce").fillna(0).values
    parts_tr, parts_va, parts_te = [], [], []
    for c in np.unique(y):
        rows = np.where(y == c)[0]
        order = rows[np.argsort(ts_num[rows], kind="stable")]
        n = len(order)
        if n < 3:
            print("  [warn] class %s has only %d samples, all placed in train" % (c, n))
            parts_tr.append(order)
            continue
        n_te = max(1, int(round(n * test_size)))
        n_va = max(1, int(round(n * val_size)))
        n_tr = n - n_te - n_va
        parts_tr.append(order[:n_tr])
        parts_va.append(order[n_tr:n_tr + n_va])
        parts_te.append(order[n_tr + n_va:])
    cat = lambda p: np.sort(np.concatenate(p)) if p else np.array([], dtype=int)
    return cat(parts_tr), cat(parts_va), cat(parts_te)


def split_group_disjoint(groups, y, test_size, val_size, seed):
    g = pd.Series(groups).astype(str).fillna("").values
    g = np.where([not is_valid(v) for v in g], "", g)
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    if len([u for u in uniq if u != ""]) < 2:
        warnings.warn("group column has < 2 non-empty values, falling back to random split")
        return split_random(len(g), y, test_size, val_size, seed)
    rng.shuffle(uniq)

    N = len(g)
    sizes = {u: int((g == u).sum()) for u in uniq}
    target_te, target_va = int(N * test_size), int(N * val_size)
    te_g, va_g = [], []
    n_te = n_va = 0
    for u in uniq:
        s = sizes[u]
        if n_te + s <= target_te or not te_g:
            te_g.append(u); n_te += s
        elif n_va + s <= target_va or not va_g:
            va_g.append(u); n_va += s
    te_set, va_set = set(te_g), set(va_g)
    idx_te = np.where(np.isin(g, list(te_set)))[0]
    idx_va = np.where(np.isin(g, list(va_set)))[0]
    idx_tr = np.where(~np.isin(g, list(te_set | va_set)))[0]
    return idx_tr, idx_va, idx_te


def report_split(name, split_mode, meta, idx_tr, idx_va, idx_te):
    print("  [split] %s mode=%s tr=%d va=%d te=%d"
          % (name, split_mode, len(idx_tr), len(idx_va), len(idx_te)))
    if meta is None:
        return
    for col in ("sni", "ja3", "dst_ip"):
        if col not in meta.columns:
            continue
        tr = set(meta.iloc[idx_tr][col].astype(str).values)
        te = set(meta.iloc[idx_te][col].astype(str).values)
        union = len(tr | te)
        ratio = len(tr & te) / union if union else 0.0
        print("    [overlap] %s tr&te=%d / union=%d (%.4f)"
              % (col, len(tr & te), union, ratio))


def finalize(X, y, feature_names, label_map, name,
             test_size, val_size, seed, meta, split_mode):
    if split_mode not in SPLIT_MODES:
        raise ValueError("unknown split_mode %r" % split_mode)
    if split_mode == "random":
        idx = split_random(len(X), y, test_size, val_size, seed)
    elif split_mode == "time":
        idx = split_time(meta["ts"].values, test_size, val_size)
    elif split_mode == "time_stratified":
        idx = split_time_stratified(meta["ts"].values, y, test_size, val_size)
    else:
        col = {"sni_disjoint": "sni", "ja3_disjoint": "ja3",
               "dst_ip_disjoint": "dst_ip"}[split_mode]
        idx = split_group_disjoint(meta[col].values, y, test_size, val_size, seed)
    idx_tr, idx_va, idx_te = idx
    report_split(name, split_mode, meta, idx_tr, idx_va, idx_te)

    scaler = StandardScaler()
    out = {"name": name, "feature_names": list(feature_names),
           "label_map": label_map, "scaler": scaler}
    out["X_train"] = scaler.fit_transform(X[idx_tr]).astype(np.float32)
    out["X_val"] = scaler.transform(X[idx_va]).astype(np.float32)
    out["X_test"] = scaler.transform(X[idx_te]).astype(np.float32)
    out["y_train"] = y[idx_tr].astype(np.int64)
    out["y_val"] = y[idx_va].astype(np.int64)
    out["y_test"] = y[idx_te].astype(np.int64)
    out["meta_train"] = meta.iloc[idx_tr].reset_index(drop=True)
    out["meta_val"] = meta.iloc[idx_va].reset_index(drop=True)
    out["meta_test"] = meta.iloc[idx_te].reset_index(drop=True)
    return out


def frame_to_xy(df, drop_cols):
    # split a raw flow frame into (meta, X): META_COLUMNS go to meta,
    # everything numeric except drop_cols goes to X
    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    meta = df[meta_cols].copy().reset_index(drop=True) if meta_cols else pd.DataFrame(index=df.index)
    for c in META_COLUMNS:
        if c not in meta.columns:
            meta[c] = 0 if c in ("sport", "dport", "ts") else ""
    meta = meta.reset_index(drop=True)

    X_df = df.drop(columns=list(set(meta_cols + drop_cols) & set(df.columns)), errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return meta, X_df.values.astype(np.float32), X_df.columns.tolist()


def extract_pcaps(raw_dir, pcap_glob, label_fn):
    from extract_pcap import extract_flow_features
    pcaps = sorted(raw_dir.glob(pcap_glob))
    if not pcaps:
        raise FileNotFoundError("no pcap under %s/%s" % (raw_dir, pcap_glob))
    print("  found %d pcap files" % len(pcaps))
    frames = []
    for p in pcaps:
        try:
            label = label_fn(p.relative_to(raw_dir).parts, p.name)
        except ValueError:
            label = "Unknown"
        try:
            d = extract_flow_features(p)
        except Exception as e:
            print("  [warn] %s: %s" % (p.name, e))
            continue
        if d.empty:
            continue
        d["_label"] = label
        frames.append(d)
    if not frames:
        raise RuntimeError("no pcap produced any TLS flow")
    df = pd.concat(frames, ignore_index=True, sort=False)
    print("  %d TLS flows extracted" % len(df))
    return df


def load_bccc(raw_dir, task="binary", test_size=0.2, val_size=0.1,
              seed=42, split_mode="random"):
    # CIRA-CIC-DoHBrw-2020, extracted from the original UNB pcaps
    def label_fn(parts, name):
        joined = "/".join(p.lower() for p in parts)
        if any(t in joined for t in ("malicious", "dns2tcp", "dnscat2", "iodine")):
            return "Malicious-DoH"
        if "benign" in joined or "non-doh" in joined or "nondoh" in joined:
            return "Benign-DoH"
        return "Unknown"

    df = extract_pcaps(Path(raw_dir), "**/*.pcap", label_fn)
    if task == "binary":
        y = (df["_label"] == "Malicious-DoH").astype(int).values
        label_map = {"benign_or_nondoh": 0, "malicious_doh": 1}
    else:
        cats = sorted(df["_label"].unique())
        label_map = {c: i for i, c in enumerate(cats)}
        y = df["_label"].map(label_map).values
    meta, X, names = frame_to_xy(df, ["_label"])
    return finalize(X, y, names, label_map, "bccc_dohbrw",
                    test_size, val_size, seed, meta, split_mode)


def load_iscx(raw_dir, task="binary", test_size=0.2, val_size=0.1,
              seed=42, split_mode="random"):
    # ISCX-VPN2016, extracted from pcaps; category comes from the file name
    cats = ["vpn_chat", "vpn_email", "vpn_file", "vpn_streaming", "vpn_voip", "vpn_p2p"]

    def label_fn(parts, name):
        n = name.lower()
        for c in cats:
            if c.split("_", 1)[1] in n:
                return c
        return "vpn_other"

    df = extract_pcaps(Path(raw_dir), "pcaps/**/*.pcap", label_fn)
    if task == "binary":
        y = df["_label"].isin({"vpn_voip", "vpn_streaming"}).astype(int).values
        label_map = {"bulk": 0, "realtime": 1}
    else:
        u = sorted(df["_label"].unique())
        label_map = {c: i for i, c in enumerate(u)}
        y = df["_label"].map(label_map).values
    meta, X, names = frame_to_xy(df, ["_label"])
    return finalize(X, y, names, label_map, "iscx_vpn",
                    test_size, val_size, seed, meta, split_mode)


def load_hikari(raw_dir, task="binary", test_size=0.2, val_size=0.1,
                seed=42, split_mode="random"):
    # HIKARI-2021, released as a single CSV (no pcap, no timestamps)
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "ALLFLOWMETER_HIKARI2021.csv"
    if not csv_path.exists():
        download_hikari(raw_dir)
        zp = raw_dir / "ALLFLOWMETER_HIKARI2021.csv.zip"
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(raw_dir)
    df = pd.read_csv(csv_path)

    if task == "binary":
        y = df["Label"].astype(int).values
        label_map = {"benign_or_background": 0, "attack": 1}
    else:
        u = sorted(df["traffic_category"].astype(str).unique())
        label_map = {c: i for i, c in enumerate(u)}
        y = df["traffic_category"].astype(str).map(label_map).values

    df = df.rename(columns={"originh": "src_ip", "responh": "dst_ip"})
    # drop every index-like column: 'Unnamed: 0.1' in the released CSV
    # correlates with the capture schedule and inflates every model (paper 5.4)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    meta, X, names = frame_to_xy(df, unnamed + ["uid", "flow_id", "Label", "traffic_category"])
    return finalize(X, y, names, label_map, "hikari",
                    test_size, val_size, seed, meta, split_mode)


def load_cic(raw_dir, task="binary", test_size=0.2, val_size=0.1,
             seed=42, split_mode="random", max_rows=500000):
    # CIC-AndMal2017, extracted from pcaps; label = top-level folder name
    def label_fn(parts, name):
        parts = [p for p in parts]
        while parts and parts[0].lower() in ("pcap", "pcaps"):
            parts = parts[1:]
        return parts[0] if len(parts) >= 2 else "Unknown"

    df = extract_pcaps(Path(raw_dir), "pcap/**/*.pcap", label_fn)
    lab = df["_label"].astype(str).str.strip().str.lower()
    if task == "binary":
        keep = lab != ""
        df, lab = df[keep.values].reset_index(drop=True), lab[keep]
        y = (~lab.isin({"benign", "normal", "0", "good"})).astype(int).values
        label_map = {"benign": 0, "attack": 1}
    else:
        coarse = {"adware": "Adware", "ransomware": "Ransomware",
                  "scareware": "Scareware", "smsmalware": "SMSMalware",
                  "sms": "SMSMalware", "benign": "Benign", "normal": "Benign"}
        norm = lab.map(lambda s: next((v for k, v in coarse.items() if s.startswith(k)), ""))
        keep = norm != ""
        df, norm = df[keep.values].reset_index(drop=True), norm[keep]
        u = sorted(norm.unique())
        label_map = {c: i for i, c in enumerate(u)}
        y = norm.map(label_map).values

    meta, X, names = frame_to_xy(df, ["_label"])
    if len(X) > max_rows:
        rng = np.random.RandomState(seed)
        keep_idx = []
        for c, n in zip(*np.unique(y, return_counts=True)):
            rows = np.where(y == c)[0]
            quota = min(len(rows), max(50, int(round(n * max_rows / len(X)))))
            keep_idx.append(rng.choice(rows, size=quota, replace=False))
        keep_idx = np.sort(np.concatenate(keep_idx))
        X, y = X[keep_idx], y[keep_idx]
        meta = meta.iloc[keep_idx].reset_index(drop=True)
        print("  subsampled to %d rows" % len(X))
    return finalize(X, y, names, label_map, "cic_andmal",
                    test_size, val_size, seed, meta, split_mode)


def load_vnat(raw_dir, task="binary", test_size=0.2, val_size=0.1,
              seed=42, split_mode="random"):
    # VNAT (MIT Lincoln Laboratory). Labels come from pcap file names:
    # vpn_<app>_capture<n>.pcap / nonvpn_<app>_capture<n>.pcap.
    # Binary task mirrors ISCX-VPN: realtime (streaming/voip) vs bulk.
    # Non-TLS encrypted classes (SSH/RDP, SFTP/rsync/scp) largely vanish
    # under TLS-only extraction; the paper discusses this in Sec. 3.1.
    cat = {"netflix": "streaming", "youtube": "streaming", "vimeo": "streaming",
           "voip": "voip", "zoiper": "voip", "skype": "chat",
           "sftp": "filetransfer", "rsync": "filetransfer", "scp": "filetransfer",
           "ssh": "remote", "rdp": "remote"}
    realtime = {"streaming", "voip"}

    def label_fn(parts, name):
        n = name.lower()
        for app in sorted(cat, key=len, reverse=True):
            if app in n:
                return cat[app]
        return "other"

    df = extract_pcaps(Path(raw_dir), "**/*.pcap", label_fn)
    lab = df["_label"].astype(str)
    if task == "binary":
        y = lab.isin(realtime).astype(int).values
        label_map = {"bulk": 0, "realtime": 1}
    else:
        u = sorted(lab.unique())
        label_map = {c: i for i, c in enumerate(u)}
        y = lab.map(label_map).values
    meta, X, names = frame_to_xy(df, ["_label"])
    return finalize(X, y, names, label_map, "vnat",
                    test_size, val_size, seed, meta, split_mode)


def download_hikari(raw_dir):
    import requests
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "ALLFLOWMETER_HIKARI2021.csv.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return
    print("downloading %s" % HIKARI_URL)
    with requests.get(HIKARI_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    h = hashlib.md5()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != HIKARI_MD5:
        print("[warn] md5 mismatch: got %s" % h.hexdigest())


LOADERS = {"bccc_dohbrw": load_bccc, "iscx_vpn": load_iscx,
           "hikari": load_hikari, "cic_andmal": load_cic,
           "vnat": load_vnat}
