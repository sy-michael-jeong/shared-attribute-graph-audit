# Extract TLS flows (features + handshake metadata) from pcap with Zeek,
# falling back to scapy. Non-TLS flows are dropped.
from __future__ import annotations
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import META_COLUMNS


# Metadata columns the HIN builder looks for. Adapters propagate these.


def _normalize_cert_issuer(issuer_str: str) -> str:
    """Map an X.509 issuer DN to an issuing-ecosystem bucket.
    Example: "CN=Let's Encrypt Authority X3,O=Let's Encrypt,C=US" -> PublicCA_LetsEncrypt
    """
    if not issuer_str or issuer_str in ("-", "(empty)", ""):
        return ""
    s = issuer_str.lower()
    if "let's encrypt" in s or "letsencrypt" in s: return "PublicCA_LetsEncrypt"
    if "digicert" in s:    return "PublicCA_DigiCert"
    if "sectigo" in s or "comodo" in s: return "PublicCA_Sectigo"
    if "cloudflare" in s:  return "PublicCA_Cloudflare"
    if "globalsign" in s:  return "PublicCA_GlobalSign"
    if "amazon" in s and "ca" in s: return "PublicCA_Amazon"
    if "google trust" in s or "gts" in s: return "PublicCA_Google"
    if "usertrust" in s:   return "PublicCA_USERTrust"
    if "entrust" in s:     return "PublicCA_Entrust"
    if "godaddy" in s:     return "PublicCA_GoDaddy"
    if "isrg" in s:        return "PublicCA_ISRG"
    if "self-signed" in s or "self signed" in s: return "SelfSigned"
    if "ca" in s or "certificate authority" in s or "o=" in s:
        return "PublicCA_Other"
    return "Private_or_Unknown"


def _tls_cipher_group(version: str, cipher: str) -> str:
    """Map a TLS version and cipher to a security-level bucket.
    Example: "TLSv13", "TLS_AES_256_GCM_SHA384" -> TLSv13_AES_GCM
    """
    if not version and not cipher:
        return ""
    v = (str(version) or "").upper().replace("-", "").replace(" ", "")
    c = (str(cipher) or "").upper()

    if "TLSV13" in v or v == "TLS13":
        if "AES" in c and "GCM" in c: return "TLSv13_AES_GCM"
        if "CHACHA20" in c:           return "TLSv13_CHACHA20"
        if "AES" in c and "CCM" in c: return "TLSv13_AES_CCM"
        return "TLSv13_Other"
    # TLS 1.2
    if "TLSV12" in v or v == "TLS12":
        if "ECDHE" in c and "GCM" in c: return "TLSv12_ECDHE_AES_GCM"
        if "ECDHE" in c and "CHACHA20" in c: return "TLSv12_ECDHE_CHACHA20"
        if "ECDHE" in c and "CBC" in c: return "TLSv12_ECDHE_CBC_Legacy"
        if "DHE" in c and "GCM" in c:   return "TLSv12_DHE_AES_GCM"
        if "RSA" in c and "CBC" in c:   return "TLSv12_RSA_CBC_Legacy"
        if "RSA" in c and "GCM" in c:   return "TLSv12_RSA_AES_GCM"
        return "TLSv12_Other"
    # TLS 1.0/1.1 (deprecated)
    if "TLSV1" in v or "TLSV11" in v or v in ("TLS10", "TLS11"):
        return "TLSv1x_Legacy_Deprecated"
    if "SSL" in v:
        return "SSL_Deprecated"
    return "Unknown"


def _cert_validity_bucket(not_valid_before: str, not_valid_after: str) -> str:
    """Bucket the certificate validity period in days.
    not_valid_before and not_valid_after are Zeek epoch timestamp strings.
    """
    try:
        from_ts = float(not_valid_before) if not_valid_before and not_valid_before != "-" else 0.0
        to_ts   = float(not_valid_after)  if not_valid_after  and not_valid_after  != "-" else 0.0
        if from_ts <= 0 or to_ts <= 0:
            return "unknown"
        if to_ts < from_ts:
            return "invalid_range"
        days = (to_ts - from_ts) / 86400.0
        if   days <= 7:    return "validity_0_7d"
        elif days <= 30:   return "validity_8_30d"
        elif days <= 90:   return "validity_31_90d"
        elif days <= 398:  return "validity_91_398d"
        elif days <= 825:  return "validity_399_825d"
        else:              return "validity_long_2y_plus"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- backend
def _verify_real_zeek(binary: str) -> bool:
    try:
        r = subprocess.run([binary, "--version"],
                           capture_output=True, text=True, timeout=5)
        out = ((r.stdout or "") + (r.stderr or "")).lower()
        return r.returncode == 0 and ("zeek" in out or "bro" in out) \
               and "usage:" not in out
    except Exception:
        return False


def _have_zeek() -> bool:
    for cand in ("zeek", "bro"):
        p = shutil.which(cand)
        if p and _verify_real_zeek(p):
            return True
    return False


def _zeek_binary() -> str:
    for cand in ("zeek", "bro"):
        p = shutil.which(cand)
        if p and _verify_real_zeek(p):
            return p
    raise RuntimeError("No real Zeek/Bro network analyzer on PATH "
                       "(found a same-named tool, but not Zeek itself)")


# ----------------------------------------------------------------------- zeek
def _zeek_extract(pcap: Path, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_zeek_binary(), "-r", str(pcap.resolve())]
    try:
        proc = subprocess.run(cmd, cwd=work_dir, check=True,
                              capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        msg = ((e.stdout or "") + "\n" + (e.stderr or "")).strip()
        raise RuntimeError(
            "Zeek failed on " + str(pcap.name) + " (exit " + str(e.returncode) + ").\n"
            "cmd: " + " ".join(cmd) + "\n"
            "output:\n" + msg[:4000]
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Zeek timed out (>600s) on " + str(pcap.name)) from e

    out = {}
    for name in ("conn.log", "ssl.log", "x509.log"):
        p = work_dir / name
        if p.exists():
            out[name] = _read_zeek_log(p)
    return out


def _read_zeek_log(path: Path) -> pd.DataFrame:
    """Parse a Zeek TSV log (#fields header)."""
    fields = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                fields = line.strip().split("\t")[1:]
                continue
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
    if not fields or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=fields)
    # Numeric conversion best-effort
    for c in df.columns:
        if c in ("ts", "duration", "orig_bytes", "resp_bytes",
                 "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
                 "missed_bytes", "id.orig_p", "id.resp_p"):
            df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    return df


def _zeek_ja3_fallback(version: str, cipher: str) -> str:
    """A fingerprint over the TLS version and cipher suite.

    Zeek exports a ja3 column only with the JA3 script package loaded, and the
    version used for this artifact does not. This digest carries the version
    and the cipher suite and nothing else, so it is coarser than JA3, which
    also covers extensions and curves. The paper names the relation built on
    this column VersionCipher for that reason. The stored column keeps the
    name ja3 because the extractor writes it under both paths."""
    s = (str(version) + "|" + str(cipher)).encode()
    return hashlib.md5(s).hexdigest()


def _features_from_zeek(conn: pd.DataFrame, ssl: pd.DataFrame,
                         x509: pd.DataFrame) -> pd.DataFrame:
    if conn.empty:
        return pd.DataFrame()

    # Keep only flows with a TLS handshake (uid present in ssl.log).
    if not ssl.empty and "uid" in ssl.columns:
        tls_uids = set(ssl["uid"].astype(str).tolist())
        conn = conn[conn["uid"].astype(str).isin(tls_uids)].copy()
    else:
        conn = conn[conn.get("id.resp_p", pd.Series([], dtype=str))
                    .astype(str) == "443"].copy()
    if conn.empty:
        return pd.DataFrame()

    keep_num = ["duration", "orig_bytes", "resp_bytes",
                "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
                "missed_bytes"]
    for c in keep_num:
        if c not in conn.columns:
            conn[c] = 0.0
    feats = conn[keep_num].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    dur = feats["duration"].replace(0, 1.0)
    feats["orig_pkt_rate"]    = feats["orig_pkts"] / dur
    feats["resp_pkt_rate"]    = feats["resp_pkts"] / dur
    feats["orig_byte_rate"]   = feats["orig_bytes"] / dur
    feats["resp_byte_rate"]   = feats["resp_bytes"] / dur
    feats["orig_avg_pkt_size"] = feats["orig_bytes"] / feats["orig_pkts"].replace(0, 1)
    feats["resp_avg_pkt_size"] = feats["resp_bytes"] / feats["resp_pkts"].replace(0, 1)

    # one-hot for proto/service/conn_state (small cardinality)
    for col in ("proto", "service", "conn_state"):
        if col in conn.columns:
            d = pd.get_dummies(conn[col].astype(str), prefix=col).astype(np.float32)
            feats = pd.concat([feats.reset_index(drop=True),
                               d.reset_index(drop=True)], axis=1)

    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    meta = pd.DataFrame(index=range(len(conn)))
    meta["src_ip"] = conn.get("id.orig_h", pd.Series([""] * len(conn))).astype(str).values
    meta["dst_ip"] = conn.get("id.resp_h", pd.Series([""] * len(conn))).astype(str).values
    meta["sport"]  = conn.get("id.orig_p", pd.Series([0] * len(conn))).fillna(0).astype(int).values
    meta["dport"]  = conn.get("id.resp_p", pd.Series([0] * len(conn))).fillna(0).astype(int).values
    meta["ts"]     = conn.get("ts", pd.Series([0.0] * len(conn))).fillna(0.0).astype(float).values

    # Defaults — filled in below from ssl/x509 logs.
    meta["sni"] = ""
    meta["ja3"] = ""
    meta["cert_subject"] = ""
    meta["alpn"] = ""
    meta["cert_issuer_org"] = ""
    meta["tls_cipher_group"] = ""
    meta["cert_validity_bucket"] = ""

    # Merge ssl info by uid for SNI / JA3 / cipher / version / next_protocol(ALPN).
    # Zeek 5.x ssl.log: ts, uid, id.*, version, cipher, curve, server_name,
    #   resumed, last_alert, next_protocol(=ALPN), established, ssl_history,
    #   cert_chain_fps, client_cert_chain_fps, sni_matches_cert
    if not ssl.empty and "uid" in ssl.columns:
        ssl_cols = ["uid"]
        for c in ("server_name", "ja3", "version", "cipher", "next_protocol",
                  "cert_chain_fps", "cert_chain_fuids",
                  "subject", "issuer"):
            if c in ssl.columns:
                ssl_cols.append(c)
        ssl_sub = ssl[ssl_cols].copy().drop_duplicates(subset=["uid"], keep="first")
        merged = conn[["uid"]].reset_index(drop=True).merge(
            ssl_sub, on="uid", how="left")

        if "server_name" in merged.columns:
            meta["sni"] = merged["server_name"].fillna("").astype(str).values

        if "ja3" in merged.columns:
            meta["ja3"] = merged["ja3"].fillna("").astype(str).values
        else:
            vers = merged.get("version", pd.Series([""]*len(merged))).fillna("").astype(str)
            ciph = merged.get("cipher",  pd.Series([""]*len(merged))).fillna("").astype(str)
            meta["ja3"] = [_zeek_ja3_fallback(v, c) if (v or c) else ""
                           for v, c in zip(vers, ciph)]

        if "subject" in merged.columns:
            meta["cert_subject"] = merged["subject"].fillna("").astype(str).values

        if "next_protocol" in merged.columns:
            meta["alpn"] = merged["next_protocol"].fillna("").astype(str).values

        if "version" in merged.columns and "cipher" in merged.columns:
            v = merged["version"].fillna("").astype(str)
            c = merged["cipher"].fillna("").astype(str)
            meta["tls_cipher_group"] = [_tls_cipher_group(vv, cc) for vv, cc in zip(v, c)]

        if not x509.empty:
            x509_id_col   = next((c for c in ("fingerprint", "id") if c in x509.columns), None)
            x509_subj_col = next((c for c in ("certificate.subject", "subject") if c in x509.columns), None)
            x509_issuer_col = next((c for c in ("certificate.issuer", "issuer") if c in x509.columns), None)
            x509_nvb_col  = next((c for c in ("certificate.not_valid_before", "not_valid_before") if c in x509.columns), None)
            x509_nva_col  = next((c for c in ("certificate.not_valid_after", "not_valid_after") if c in x509.columns), None)
            chain_col = next((c for c in ("cert_chain_fps", "cert_chain_fuids") if c in merged.columns), None)

            if x509_id_col and chain_col:
                first_fp = merged[chain_col].fillna("").astype(str)\
                                .str.split(",").str[0].str.strip()
                if x509_subj_col and meta["cert_subject"].eq("").all():
                    subj_map = dict(zip(x509[x509_id_col].astype(str),
                                        x509[x509_subj_col].astype(str)))
                    meta["cert_subject"] = [subj_map.get(fp, "") for fp in first_fp]
                if x509_issuer_col:
                    iss_map = dict(zip(x509[x509_id_col].astype(str),
                                       x509[x509_issuer_col].astype(str)))
                    meta["cert_issuer_org"] = [_normalize_cert_issuer(iss_map.get(fp, ""))
                                                for fp in first_fp]
                if x509_nvb_col and x509_nva_col:
                    nvb_map = dict(zip(x509[x509_id_col].astype(str),
                                       x509[x509_nvb_col].astype(str)))
                    nva_map = dict(zip(x509[x509_id_col].astype(str),
                                       x509[x509_nva_col].astype(str)))
                    meta["cert_validity_bucket"] = [
                        _cert_validity_bucket(nvb_map.get(fp, ""), nva_map.get(fp, ""))
                        for fp in first_fp
                    ]

    # Combine: features + metadata in one DataFrame, mark metadata columns
    # explicitly by membership in META_COLUMNS (downstream can split).
    out = pd.concat([feats.reset_index(drop=True),
                     meta.reset_index(drop=True)], axis=1)
    return out


# ---------------------------------------------------------------------- scapy
def _scapy_extract(pcap: Path) -> pd.DataFrame:
    """Pure-Python fallback. SNI extraction from ClientHello; JA3 via
    scapy.layers.tls when available."""
    try:
        from scapy.all import PcapReader, TCP, IP, IPv6, Raw
    except ImportError as e:
        raise RuntimeError(
            "Neither zeek nor scapy is available. Install one: "
            "`apt install zeek` or `pip install scapy`.") from e

    flows = {}
    tls_uids = set()
    sni_map  = {}   # flow_key -> SNI string
    ja3_map  = {}   # flow_key -> partial ja3-like hash from ClientHello bytes

    def _canonical(src_ip, sport, dst_ip, dport):
        a = (src_ip, sport); b = (dst_ip, dport)
        if a <= b:
            return (a[0], a[1], b[0], b[1], "tcp"), True
        return (b[0], b[1], a[0], a[1], "tcp"), False

    def _parse_sni_and_ja3(data: bytes) -> tuple:
        """Best-effort raw-byte parser for TLS ClientHello.
        Returns (sni_str, ja3_hash) — empty strings on failure.
        Reference: RFC 5246/8446; scapy.layers.tls.handshake.
        """
        try:
            # Skip TLS record header (5B) + handshake header (4B)
            if len(data) < 43:
                return "", ""
            # Handshake type 0x01 = ClientHello
            if data[5] != 0x01:
                return "", ""
            i = 5 + 4  # after handshake header
            # client_version (2) + random (32)
            client_version = data[i:i+2]
            i += 2 + 32
            # session_id_length (1) + session_id
            sid_len = data[i]; i += 1 + sid_len
            # cipher_suites_length (2)
            cs_len = int.from_bytes(data[i:i+2], "big"); i += 2
            cipher_bytes = data[i:i+cs_len]; i += cs_len
            # compression_methods_length (1)
            cm_len = data[i]; i += 1 + cm_len
            # extensions_length (2)
            ext_len = int.from_bytes(data[i:i+2], "big"); i += 2
            ext_end = i + ext_len
            sni = ""
            ext_types = []
            curves = b""
            ec_pt_fmts = b""
            while i + 4 <= ext_end:
                etype = int.from_bytes(data[i:i+2], "big"); i += 2
                elen  = int.from_bytes(data[i:i+2], "big"); i += 2
                ebody = data[i:i+elen]; i += elen
                ext_types.append(etype)
                if etype == 0x0000 and len(ebody) >= 5:  # server_name
                    # server_name_list_length (2) skip; type (1); name_length (2)
                    name_len = int.from_bytes(ebody[3:5], "big")
                    sni_bytes = ebody[5:5+name_len]
                    try: sni = sni_bytes.decode("ascii")
                    except Exception: sni = sni_bytes.decode("utf-8", errors="ignore")
                elif etype == 0x000a:  # supported_groups (curves)
                    curves = ebody[2:]
                elif etype == 0x000b:  # ec_point_formats
                    ec_pt_fmts = ebody[1:]
            # JA3 string per ja3 spec:
            #   SSLVersion,CipherSuites,Extensions,EllipticCurves,EllipticCurveFormats
            # where each list is "-"-joined decimal values.
            ja3_str_parts = [
                str(int.from_bytes(client_version, "big")),
                "-".join(str(int.from_bytes(cipher_bytes[k:k+2], "big"))
                         for k in range(0, len(cipher_bytes), 2)),
                "-".join(str(t) for t in ext_types),
                "-".join(str(int.from_bytes(curves[k:k+2], "big"))
                         for k in range(0, len(curves), 2)) if curves else "",
                "-".join(str(b) for b in ec_pt_fmts) if ec_pt_fmts else "",
            ]
            ja3_str = ",".join(ja3_str_parts)
            ja3_hash = hashlib.md5(ja3_str.encode()).hexdigest()
            return sni, ja3_hash
        except Exception:
            return "", ""

    with PcapReader(str(pcap)) as rd:
        for pkt in rd:
            if TCP not in pkt:
                continue
            try:
                if IP in pkt:
                    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
                else:
                    src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            except Exception:
                continue
            k, is_forward = _canonical(src_ip, sport, dst_ip, dport)

            f = flows.setdefault(k, dict(
                orig_pkts=0, resp_pkts=0,
                orig_bytes=0, resp_bytes=0,
                first_ts=float(pkt.time), last_ts=float(pkt.time),
                src_ip=k[0], sport=k[1], dst_ip=k[2], dport=k[3]))
            f["last_ts"] = float(pkt.time)
            n = len(pkt)
            if is_forward:
                f["orig_pkts"] += 1; f["orig_bytes"] += n
            else:
                f["resp_pkts"] += 1; f["resp_bytes"] += n

            # TLS ClientHello detection + SNI/JA3 extraction
            if (dport == 443 or sport == 443) and Raw in pkt:
                data = bytes(pkt[Raw].load)
                if len(data) >= 6 and data[0] == 0x16 and data[1] == 0x03:
                    tls_uids.add(k)
                    if data[5] == 0x01 and k not in sni_map:  # ClientHello
                        sni, ja3 = _parse_sni_and_ja3(data)
                        sni_map[k] = sni; ja3_map[k] = ja3

    rows = []
    for k, f in flows.items():
        if k not in tls_uids:
            continue
        dur = max(f["last_ts"] - f["first_ts"], 1e-3)
        rows.append({
            # numeric features (input to classifier)
            "duration": dur,
            "orig_pkts":  f["orig_pkts"],  "resp_pkts":  f["resp_pkts"],
            "orig_bytes": f["orig_bytes"], "resp_bytes": f["resp_bytes"],
            "orig_pkt_rate":  f["orig_pkts"]  / dur,
            "resp_pkt_rate":  f["resp_pkts"]  / dur,
            "orig_byte_rate": f["orig_bytes"] / dur,
            "resp_byte_rate": f["resp_bytes"] / dur,
            "resp_avg_pkt_size": f["resp_bytes"] / max(f["resp_pkts"], 1),
            # metadata columns (input to HIN graph builder)
            "src_ip": f["src_ip"], "dst_ip": f["dst_ip"],
            "sport":  f["sport"],  "dport":  f["dport"],
            "ts":     f["first_ts"],
            "sni": sni_map.get(k, ""),
            "ja3": ja3_map.get(k, ""),
            "cert_subject": "",  # scapy backend doesn't parse server cert
            "alpn": "",
            "cert_issuer_org": "",
            "tls_cipher_group": "",
            "cert_validity_bucket": "",
        })
    return pd.DataFrame(rows)


def extract_flow_features(pcap, backend: str = "auto"):
    from pathlib import Path as _P
    pcap = _P(pcap)
    if not pcap.exists():
        raise FileNotFoundError(pcap)
    if backend == "auto":
        backend = "zeek" if _have_zeek() else "scapy"
    if backend == "zeek":
        with tempfile.TemporaryDirectory() as td:
            logs = _zeek_extract(pcap, Path(td))
        return _features_from_zeek(
            logs.get("conn.log", pd.DataFrame()),
            logs.get("ssl.log", pd.DataFrame()),
            logs.get("x509.log", pd.DataFrame()),
        )
    if backend == "scapy":
        return _scapy_extract(pcap)
    raise ValueError("unknown backend: " + str(backend))


def split_meta_and_features(df: pd.DataFrame):
    """Helper: separate the metadata columns (META_COLUMNS) from the numeric
    feature columns. Returns (feat_df, meta_df) with matching row index.
    """
    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return df[feat_cols].copy(), (df[meta_cols].copy() if meta_cols else None)
