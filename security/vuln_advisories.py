"""Installed packages affected by Arch Linux security advisories.

Arch publishes every advisory group as JSON (security.archlinux.org/issues/
all.json). The whole list is fetched (nothing about this host is sent) and
matched locally against pacman's installed versions, the same rule arch-audit
uses: a package is affected when its version is at or above ``affected`` and,
if a fix exists, below ``fixed``. Groups marked "Not affected" are skipped.

Pure functions; the fetch lives in scripts/kaia_vuln_check.py, which runs from
kaia-vuln-check.timer as an unprivileged dynamic user, outside the gate.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

SEVERITY_RANK = {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}


# --- pacman version comparison (alpm_pkg_vercmp / rpmvercmp) ---------------

def _rpmvercmp(a: str, b: str) -> int:
    if a == b:
        return 0
    i = j = 0
    while i < len(a) and j < len(b):
        pi, pj = i, j
        while i < len(a) and not a[i].isalnum():
            i += 1
        while j < len(b) and not b[j].isalnum():
            j += 1
        if i >= len(a) or j >= len(b):
            break
        # Separator runs of different length decide: the longer one is newer.
        if (i - pi) != (j - pj):
            return -1 if (i - pi) < (j - pj) else 1
        si, sj = i, j
        if a[i].isdigit():
            while i < len(a) and a[i].isdigit():
                i += 1
            while j < len(b) and b[j].isdigit():
                j += 1
            is_num = True
        else:
            while i < len(a) and a[i].isalpha():
                i += 1
            while j < len(b) and b[j].isalpha():
                j += 1
            is_num = False
        seg_a, seg_b = a[si:i], b[sj:j]
        if not seg_b:
            return 1 if is_num else -1
        if is_num:
            seg_a, seg_b = seg_a.lstrip("0"), seg_b.lstrip("0")
            if len(seg_a) != len(seg_b):
                return 1 if len(seg_a) > len(seg_b) else -1
        if seg_a != seg_b:
            return 1 if seg_a > seg_b else -1
    a_rest, b_rest = a[i:], b[j:]
    if not a_rest and not b_rest:
        return 0
    if (not a_rest and not b_rest[:1].isalpha()) or a_rest[:1].isalpha():
        return -1
    return 1


def _split_evr(v: str):
    epoch, _, rest = v.partition(":") if ":" in v else ("0", "", v)
    if not epoch.isdigit():
        epoch, rest = "0", v
    version, _, release = rest.rpartition("-") if "-" in rest else (rest, "", "")
    return epoch, version, release or None


def vercmp(a: str, b: str) -> int:
    """-1, 0, 1 like pacman's vercmp."""
    if a == b:
        return 0
    ea, va, ra = _split_evr(a)
    eb, vb, rb = _split_evr(b)
    r = _rpmvercmp(ea, eb) or _rpmvercmp(va, vb)
    if r == 0 and ra is not None and rb is not None:
        r = _rpmvercmp(ra, rb)
    return r


# --- installed packages and matching ---------------------------------------

def installed_versions(local_db: str = "/var/lib/pacman/local") -> Dict[str, str]:
    out = {}
    for entry in os.listdir(local_db):
        desc = os.path.join(local_db, entry, "desc")
        try:
            with open(desc, encoding="utf-8", errors="replace") as f:
                text = f.read(256 * 1024)
        except OSError:
            continue
        m_name = re.search(r"^%NAME%\n(.+)$", text, re.MULTILINE)
        m_ver = re.search(r"^%VERSION%\n(.+)$", text, re.MULTILINE)
        if m_name and m_ver:
            out[m_name.group(1).strip()] = m_ver.group(1).strip()
    return out


def affected(groups: list, installed: Dict[str, str]) -> List[dict]:
    """Advisory groups that affect an installed package version."""
    hits = []
    for g in groups if isinstance(groups, list) else []:
        if not isinstance(g, dict) or g.get("status") == "Not affected":
            continue
        first, fixed = g.get("affected"), g.get("fixed")
        if not isinstance(first, str) or not first:
            continue
        for pkg in g.get("packages") or []:
            version = installed.get(pkg) if isinstance(pkg, str) else None
            if version is None:
                continue
            if vercmp(version, first) < 0:
                continue
            if isinstance(fixed, str) and fixed and vercmp(version, fixed) >= 0:
                continue
            issues = [i for i in (g.get("issues") or []) if isinstance(i, str) and re.fullmatch(r"CVE-\d{4}-\d{4,7}", i)]
            hits.append({
                "group": str(g.get("name", ""))[:20], "package": pkg, "installed": version,
                "fixed": fixed if isinstance(fixed, str) and fixed else None,
                "severity": g.get("severity") if g.get("severity") in SEVERITY_RANK else "Unknown",
                "type": str(g.get("type", ""))[:60], "cves": issues[:50],
            })
    hits.sort(key=lambda h: (-SEVERITY_RANK[h["severity"]], h["package"]))
    return hits
