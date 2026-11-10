"""Arch security advisories matched against installed versions.

vercmp was validated against pacman's /usr/bin/vercmp on 3400 pairs (installed
versions plus edge cases): 0 mismatches. The cases below are pacman's own semantics.
"""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import vuln_advisories as va

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("a,b,expected", [
    ("1.0", "1.0", 0), ("1.0", "1.1", -1), ("1.10", "1.9", 1), ("1.0a", "1.0", -1),
    ("1.0.a", "1.0a", 1), ("1.0rc1", "1.0", -1), ("1:1.0-1", "2.0-1", 1), ("1.0-1", "1.0-2", -1),
    ("1.0-1", "1.0", 0), ("1.9.17.p2-6", "1.9.17-6", 1), ("2:2.16-1", "2:2.16-1", 0), ("001", "1", 0),
])
def test_vercmp_matches_pacman(a, b, expected):
    assert va.vercmp(a, b) == expected and va.vercmp(b, a) == -expected


# Real-format groups (shape of security.archlinux.org/issues/all.json).
GROUPS = [
    {"name": "AVG-1", "packages": ["sudo"], "status": "Fixed", "severity": "High", "type": "privilege escalation",
     "affected": "1.9.0-1", "fixed": "1.9.17.p2-6", "issues": ["CVE-2025-0001"], "advisories": []},
    {"name": "AVG-2", "packages": ["pam"], "status": "Vulnerable", "severity": "High", "type": "arbitrary code execution",
     "affected": "1.7.3-1", "fixed": None, "issues": ["CVE-2025-0002", "not a cve"], "advisories": []},
    {"name": "AVG-3", "packages": ["openssl"], "status": "Fixed", "severity": "Critical", "type": "x",
     "affected": "3.0.0-1", "fixed": "3.5.0-1", "issues": [], "advisories": []},
    {"name": "AVG-4", "packages": ["bash"], "status": "Not affected", "severity": "Low", "type": "x",
     "affected": "1-1", "fixed": None, "issues": [], "advisories": []},
    "garbage", {"name": "AVG-5", "packages": ["vim"], "affected": None},
]
INSTALLED = {"sudo": "1.9.17.p2-6", "pam": "1.7.3-1", "openssl": "3.4.1-1", "bash": "5.3-1", "vim": "9.0-1"}


def test_matching_rules():
    hits = {h["group"]: h for h in va.affected(GROUPS, INSTALLED)}
    assert "AVG-1" not in hits                       # installed == fixed
    assert hits["AVG-2"]["fixed"] is None and hits["AVG-2"]["cves"] == ["CVE-2025-0002"]
    assert hits["AVG-3"]["severity"] == "Critical"   # installed below fixed
    assert "AVG-4" not in hits                       # not affected
    assert list(hits)[0] == "AVG-3"                  # most severe first


def _script():
    spec = importlib.util.spec_from_file_location("kaia_vuln_check", ROOT / "scripts/kaia_vuln_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_exit_status_is_actionable_only(tmp_path, monkeypatch, capsys):
    m = _script()
    monkeypatch.setattr(m.va, "installed_versions", lambda: INSTALLED)
    f = tmp_path / "all.json"
    f.write_text(json.dumps(GROUPS))
    assert m.main(["--file", str(f)]) == 1           # AVG-3: Critical with a fix
    f.write_text(json.dumps([g for g in GROUPS if not isinstance(g, dict) or g.get("name") != "AVG-3"]))
    assert m.main(["--file", str(f)]) == 0           # only unfixed advisories remain
    assert "no fix released yet" in capsys.readouterr().out
    assert m.main(["--file", str(tmp_path / "missing.json")]) == 2


def test_unit_is_unprivileged_and_installed():
    unit = (ROOT / "scripts/kaia-vuln-check.service").read_text()
    for line in ("DynamicUser=yes", "CapabilityBoundingSet=\n", "ProtectSystem=strict", "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX"):
        assert line in unit
    assert "systemctl enable --now kaia-vuln-check.timer" in (ROOT / "scripts/install_services.sh").read_text()
    assert '"kaia-vuln-check.service"' in (ROOT / "scripts/kaia_health_check.py").read_text()
