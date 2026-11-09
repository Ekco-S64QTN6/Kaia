"""Credentials in Kaia's own logs, state and deployed tree are found, never repeated.

The signing secret was once committed to this public repository. The scan must
find it (and common credential formats) wherever Kaia writes, and must never
copy the matched value into a finding, a log line or a security event.
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health
from security import secret_exposure as se

SECRET = "4f" * 32  # a 64-hex signing secret


def _write(p: Path, text) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text if isinstance(text, bytes) else text.encode())
    return p


@pytest.mark.parametrize("text,pattern", [
    ("-----BEGIN OPENSSH " + "PRIVATE KEY-----\nb3Bl\n", "private_key"),
    ("-----BEGIN " + "PRIVATE KEY-----\n", "private_key"),
    ("aws = AKIA" + "Q3XN2P7RTV5WZK4M\n", "aws_access_key_id"),
    ("token ghp_" + "a1B2" * 9 + "\n", "github_token"),
    ("x = sk-ant-api03-" + "Ab9_" * 8 + "\n", "anthropic_api_key"),
    ("OPENAI=sk-proj-" + "Zx81" * 12 + "\n", "openai_api_key"),
    ("export KAIA_CAPABILITY_TOKEN_SECRET=" + "ab" * 32 + "\n", "kaia_secret_assignment"),
])
def test_credential_formats_are_found(tmp_path, text, pattern):
    _write(tmp_path / "logs/kaia.log", "line one\n" + text)
    (f,) = se.scan([str(tmp_path)])["findings"]
    assert f["pattern"] == pattern and f["line"] == 2


def test_the_live_secret_is_found_anywhere_and_never_repeated(tmp_path, caplog):
    _write(tmp_path / "state/audit_ledger.json", '{"justification": "use ' + SECRET + '"}\n')
    events = []
    mon = se.SecretExposureMonitor([str(tmp_path)], SECRET, log_event=lambda **kw: events.append(kw))
    with caplog.at_level(logging.DEBUG):
        report = mon.run_once()
    (f,) = report["findings"]
    assert f["pattern"] == se.LIVE_SECRET
    blob = repr(report) + repr(events) + "".join(r.getMessage() for r in caplog.records)
    assert SECRET not in blob and SECRET[:16] not in blob
    assert events[0]["event_type"] == "secret_exposure" and events[0]["disposition"] == "observed"


def test_match_spanning_a_chunk_boundary_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "CHUNK", 64)
    _write(tmp_path / "f", "x" * 50 + "\n" * 3 + SECRET + "\n")
    (f,) = se.scan([str(tmp_path)], SECRET)["findings"]
    assert f["line"] == 4


def test_documentation_examples_and_short_secrets_are_ignored(tmp_path):
    _write(tmp_path / "decoy.json", '{"aws_key": "AKIAIOSFODNN7EXAMPLE"}\n')
    assert se.scan([str(tmp_path)], live_secret="short")["findings"] == []


def test_symlinks_git_and_large_files_are_not_followed(tmp_path, monkeypatch):
    outside = _write(tmp_path / "outside/leak", SECRET)
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "link").symlink_to(outside)
    (scope / "dirlink").symlink_to(outside.parent)
    _write(scope / ".git/config", SECRET)
    monkeypatch.setattr(se, "MAX_FILE_BYTES", 100)
    _write(scope / "big", SECRET + "x" * 200)
    assert se.scan([str(scope)], SECRET)["findings"] == []


def test_each_exposure_is_reported_once(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    _write(tmp_path / "f", SECRET)
    events = []
    mon = se.SecretExposureMonitor([str(tmp_path)], SECRET, log_event=lambda **kw: events.append(kw))
    mon.run_once()
    mon.run_once()
    assert len(events) == 1 and "1 exposures" in seen["secret_exposure"][1]
    (tmp_path / "f").write_text("clean")
    mon.run_once()
    assert seen["secret_exposure"][0] == "active" and "0 exposures" in seen["secret_exposure"][1]


def test_without_the_live_secret_the_check_is_degraded(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    se.SecretExposureMonitor([str(tmp_path)], None, log_event=lambda **kw: None).run_once()
    assert seen["secret_exposure"][0] == "degraded"


def test_gate_scans_its_state_logs_and_tree_alert_only():
    gate = Path("security/policy_gate.py").read_text()
    assert "scopes=[str(config.STATE_DIR), str(config.LOG_DIR), config.WORKSPACE_DIR]" in gate
    assert "live_secret=config.CAPABILITY_TOKEN_SECRET" in gate
    src = Path(se.__file__).read_text()
    assert "trigger_lockdown" not in src and "import config" not in src
    assert "secret_exposure" in control_health.GATE_CONTROLS


def test_the_shipped_tree_scans_clean():
    """The gate scans its deployed tree, tests included: nothing here may match."""
    root = Path(__file__).resolve().parent.parent
    scopes = [str(root / d) for d in ("core", "security", "scripts", "tests", "docs", "static")]
    scopes += [str(root / f) for f in ("README.md", "CLAUDE.md", "CHANGELOG.md", "kaia_dashboard.py", "main.py")]
    assert se.scan(scopes)["findings"] == []
