from types import SimpleNamespace

import pytest

import config
import security.policy_gate as policy_gate_module


def test_trigger_lockdown_reports_success_when_service_starts(monkeypatch):
    monkeypatch.setattr(policy_gate_module, "log_security_event", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        policy_gate_module.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(returncode=0),
    )

    assert policy_gate_module.trigger_lockdown("test") is True
    assert len(calls) == 1


def test_trigger_lockdown_raises_when_service_and_fallback_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_gate_module, "log_security_event", lambda *a, **k: None)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "kaia-lockdown.sh").write_text("#!/bin/sh\nexit 1\n")
    calls = []
    monkeypatch.setattr(
        policy_gate_module.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="fallback exited with status 1"):
        policy_gate_module.trigger_lockdown("test")

    assert len(calls) == 2
