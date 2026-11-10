"""Test-wide guards.

The suite runs on the owner's machine, where the user is in group kaiacord and
the production Policy Gate listens on /run/kaiacord/policy_gate.sock. A test
that called the console's `status` command reached it, and every run wrote an
audited request into the live ledger (found 2026-11-09). No test may talk to the
production gate: its socket path is redirected to a path that does not exist.
Tests that need a gate start their own (on the per-user fallback socket).
"""

import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest


@pytest.fixture(autouse=True)
def _never_reach_the_production_gate(tmp_path_factory, monkeypatch):
    import config
    unreachable = str(tmp_path_factory.getbasetemp() / "no-production-gate" / "policy_gate.sock")
    monkeypatch.setattr(config, "POLICY_GATE_SOCKET", unreachable)
    yield
