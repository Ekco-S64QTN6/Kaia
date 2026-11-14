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


@pytest.fixture(autouse=True)
def _kernel_build_tree_is_not_read_from_the_host(monkeypatch):
    """The eBPF engine reports `degraded` when /lib/modules/<running kernel>/build
    is gone (a kernel upgrade without a reboot). Tests that load a fake BPF read
    that from the real host, so four of them failed on the owner's machine for
    as long as a reboot was pending (2026-11-14). Tests of that state set their
    own answer over this one."""
    from security import ebpf_telemetry
    monkeypatch.setattr(ebpf_telemetry, "kernel_build_tree_present", lambda *a, **k: True)
    yield
