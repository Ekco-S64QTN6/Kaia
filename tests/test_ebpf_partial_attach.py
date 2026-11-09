"""One kprobe that cannot attach must not take every probe down with it.

With BCC's kprobe__ auto-attach, a single missing or inlined symbol made BPF()
raise, and every probe was off, honeypot watching included. Kprobes are now
attached one by one and a partial load is reported as ebpf: degraded.
"""

import os
import re

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health, ebpf_telemetry
from security.ebpf_telemetry import KPROBES, PRIVILEGE_KPROBES, EBPFTelemetryEngine


class FakeTable:
    def open_ring_buffer(self, cb):
        pass


class FakeBPF:
    fail = set()

    def __init__(self, text):
        self.attached, self.detached = [], []

    def __getitem__(self, name):
        return FakeTable()

    def attach_kprobe(self, event, fn_name):
        if event in self.fail:
            raise Exception(f"cannot attach kprobe, probe entry may not exist: {event}")
        self.attached.append((event, fn_name))

    def detach_kprobe(self, event):
        self.detached.append(event)

    def ring_buffer_poll(self, timeout):
        pass


@pytest.fixture
def engine(monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    monkeypatch.setattr(ebpf_telemetry, "HAS_BCC", True)
    monkeypatch.setattr(ebpf_telemetry, "BPF", FakeBPF, raising=False)
    eng = EBPFTelemetryEngine()
    eng._poll_ring_buffers = lambda: None
    yield eng, seen
    eng.stop()


def test_no_kprobe_is_auto_attached():
    text = ebpf_telemetry.bpf_text
    assert not re.search(r"\bint kprobe__\w+\(", text)
    for fn_name, _symbol, _ in KPROBES:
        assert f"int {fn_name}(" in text
    for fn_name, _symbol in PRIVILEGE_KPROBES:
        assert f"int {fn_name}(" in text


def test_all_attached_is_active(engine):
    eng, seen = engine
    FakeBPF.fail = set()
    assert eng.start() is True
    assert seen["ebpf"][0] == "active"
    assert len(eng.bpf.attached) == len(KPROBES) + len(PRIVILEGE_KPROBES)


def test_one_missing_symbol_degrades_instead_of_failing(engine):
    eng, seen = engine
    FakeBPF.fail = {"tcp_retransmit_skb"}
    assert eng.start() is True
    state, detail = seen["ebpf"]
    assert state == "degraded" and "tcp_retransmit_skb (retransmission telemetry)" in detail
    attached = {e for e, _ in eng.bpf.attached}
    assert "security_file_open" in attached and "commit_creds" in attached


def test_commit_creds_never_runs_without_the_exec_marker(engine):
    eng, seen = engine
    FakeBPF.fail = {"security_bprm_committing_creds"}
    eng.start()
    assert "commit_creds" not in {e for e, _ in eng.bpf.attached}
    assert "security_bprm_committing_creds" in eng.missing_probes


def test_marker_without_commit_creds_is_detached(engine):
    eng, seen = engine
    FakeBPF.fail = {"commit_creds"}
    eng.start()
    assert eng.bpf.detached == ["security_bprm_committing_creds"]
    assert "privilege-gain" in seen["ebpf"][1]


def test_honeypot_reports_missing_inode_probe(monkeypatch):
    import types
    from security.honeypot import HoneypotCoordinator
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    fake = types.SimpleNamespace(decoys={"a": ""}, canary_paths=["f"])
    ebpf = types.SimpleNamespace(bpf=object(), missing_probes={"security_file_open": "x"})
    HoneypotCoordinator._report_health(fake, ebpf, ["a", "f"], 2)
    assert seen["honeypot"][0] == "degraded" and "inode probe not attached" in seen["honeypot"][1]
