"""After a kernel upgrade the gate says why probes will not (re)load (#104).

On Arch, upgrading the kernel package deletes /lib/modules/<running>/build.
Loaded probes keep working, but BCC cannot compile again until a reboot, so a
gate restart in between silently loaded nothing.
"""

import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health, ebpf_telemetry
from security.ebpf_telemetry import EBPFTelemetryEngine, RINGBUF_STREAMS, kernel_build_tree_present


def test_build_tree_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(ebpf_telemetry.os.path, "exists", lambda p: False)
    (tmp_path / "6.18.1-lts" / "build").mkdir(parents=True)
    assert kernel_build_tree_present("6.18.1-lts", str(tmp_path))
    assert not kernel_build_tree_present("6.18.2-lts", str(tmp_path))


@pytest.fixture
def seen(monkeypatch):
    out = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: out.__setitem__(name, (state, detail)))
    return out


def test_failed_load_after_an_upgrade_says_reboot(monkeypatch, seen):
    monkeypatch.setattr(ebpf_telemetry, "HAS_BCC", True)
    monkeypatch.setattr(ebpf_telemetry, "kernel_build_tree_present", lambda *a: False)

    def broken(text):
        raise Exception("Failed to compile BPF module <text>")

    monkeypatch.setattr(ebpf_telemetry, "BPF", broken, raising=False)
    assert EBPFTelemetryEngine().start() is False
    state, detail = seen["ebpf"]
    assert state == "failed" and "reboot to load the probes" in detail


class _Drops:
    def Key(self, i):
        return i

    def __getitem__(self, k):
        return [0]


def test_running_probes_turn_degraded_when_the_tree_disappears(monkeypatch, seen):
    engine = EBPFTelemetryEngine()
    engine.bpf = {"ringbuf_drops": _Drops()}
    present = [True]
    monkeypatch.setattr(ebpf_telemetry, "kernel_build_tree_present", lambda *a: present[0])
    engine._report_ringbuf_drops(now=1000.0)
    assert seen["ebpf"][0] == "active"
    present[0] = False          # pacman -Syu removed the running kernel's tree
    engine._report_ringbuf_drops(now=1100.0)
    assert seen["ebpf"][0] == "degraded" and "reboot" in seen["ebpf"][1]
