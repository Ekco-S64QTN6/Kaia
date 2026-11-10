"""The HUD shows the gate's eBPF events instead of compiling its own (failing) engine."""

import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security import telemetry_publish as tp


class FakeEngine:
    bpf = object()

    def get_recent_execs(self, n):
        return [{"pid": 7, "comm": "bash", "filename": "/usr/bin/id", "timestamp": 1.0}]

    def get_recent_opens(self, n):
        return []

    def get_recent_deletes(self, n):
        return [{"pid": 8, "comm": "rm", "filename": "/tmp/x", "syscall": "sys_enter_unlinkat", "timestamp": 2.0}]

    def get_recent_connections(self, n):
        raise RuntimeError("probe gone")


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def test_round_trip_serves_the_engine_api(tmp_path):
    path = str(tmp_path / tp.FILE_NAME)
    tp.TelemetryPublisher(FakeEngine(), path).publish_once(now=1000.0)
    assert (Path(path).stat().st_mode & 0o777) == 0o640
    view = tp.PublishedTelemetry(path, clock=Clock(1001.0))
    assert view.bpf and view.get_recent_execs(5)[0]["filename"] == "/usr/bin/id"
    assert view.get_recent_deletes(5)[0]["comm"] == "rm" and view.get_recent_connections(5) == []


def test_stale_or_symlinked_or_missing_data_is_not_live(tmp_path):
    path = tmp_path / tp.FILE_NAME
    tp.TelemetryPublisher(FakeEngine(), str(path)).publish_once(now=1000.0)
    assert not tp.PublishedTelemetry(str(path), clock=Clock(1000.0 + tp.MAX_AGE + 1)).bpf
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert tp.PublishedTelemetry(str(link), clock=Clock(1001.0)).get_recent_execs() == []
    assert tp.PublishedTelemetry(str(tmp_path / "none.json")).bpf is False


def test_waterfall_uses_published_events(tmp_path):
    from kaia_dashboard import KaiamonSnapshot
    from core.web_bridge import serialize_snapshot
    path = str(tmp_path / tp.FILE_NAME)
    import time
    tp.TelemetryPublisher(FakeEngine(), path).publish_once(now=time.time())
    data = serialize_snapshot(KaiamonSnapshot(), ebpf_engine=tp.PublishedTelemetry(path))
    files = {e.get("filename") for e in data.get("waterfall_events", data.get("waterfall", []))}
    assert "/usr/bin/id" in files or any("/usr/bin/id" in str(v) for v in data.values())


def test_the_hud_never_loads_ebpf_itself():
    src = Path("core/web_bridge.py").read_text()
    assert "EBPFTelemetryEngine" not in src and "ebpf_engine.start()" not in src
    assert "PublishedTelemetry(" in src
    assert "TelemetryPublisher(ebpf," in Path("security/policy_gate.py").read_text()
