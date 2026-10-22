"""Regression tests for bounded eBPF honeypot event buffering."""

import os
from types import SimpleNamespace

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security.ebpf_telemetry import EBPFTelemetryEngine, HONEYPOT_HITS_MAX_ENTRIES


class FakePerfTable:
    def __init__(self, event):
        self._event = event

    def event(self, _data):
        return self._event


def _honeypot_event(number):
    return SimpleNamespace(
        pid=1000 + number,
        comm=b"attacker",
        filename=f"/tmp/kaia-decoy-{number}.txt".encode(),
    )


def test_honeypot_event_buffer_has_hard_capacity_and_tracks_overflow():
    engine = EBPFTelemetryEngine()

    for number in range(HONEYPOT_HITS_MAX_ENTRIES + 7):
        engine.bpf = {"honeypot_events": FakePerfTable(_honeypot_event(number))}
        engine._handle_honeypot_event(0, b"", 0)

    assert len(engine.honeypot_hits) == HONEYPOT_HITS_MAX_ENTRIES
    assert engine.honeypot_dropped_events == 7

    consumed = engine.get_honeypot_hits()
    assert len(consumed) == HONEYPOT_HITS_MAX_ENTRIES
    assert consumed[0]["pid"] == 1007
    assert consumed[-1]["pid"] == 1000 + HONEYPOT_HITS_MAX_ENTRIES + 6
    assert engine.honeypot_dropped_events == 0
    assert len(engine.honeypot_hits) == 0


def test_honeypot_buffer_restarts_empty_after_drain():
    engine = EBPFTelemetryEngine()
    engine.bpf = {"honeypot_events": FakePerfTable(_honeypot_event(1))}
    engine._handle_honeypot_event(0, b"", 0)

    assert len(engine.get_honeypot_hits()) == 1
    assert engine.get_honeypot_hits() == []
    assert engine.honeypot_dropped_events == 0
