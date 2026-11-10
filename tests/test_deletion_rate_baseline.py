"""Deletion bursts far above a program's learned peak are reported (wiper/ransomware shape)."""

import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security import behaviour_baseline as bb


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _b(tmp_path, events, clock):
    return bb.DeletionRateBaseline(str(tmp_path / bb.DELETE_FILE_NAME),
                                   log_event=lambda **kw: events.append(kw), clock=clock, learning_seconds=100)


def _burst(b, comm, n):
    return [b.observe(comm, 7, f"/home/u/doc{i}") for i in range(n)]


def test_learned_peaks_raise_the_threshold_and_bursts_report_once(tmp_path):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    _burst(b, "rm", 400)                      # learning: rm -rf of a build tree
    clock.t += 101
    _burst(b, "rm", 3000)                     # 10 x 400 = 4000 not reached
    assert events == []
    clock.t += 11
    out = _burst(b, "rm", 4500)
    assert out.count(bb.EVENT_MASS_DELETION) == 1 and len(events) == 1
    assert events[0]["event_type"] == "mass_deletion" and "learned peak 400" in events[0]["payload_hash"]


def test_a_program_never_seen_deleting_alerts_at_the_floor(tmp_path):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    clock.t += 101
    _burst(b, "evil", bb.DELETE_FLOOR - 1)
    assert events == []
    _burst(b, "evil", 1)
    assert len(events) == 1


def test_cooldown_and_windows(tmp_path):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    clock.t += 101
    _burst(b, "evil", bb.DELETE_FLOOR)
    clock.t += 11
    _burst(b, "evil", bb.DELETE_FLOOR)        # new window, still in cooldown
    assert len(events) == 1
    clock.t += bb.DELETE_REPORT_COOLDOWN
    _burst(b, "evil", bb.DELETE_FLOOR)
    assert len(events) == 2


def test_peaks_persist_and_learning_does_not_restart(tmp_path):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    _burst(b, "make", 50)
    b.persist(force=True)
    clock.t += 101
    again = _b(tmp_path, events, clock)
    assert again.peaks == {"make": 50} and not again.learning
    assert (tmp_path / bb.DELETE_FILE_NAME).stat().st_mode & 0o777 == 0o640


def test_engine_and_gate_wiring():
    from pathlib import Path
    assert 'rates.observe(record["comm"], record["pid"], record["filename"])' in Path("security/ebpf_telemetry.py").read_text()
    assert "ebpf.delete_baseline = DeletionRateBaseline(" in Path("security/policy_gate.py").read_text()
