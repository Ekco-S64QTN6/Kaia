"""First-seen executables after a learning period, alert-only (Tier 5 behavioural baseline)."""

import json
import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import behaviour_baseline as bb
from security.alert_levels import level_of


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _baseline(tmp_path, events, clock):
    return bb.ExecBaseline(str(tmp_path / bb.FILE_NAME), log_event=lambda **kw: events.append(kw),
                           clock=clock, learning_seconds=100)


def test_learning_is_silent_then_new_paths_are_reported_once(tmp_path):
    events, clock = [], Clock()
    b = _baseline(tmp_path, events, clock)
    assert b.observe("/usr/bin/ls", "bash", 1) is None          # learning
    clock.t += 101
    assert b.observe("/usr/bin/ls", "bash", 2) is None          # known
    assert b.observe("/usr/bin/newtool", "bash", 3) == bb.EVENT_FIRST_SEEN
    assert b.observe("/usr/bin/newtool", "bash", 4) is None     # reported once
    assert [e["event_type"] for e in events] == [bb.EVENT_FIRST_SEEN]
    assert events[0]["payload_hash"] == "/usr/bin/newtool" and events[0]["disposition"] == "observed"


@pytest.mark.parametrize("path", ["/tmp/x", "/dev/shm/.k", "/home/u/.cache/a", "/root/a", "./a", "a",
                                  "/usr/bin/../../tmp/x", "/run/user/1000/x", "/var/tmp/y"])
def test_writable_or_relative_locations_are_high(path):
    assert bb.classify(path) == bb.EVENT_UNUSUAL
    assert level_of(bb.EVENT_UNUSUAL)[0] == "high" and level_of(bb.EVENT_FIRST_SEEN)[0] == "low"


def test_reports_are_rate_capped(tmp_path):
    events, clock = [], Clock()
    b = _baseline(tmp_path, events, clock)
    clock.t += 101
    for i in range(bb.MAX_REPORTS_PER_MINUTE + 20):
        b.observe(f"/usr/bin/t{i}")
    assert len(events) == bb.MAX_REPORTS_PER_MINUTE and b.suppressed == 20
    assert "suppressed" in b.health()[1]


def test_state_survives_a_restart_and_is_private(tmp_path):
    events, clock = [], Clock()
    b = _baseline(tmp_path, events, clock)
    b.observe("/usr/bin/a")
    b.persist(force=True)
    path = tmp_path / bb.FILE_NAME
    assert (path.stat().st_mode & 0o777) == 0o640 and "/usr/bin/a" in json.loads(path.read_text())["known"]
    clock.t += 101
    again = _baseline(tmp_path, events, clock)
    assert again.started_at == 1000.0 and not again.learning          # learning not restarted
    assert again.observe("/usr/bin/a") is None


def test_corrupt_or_symlinked_state_restarts_learning_safely(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text('{"started_at": 0, "known": []}')
    (tmp_path / bb.FILE_NAME).symlink_to(victim)
    b = _baseline(tmp_path, [], Clock())
    assert b.learning and b.known == set()


def test_engine_feeds_the_baseline_and_never_lets_it_break_telemetry():
    from pathlib import Path
    src = Path("security/ebpf_telemetry.py").read_text()
    assert "baseline.observe(clean_filename, clean_comm, clean_pid)" in src
    assert 'logger.error("Exec baseline failed: %s", e)' in src
    gate = Path("security/policy_gate.py").read_text()
    assert "ebpf.exec_baseline = ExecBaseline(" in gate and "persist(force=True)" in gate
    assert "trigger_lockdown" not in Path(bb.__file__).read_text()


def test_random_build_directories_collapse_so_builds_are_learned_once(tmp_path):
    assert bb.baseline_key("/tmp/pip-build-env-ab12cd34/overlay/bin/cmake") == "/tmp/*/overlay/bin/cmake"
    assert bb.baseline_key("/tmp/tmpk3j9x2q1/conftest") == "/tmp/*/conftest"
    assert bb.baseline_key("/home/u/.cache/yay/pkg/src/build-1a2b3c/cc1") == "/home/u/.cache/yay/pkg/src/*/cc1"
    assert bb.baseline_key("/usr/bin/ls") == "/usr/bin/ls"
    events, clock = [], Clock()
    b = _baseline(tmp_path, events, clock)
    clock.t += 101
    b.observe("/tmp/pip-build-env-aaaa1111/overlay/bin/cmake")
    b.observe("/tmp/pip-build-env-bbbb2222/overlay/bin/cmake")
    assert len(events) == 1 and events[0]["event_type"] == bb.EVENT_UNUSUAL
