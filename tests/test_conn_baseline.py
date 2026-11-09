"""First-seen outbound (process, port) pairs; shells and interpreters are high."""

import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import behaviour_baseline as bb
from security.alert_levels import level_of


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _b(tmp_path, events, clock):
    return bb.ConnBaseline(str(tmp_path / bb.CONN_FILE_NAME), log_event=lambda **kw: events.append(kw),
                           clock=clock, learning_seconds=10)


def test_pairs_are_learned_then_new_ones_reported_and_addresses_not_learned(tmp_path):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    b.observe(bb.ConnBaseline.describe("firefox", "198.51.100.1", 443))
    clock.t += 11
    assert b.observe(bb.ConnBaseline.describe("firefox", "203.0.113.50", 443)) is None   # new IP, known pair
    assert b.observe(bb.ConnBaseline.describe("firefox", "203.0.113.50", 8443)) == bb.EVENT_NET_FIRST_SEEN
    assert events[0]["payload_hash"] == "firefox|8443|203.0.113.50"


@pytest.mark.parametrize("comm", ["bash", "sh", "zsh", "nc", "ncat", "socat", "python3", "python3.14", "perl", "php8", "node"])
def test_shells_and_interpreters_on_a_new_port_are_high(tmp_path, comm):
    events, clock = [], Clock()
    b = _b(tmp_path, events, clock)
    clock.t += 11
    assert b.observe(bb.ConnBaseline.describe(comm, "203.0.113.9", 4444)) == bb.EVENT_NET_SHELL
    assert level_of(bb.EVENT_NET_SHELL)[0] == "high" and level_of(bb.EVENT_NET_FIRST_SEEN)[0] == "low"


def test_ordinary_programs_are_low(tmp_path):
    for comm in ("firefox", "pacman", "ollama", "sshd", "bashful"):
        assert bb.ConnBaseline(None).judge(bb.ConnBaseline.describe(comm, "203.0.113.9", 1)) == bb.EVENT_NET_FIRST_SEEN


def test_engine_skips_loopback_and_guards_the_baseline():
    from pathlib import Path
    src = Path("security/ebpf_telemetry.py").read_text()
    assert 'not clean_daddr.startswith("127.")' in src
    assert 'logger.error("Connection baseline failed: %s", e)' in src
    gate = Path("security/policy_gate.py").read_text()
    assert "ebpf.conn_baseline = ConnBaseline(" in gate
