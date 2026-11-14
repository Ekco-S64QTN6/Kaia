"""The honeypot's automatic block must reach the executor and report its outcome.

Its request had no ``justification``, which MitigationRequest requires: the gate
refused every auto-block as a schema failure, the sender never read the reply
and logged "dispatched successfully". No decoy read ever blocked anyone.
"""
import json
import os
import socket
import threading

import pytest

import config
from security import honeypot
from security.honeypot import HoneypotCoordinator
from security.policy_gate import PolicyGate, generate_capability_token
from security.schemas import MitigationRequest


def test_auto_block_request_satisfies_the_gate_schema():
    token = generate_capability_token("block_ip", "203.0.113.9", duration_seconds=30)
    envelope = HoneypotCoordinator._auto_block_request("203.0.113.9", token)
    # The gate flattens the envelope exactly like this before validating.
    flat = {**envelope["payload"], "action": envelope["action"],
            "capability_token": envelope["capability_token"]}
    request = MitigationRequest(**flat)
    assert request.target_ip == "203.0.113.9" and request.protocol == "all"
    assert not {"action", "capability_token"} & set(envelope["payload"])


class _FakeGate:
    """Serves one framed request with PolicyGate's own handler."""

    def __init__(self, tmp_path, gate):
        self.path = str(tmp_path / "gate.sock")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.gate = gate
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.server.accept()
        self.gate._handle_client(conn)

    def close(self):
        self.thread.join(timeout=5)
        self.server.close()


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(tmp_path / "ledger.json"))
    import security.db
    security.db.initialize_db()
    events = []
    monkeypatch.setattr(honeypot, "log_security_event",
                        lambda **kw: events.append((kw["event_type"], kw["actor"], kw["disposition"])))
    calls = []
    fake = _FakeGate(tmp_path, PolicyGate())
    monkeypatch.setattr(config, "POLICY_GATE_SOCKET", fake.path)
    yield calls, events
    fake.close()


def test_auto_block_goes_through_the_real_gate_to_the_executor(wired, monkeypatch):
    calls, events = wired
    from security.host_executor import HostExecutor
    monkeypatch.setattr(HostExecutor, "execute_mitigation",
                        staticmethod(lambda ip, proto, port=None: (calls.append((ip, proto, port)) or True, "", "")))
    assert HoneypotCoordinator()._dispatch_block_ip("203.0.113.9") is True
    assert calls == [("203.0.113.9", "all", None)]
    assert events == [("honeypot_auto_block", "203.0.113.9", "blocked")]
    ledger = [json.loads(line) for line in open(config.AUDIT_LOG_PATH)]
    assert ledger[-1]["result"] == "approved" and ledger[-1]["request"]["target_ip"] == "203.0.113.9"


def test_a_refused_auto_block_is_reported_not_claimed(wired, monkeypatch):
    calls, events = wired
    from security.host_executor import HostExecutor
    monkeypatch.setattr(HostExecutor, "execute_mitigation",
                        staticmethod(lambda ip, proto, port=None: (False, "", f"Refusing to block {ip}: a gateway of this host")))
    assert HoneypotCoordinator()._dispatch_block_ip("192.0.2.1") is False
    assert events == [("honeypot_auto_block_failed", "192.0.2.1", "failed")]


def test_an_unreachable_gate_is_reported(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(honeypot, "log_security_event", lambda **kw: events.append(kw["event_type"]))
    monkeypatch.setattr(config, "POLICY_GATE_SOCKET", str(tmp_path / "absent.sock"))
    assert HoneypotCoordinator()._dispatch_block_ip("203.0.113.9") is False
    assert events == ["honeypot_auto_block_failed"]
