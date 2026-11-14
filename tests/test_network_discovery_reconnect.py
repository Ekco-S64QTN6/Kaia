"""Passive discovery must survive losing its capture socket.

On the host the gate starts before Wi-Fi is up: recvfrom() raised ENETDOWN
once, the capture thread ended and discovery stayed "failed" for four days.
"""
import errno
import socket

from security import control_health
from security.network_discovery import PassiveDiscoveryEngine


class FakeSocket:
    def __init__(self, script):
        self.script = script
        self.closed = False

    def recvfrom(self, _size):
        return self.script(self)

    def close(self):
        self.closed = True


def _engine(monkeypatch):
    engine = PassiveDiscoveryEngine()
    engine.CAPTURE_RETRY_SECONDS = 0.0
    states = []
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: states.append((state, detail)))
    monkeypatch.setattr(control_health, "heartbeat", lambda name: None)
    return engine, states


def test_capture_reopens_after_network_down(monkeypatch):
    engine, states = _engine(monkeypatch)
    opened = []

    def first(sock):
        raise OSError(errno.ENETDOWN, "Network is down")

    def second(sock):
        engine.stop_event.set()
        raise socket.timeout()

    scripts = [first, second]

    def open_socket():
        sock = FakeSocket(scripts[len(opened)])
        opened.append(sock)
        return sock, "wlan-test"

    monkeypatch.setattr(engine, "_open_capture_socket", open_socket)
    engine._run()

    assert len(opened) == 2, "the capture socket was not reopened"
    assert opened[0].closed
    assert [s for s, _ in states] == ["active", "degraded", "active"]
    assert "Network is down" in states[1][1]


def test_open_failure_is_retried_and_reported_once(monkeypatch):
    engine, states = _engine(monkeypatch)
    attempts = []

    def open_socket():
        attempts.append(1)
        if len(attempts) < 4:
            raise OSError(errno.ENODEV, "No such device")
        engine.stop_event.set()
        return FakeSocket(lambda s: (_ for _ in ()).throw(socket.timeout())), "wlan-test"

    monkeypatch.setattr(engine, "_open_capture_socket", open_socket)
    engine._run()

    assert len(attempts) == 4
    # Three identical failures are one report, then recovery.
    assert [s for s, _ in states] == ["degraded", "active"]


def test_missing_capability_is_failed_and_not_retried(monkeypatch):
    engine, states = _engine(monkeypatch)
    attempts = []

    def open_socket():
        attempts.append(1)
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(engine, "_open_capture_socket", open_socket)
    engine._run()

    assert len(attempts) == 1
    assert [s for s, _ in states] == ["failed"]
