"""Tests for passive, transaction-aware pacman package monitoring."""

import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security.supply_chain_monitor import PacmanLogMonitor


def _line(stamp, message):
    return f"[{stamp}] [ALPM] {message}\n"


def test_first_poll_baselines_history_and_new_completed_transaction_is_recorded(tmp_path, monkeypatch):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text(
        _line("2026-10-01T00:00:00+0000", "transaction started") +
        _line("2026-10-01T00:00:01+0000", "installed old-package (1.0-1)") +
        _line("2026-10-01T00:00:02+0000", "transaction completed"),
        encoding="utf-8",
    )
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kwargs: events.append(kwargs))
    monitor = PacmanLogMonitor(str(log), str(cursor), poll_interval=1)

    assert monitor.poll_once() == 0
    assert not [e for e in events if e["event_type"].startswith("pacman_package_")]

    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-10-22T11:00:00+0000", "transaction started"))
        f.write(_line("2026-10-22T11:00:01+0000", "upgraded openssl (3.3.1-1 -> 3.3.2-1)"))
        f.write(_line("2026-10-22T11:00:02+0000", "installed new-tool (2.0-1)"))
        f.write(_line("2026-10-22T11:00:03+0000", "transaction completed"))

    assert monitor.poll_once() == 2
    actions = [e for e in events if e["event_type"].startswith("pacman_package_")]
    assert [e["event_type"] for e in actions] == ["pacman_package_upgraded", "pacman_package_installed"]
    assert actions[0]["actor"].startswith("openssl")
    assert all(e["source"] == "pacman_log" for e in actions)
    assert all(len(e["payload_hash"]) == 64 for e in actions)


def test_incomplete_transaction_survives_monitor_restart(tmp_path, monkeypatch):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text(_line("2026-10-22T11:00:00+0000", "transaction started"), encoding="utf-8")
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kwargs: events.append(kwargs))
    first = PacmanLogMonitor(str(log), str(cursor))
    first.poll_once()

    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-10-22T11:00:01+0000", "removed old-package (1.0-1)"))
    assert first.poll_once() == 0
    restarted = PacmanLogMonitor(str(log), str(cursor))
    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-10-22T11:00:02+0000", "transaction completed"))

    assert restarted.poll_once() == 1
    actions = [e for e in events if e["event_type"].startswith("pacman_package_")]
    assert len(actions) == 1
    assert actions[0]["event_type"] == "pacman_package_removed"


def test_failed_transaction_does_not_emit_package_change_events(tmp_path, monkeypatch):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kwargs: events.append(kwargs))
    monitor = PacmanLogMonitor(str(log), str(cursor))
    monitor.poll_once()
    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-10-22T11:00:00+0000", "transaction started"))
        f.write(_line("2026-10-22T11:00:01+0000", "installed bad-package (1.0-1)"))
        f.write(_line("2026-10-22T11:00:02+0000", "transaction failed"))
    assert monitor.poll_once() == 0
    assert not [e for e in events if e["event_type"].startswith("pacman_package_")]


def test_log_rotation_tracks_new_inode_without_replaying_old_history(tmp_path, monkeypatch):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kwargs: events.append(kwargs))
    monitor = PacmanLogMonitor(str(log), str(cursor))
    monitor.poll_once()

    rotated = tmp_path / "pacman.log.1"
    log.rename(rotated)
    log.write_text(
        _line("2026-10-22T12:00:00+0000", "transaction started") +
        _line("2026-10-22T12:00:01+0000", "installed rotated-package (1.0-1)") +
        _line("2026-10-22T12:00:02+0000", "transaction completed"),
        encoding="utf-8",
    )
    assert monitor.poll_once() == 1
    assert any(e["actor"].startswith("rotated-package") for e in events)


def test_symlink_pacman_log_is_rejected(tmp_path, monkeypatch):
    real_log = tmp_path / "real.log"
    real_log.write_text("", encoding="utf-8")
    link = tmp_path / "pacman.log"
    link.symlink_to(real_log)
    monitor = PacmanLogMonitor(str(link), str(tmp_path / "cursor.json"))
    assert monitor.poll_once() == 0
