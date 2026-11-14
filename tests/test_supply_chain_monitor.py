"""Tests for passive, transaction-aware pacman package monitoring."""

import os
import json
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
    # First startup baselines EOF, so the transaction must start afterwards.
    log.write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kwargs: events.append(kwargs))
    first = PacmanLogMonitor(str(log), str(cursor))
    first.poll_once()

    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-10-22T11:00:00+0000", "transaction started"))
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


def test_empty_rotated_log_persists_new_inode_without_repeated_resets(tmp_path, monkeypatch, caplog):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text("", encoding="utf-8")
    monitor = PacmanLogMonitor(str(log), str(cursor))
    monitor.poll_once()

    rotated = tmp_path / "pacman.log.1"
    log.rename(rotated)
    log.write_text("", encoding="utf-8")

    assert monitor.poll_once() == 0
    state = json.loads(cursor.read_text(encoding="utf-8"))
    assert state["inode"] == log.stat().st_ino
    assert state["offset"] == 0

    caplog.clear()
    assert monitor.poll_once() == 0
    assert "Pacman log rotated or truncated" not in caplog.text


def test_cursor_file_is_created_with_private_permissions(tmp_path, monkeypatch):
    log = tmp_path / "pacman.log"
    cursor = tmp_path / "state" / "pacman.cursor.json"
    log.write_text("", encoding="utf-8")
    monitor = PacmanLogMonitor(str(log), str(cursor))
    monitor.poll_once()
    assert cursor.stat().st_mode & 0o777 == 0o600


def test_unreadable_log_is_degraded_in_control_health_not_active(tmp_path, monkeypatch):
    """Polls that cannot read pacman.log used to leave the monitor reported active."""
    from security import control_health
    from security.supply_chain_monitor import PacmanLogMonitor
    seen = []
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.append((name, state, detail)))
    log, cursor = tmp_path / "pacman.log", tmp_path / "cursor.json"
    monitor = PacmanLogMonitor(str(log), str(cursor), poll_interval=1)
    rounds = iter([False, False, True])

    class Stop:
        def is_set(self):
            return False

        def wait(self, _):
            done = next(rounds)
            if not done:
                log.write_text("[2026-11-09T10:00:00+0000] [ALPM] transaction started\n")
            Stop.is_set = lambda self: done
            return done

    monitor._stop = Stop()
    monitor._run()
    states = [(s, d) for n, s, d in seen if n == "pacman_monitor"]
    assert states[0][0] == "degraded" and "not found" in states[0][1]
    assert states[-1][0] == "active"


def _started_monitor(tmp_path, monkeypatch, **kwargs):
    log = tmp_path / "pacman.log"
    log.write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr("security.supply_chain_monitor.log_security_event", lambda **kw: events.append(kw))
    monitor = PacmanLogMonitor(str(log), str(tmp_path / "state" / "cursor.json"), poll_interval=1, **kwargs)
    assert monitor.poll_once() == 0
    return log, monitor, events


def test_full_system_upgrade_is_recorded(tmp_path, monkeypatch):
    # The owner's largest real transaction had 476 package actions. The cap was
    # 256 and a transaction over it was dropped whole.
    log, monitor, events = _started_monitor(tmp_path, monkeypatch)
    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-11-01T10:00:00+0000", "transaction started"))
        for i in range(476):
            f.write(_line("2026-11-01T10:00:01+0000", f"upgraded pkg{i} (1.0-1 -> 1.0-2)"))
        f.write(_line("2026-11-01T10:05:00+0000", "transaction completed"))

    assert monitor.poll_once() == 476
    assert len([e for e in events if e["event_type"] == "pacman_package_upgraded"]) == 476


def test_transaction_over_the_cap_leaves_a_ledger_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("security.supply_chain_monitor.MAX_PENDING_PACKAGES", 3)
    log, monitor, events = _started_monitor(tmp_path, monkeypatch)
    with log.open("a", encoding="utf-8") as f:
        f.write(_line("2026-11-01T10:00:00+0000", "transaction started"))
        for i in range(5):
            f.write(_line("2026-11-01T10:00:01+0000", f"installed pkg{i} (1.0-1)"))
        f.write(_line("2026-11-01T10:05:00+0000", "transaction completed"))

    assert monitor.poll_once() == 0
    assert [e["event_type"] for e in events if "too_large" in e["event_type"]] == [
        "pacman_monitor_transaction_too_large"]


def test_line_longer_than_one_read_does_not_stop_the_monitor(tmp_path, monkeypatch):
    log, monitor, events = _started_monitor(tmp_path, monkeypatch, max_bytes_per_poll=4096)
    with log.open("a", encoding="utf-8") as f:
        f.write("x" * 5000 + "\n")
        f.write(_line("2026-11-01T10:00:00+0000", "transaction started"))
        f.write(_line("2026-11-01T10:00:01+0000", "installed after-junk (1.0-1)"))
        f.write(_line("2026-11-01T10:00:02+0000", "transaction completed"))

    assert sum(monitor.poll_once() for _ in range(4)) == 1
    assert [e["actor"] for e in events if e["event_type"] == "pacman_package_installed"] == ["after-junk (1.0-1)"]
