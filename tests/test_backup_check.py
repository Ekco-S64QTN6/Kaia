"""Backup freshness from the mirror's ctime; restorability from unchanged samples."""

import os
import time

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import backup_check as bc


@pytest.fixture
def system(tmp_path, monkeypatch):
    live, backup = tmp_path / "live", tmp_path / "backup"
    for d in ("etc", "usr", "var"):
        (live / d).mkdir(parents=True)
        (backup / d).mkdir(parents=True)
    (live / "etc/fstab").write_text("UUID=x / ext4\n")
    (backup / "etc/fstab").write_text("UUID=x / ext4\n")
    monkeypatch.setattr(bc, "SAMPLE_FILES", ("/etc/fstab",))
    monkeypatch.setattr(bc.os.path, "ismount", lambda p: p == str(backup))
    return live, backup


def _check(system, now=None):
    live, backup = system
    return bc.check(str(backup), live_root=str(live), now=now)


def test_fresh_matching_backup_is_active(system):
    state, detail = _check(system)
    assert state == "active" and "1 of 1 unchanged sample files match" in detail


def test_stale_backups_degrade_then_fail(system):
    now = time.time()
    assert _check(system, now + 15 * 86400)[0] == "degraded"
    assert _check(system, now + 31 * 86400)[0] == "failed"


def test_a_corrupt_copy_of_an_unchanged_file_is_degraded(system):
    live, backup = system
    os.utime(live / "etc/fstab", (time.time() - 10, time.time() - 10))
    (backup / "etc/fstab").write_text("UUID=x / ext4\n")  # ensure ctime >= live ctime
    (backup / "etc/fstab").write_bytes(b"\x00garbage")
    # the live file's ctime is older than the mirror's newest marker dir ctime
    os.utime(backup / "etc", None)
    os.chmod(backup / "etc", 0o755)
    state, detail = _check(system)
    assert state == "degraded" and "/etc/fstab" in detail


def test_files_changed_since_the_backup_are_not_compared(system):
    live, backup = system
    os.chmod(backup / "etc", 0o755)          # backup marker ctime: now
    time.sleep(0.01)
    (live / "etc/fstab").write_text("changed after the backup\n")
    state, detail = _check(system)
    assert state == "active" and "0 of 0" in detail


def test_unmounted_or_missing_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.os.path, "ismount", lambda p: False)
    (tmp_path / "b").mkdir()
    assert bc.check(str(tmp_path / "b"))[0] == "failed"
    assert bc.check(str(tmp_path / "nope"))[0] == "off"
    assert bc.check(str(tmp_path / "nope"), explicit=True)[0] == "failed"


def test_gate_runs_the_monitor():
    from pathlib import Path
    from security import control_health
    assert "BackupMonitor()" in Path("security/policy_gate.py").read_text()
    assert "backup" in control_health.GATE_CONTROLS
