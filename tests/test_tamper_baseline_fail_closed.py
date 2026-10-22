import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

"""Regression tests for integrity baseline trust boundaries."""

import sqlite3

import pytest

import config
from security import tamper_detection as td


def test_existing_malformed_baseline_does_not_fall_back_to_disk_hashes(tmp_path, monkeypatch):
    path = tmp_path / "baseline.sha256"
    path.write_text("not-a-digest  /opt/kaia/security/policy_gate.py\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRUSTED_BASELINE_PATH", str(path))

    with pytest.raises(RuntimeError, match="refusing to start"):
        td.TamperDetector()._load_trusted_baseline()


def test_sqlite_prefix_hash_cannot_be_aliased_by_delimiter_text(tmp_path):
    path = tmp_path / "security_events.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE security_events (
                event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, type TEXT NOT NULL,
                source TEXT NOT NULL, actor TEXT NOT NULL, payload_hash TEXT,
                disposition TEXT NOT NULL, session_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO security_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("evt1", "now", "event", "unit", "alice|bob", "xyz", "observed", "test"),
        )
        conn.commit()
    finally:
        conn.close()

    detector = td.TamperDetector()
    before = detector._compute_sqlite_prefix_hash(str(path), 1)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE security_events SET actor = ?, payload_hash = ? WHERE event_id = ?",
            ("alice", "bob|xyz", "evt1"),
        )
        conn.commit()
    finally:
        conn.close()

    after = detector._compute_sqlite_prefix_hash(str(path), 1)
    assert before and after
    assert before != after


def test_append_only_ledger_is_created_and_baselined_when_initially_absent(tmp_path, monkeypatch):
    audit_path = tmp_path / "state" / "audit_ledger.json"
    monkeypatch.setattr(config, "TRUSTED_BASELINE_PATH", str(tmp_path / "missing-baseline"))
    detector = td.TamperDetector()
    protected = tmp_path / "protected.py"
    protected.write_text("trusted code", encoding="utf-8")
    detector.immutable_files = [str(protected)]
    detector.append_only_files = [str(audit_path)]
    detector.sqlite_append_only_files = []
    monkeypatch.setattr(detector, "_run", lambda: None)

    detector.start()
    try:
        assert audit_path.exists()
        digest, size = detector._baselines[str(audit_path)]
        assert size == 0
        assert digest
    finally:
        detector.stop()


def test_installed_signing_secret_is_in_the_integrity_watchlist(monkeypatch):
    real_exists = td.os.path.exists
    monkeypatch.setattr(
        td.os.path, "exists",
        lambda path: True if path == "/etc/kaia/secret.env" else real_exists(path),
    )
    detector = td.TamperDetector()
    assert "/etc/kaia/secret.env" in detector.immutable_files


def test_missing_trusted_protected_file_alerts_before_startup_aborts(tmp_path, monkeypatch):
    target = tmp_path / "security" / "missing.py"
    target.parent.mkdir()
    baseline = tmp_path / "baseline.sha256"
    baseline.write_text(f"{'0' * 64}  {target}\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRUSTED_BASELINE_PATH", str(baseline))
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(tmp_path))

    detector = td.TamperDetector()
    detector.immutable_files = [str(target)]
    detector.append_only_files = []
    detector.sqlite_append_only_files = []
    events = []
    monkeypatch.setattr(td, "log_security_event", lambda **kwargs: events.append(kwargs))
    import security.policy_gate as policy_gate_module
    lockdowns = []
    monkeypatch.setattr(policy_gate_module, "trigger_lockdown", lambda reason: lockdowns.append(reason))

    with pytest.raises(RuntimeError, match="missing"):
        detector.start()

    assert events and events[0]["event_type"] == "tamper_detected"
    assert lockdowns


def test_system_mode_refuses_to_bootstrap_a_missing_trusted_baseline(tmp_path, monkeypatch):
    from security import tamper_detection as td

    baseline = tmp_path / "missing-baseline.sha256"
    monkeypatch.setattr(config, "TRUSTED_BASELINE_PATH", str(baseline))
    monkeypatch.setattr(config, "STATE_DIR_IS_SYSTEM", True)
    detector = td.TamperDetector()

    with pytest.raises(RuntimeError, match="Trusted integrity baseline is missing"):
        detector.start()


def test_detector_fails_closed_when_security_watchlist_cannot_be_enumerated(monkeypatch):
    real_listdir = td.os.listdir
    security_dir = os.path.abspath(os.path.join(config.WORKSPACE_DIR, "security"))

    def fail_security_listdir(path):
        if os.path.abspath(path) == security_dir:
            raise PermissionError("simulated directory enumeration failure")
        return real_listdir(path)

    monkeypatch.setattr(td.os, "listdir", fail_security_listdir)
    with pytest.raises(RuntimeError, match="Could not enumerate protected security modules"):
        td.TamperDetector()


def test_baseline_script_propagates_watchlist_enumeration_failure():
    script = open("scripts/kaia-baseline.sh", encoding="utf-8").read()

    # Process substitution masks the command's non-zero exit status in Bash.
    # Capture into a checked command group before reading any paths instead.
    assert 'files_tmp="$(mktemp)"' in script
    assert 'if ! (' in script
    assert ') > "$files_tmp"; then' in script
    assert 'mapfile -t FILES < "$files_tmp"' in script
    assert "mapfile -t FILES < <(" not in script
    assert 'if [ -n "$files_tmp" ] && [ -e "$files_tmp" ]; then rm -f -- "$files_tmp"; fi' in script



def test_baseline_script_refuses_missing_or_nonregular_watchlist_paths():
    script = open("scripts/kaia-baseline.sh", encoding="utf-8").read()

    assert "metadata = os.lstat(f)" in script
    assert 'if os.path.basename(f) == ".env":' in script
    assert "Missing protected file in integrity watchlist" in script
    assert "Protected watchlist path is not a regular file" in script
    assert "if os.path.exists(f):" not in script


def test_compute_sha256_reads_bounded_chunks_and_preserves_full_digest(tmp_path, monkeypatch):
    import hashlib

    payload = bytes(range(256)) * 16_384
    target = tmp_path / "large-protected-file.bin"
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    read_sizes = []
    real_read = os.read

    def bounded_read(fd, size):
        assert size > 0, "unbounded file reads are forbidden in integrity hashing"
        assert size <= 1024 * 1024, "hash chunk exceeds the configured memory bound"
        read_sizes.append(size)
        return real_read(fd, size)

    monkeypatch.setattr(td.os, "read", bounded_read)
    digest = td.TamperDetector()._compute_sha256(str(target))

    assert digest == expected
    assert len(read_sizes) > 1
    assert max(read_sizes) == 1024 * 1024


def test_compute_sha256_limit_hashes_only_the_requested_prefix(tmp_path):
    import hashlib

    payload = b"protected-prefix" + b"x" * 100_000
    target = tmp_path / "prefix.bin"
    target.write_bytes(payload)

    digest = td.TamperDetector()._compute_sha256(str(target), limit=16)

    assert digest == hashlib.sha256(payload[:16]).hexdigest()


def test_compute_sha256_refuses_final_symlink_and_fifo_without_blocking(tmp_path):
    import hashlib
    import os

    target = tmp_path / "real.bin"
    target.write_bytes(b"trusted external target")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(target)
    fifo = tmp_path / "watch.fifo"
    os.mkfifo(fifo)

    detector = td.TamperDetector()

    assert detector._compute_sha256(str(alias)) == ""
    assert detector._compute_sha256(str(fifo)) == ""
    assert target.read_bytes() == b"trusted external target"
    assert detector._compute_sha256(str(target)) == hashlib.sha256(
        b"trusted external target"
    ).hexdigest()
