"""Tests that FIM scans bytes from the original fanotify event descriptor."""

import hashlib
import os
import sqlite3

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

from security.fim_daemon import FIMDaemon, FAN_CLOSE_WRITE


class FakeYaraScanner:
    def __init__(self):
        self.scanned = []

    def match(self, *, data):
        self.scanned.append(data)
        return []


def _init_fim_db(path):
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE fim_events (
                timestamp TEXT, event_type TEXT, pid INTEGER, comm TEXT,
                path TEXT, yara_matches TEXT, sha256 TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_fim_scans_and_hashes_original_event_fd_not_replacement_path(tmp_path, monkeypatch):
    original = tmp_path / "original.sh"
    replacement = tmp_path / "replacement.sh"
    original_bytes = b"#!/bin/sh\necho original-inode\n"
    replacement.write_bytes(b"#!/bin/sh\necho replacement-path\n")
    original.write_bytes(original_bytes)

    scanner = FakeYaraScanner()
    fim = FIMDaemon(yara_scanner=scanner)
    fim.db_path = str(tmp_path / "fim.db")
    fim.exclude_prefixes = []  # tmp_path is intentionally outside the exclusion policy for this unit test
    _init_fim_db(fim.db_path)
    monkeypatch.setattr("security.fim_daemon.log_security_event", lambda **kwargs: None)

    event_fd = os.open(original, os.O_RDONLY)
    try:
        # A different current pathname models rename/path replacement after
        # notification; scan content must still come from the event descriptor.
        fim._handle_event(str(replacement), 12345, FAN_CLOSE_WRITE, event_fd)
    finally:
        os.close(event_fd)

    assert scanner.scanned == [original_bytes]
    conn = sqlite3.connect(fim.db_path)
    try:
        event_type, logged_path, sha = conn.execute(
            "SELECT event_type, path, sha256 FROM fim_events"
        ).fetchone()
    finally:
        conn.close()
    assert event_type == "close_write"
    assert logged_path == str(replacement)
    assert sha == hashlib.sha256(original_bytes).hexdigest()
    assert sha != hashlib.sha256(replacement.read_bytes()).hexdigest()


def test_fim_bounds_memory_for_large_file_and_records_scan_skip(tmp_path, monkeypatch):
    target = tmp_path / "large.bin"
    target.write_bytes(b"X" * 8192)
    scanner = FakeYaraScanner()
    fim = FIMDaemon(yara_scanner=scanner)
    fim.db_path = str(tmp_path / "fim.db")
    fim.exclude_prefixes = []  # tmp_path is intentionally outside the exclusion policy for this unit test
    _init_fim_db(fim.db_path)
    monkeypatch.setattr("security.fim_daemon.MAX_FIM_SCAN_BYTES", 1024)
    events = []
    monkeypatch.setattr("security.fim_daemon.log_security_event", lambda **kwargs: events.append(kwargs))

    fd = os.open(target, os.O_RDONLY)
    try:
        fim._handle_event(str(target), 4321, FAN_CLOSE_WRITE, fd)
    finally:
        os.close(fd)

    assert scanner.scanned == []
    assert events and events[0]["event_type"] == "fim_scan_skipped"
    conn = sqlite3.connect(fim.db_path)
    try:
        event_type, sha = conn.execute("SELECT event_type, sha256 FROM fim_events").fetchone()
    finally:
        conn.close()
    assert event_type == "scan_skipped_size_limit_exceeded"
    assert sha == ""
