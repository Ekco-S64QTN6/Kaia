"""Regression tests for privileged Script Sentinel file inspection."""

import hashlib
import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test-sentinel-secret")

import pytest
from security import telemetry_daemon


def test_script_sentinel_hashes_large_scripts_with_bounded_reads(tmp_path, monkeypatch):
    script = tmp_path / "sample.sh"
    contents = b"#!/bin/sh\n" + (b"# bounded streaming test\n" * 100_000)
    script.write_bytes(contents)

    real_read = os.read
    requested_sizes = []

    def bounded_read(fd, size):
        requested_sizes.append(size)
        assert size <= telemetry_daemon.SCRIPT_SENTINEL_CHUNK_BYTES
        return real_read(fd, size)

    monkeypatch.setattr(os, "read", bounded_read)
    digest = telemetry_daemon.ScriptSentinelHandler._hash_script_no_follow(str(script))

    assert digest == hashlib.sha256(contents).hexdigest()
    assert requested_sizes
    assert max(requested_sizes) <= telemetry_daemon.SCRIPT_SENTINEL_CHUNK_BYTES


def test_script_sentinel_never_follows_script_symlinks(tmp_path):
    target = tmp_path / "secret.txt"
    target.write_bytes(b"do not hash the symlink target")
    link = tmp_path / "suspect.sh"
    link.symlink_to(target)

    digest = telemetry_daemon.ScriptSentinelHandler._hash_script_no_follow(str(link))

    assert digest == "unknown"
    assert target.read_bytes() == b"do not hash the symlink target"


def test_script_sentinel_never_reads_oversized_script(tmp_path, monkeypatch):
    script = tmp_path / "oversized.sh"
    script.write_bytes(b"#!/bin/sh\n")
    with script.open("r+b") as stream:
        stream.truncate(telemetry_daemon.SCRIPT_SENTINEL_MAX_BYTES + 1)

    monkeypatch.setattr(
        os,
        "read",
        lambda *args, **kwargs: pytest.fail("oversized scripts must not be read"),
    )
    assert telemetry_daemon.ScriptSentinelHandler._hash_script_no_follow(str(script)) == "unknown"


def test_script_sentinel_ignores_non_script_regular_files(tmp_path):
    data = tmp_path / "notes.txt"
    data.write_bytes(b"ordinary text without a shebang")

    assert telemetry_daemon.ScriptSentinelHandler._hash_script_no_follow(str(data)) is None


def test_script_sentinel_does_not_block_on_fifo(tmp_path):
    pipe = tmp_path / "suspect.sh"
    os.mkfifo(pipe)

    assert telemetry_daemon.ScriptSentinelHandler._hash_script_no_follow(str(pipe)) == "unknown"
