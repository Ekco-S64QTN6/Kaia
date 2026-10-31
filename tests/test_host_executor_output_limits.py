"""Regression tests for bounded HostExecutor subprocess output capture."""

import os
import sys

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_host_executor_output_secret")

from security import host_executor
from security.host_executor import HostExecutor


def test_run_cmd_terminates_process_group_when_combined_output_exceeds_limit(monkeypatch):
    monkeypatch.setattr(
        host_executor,
        "ALLOWED_EXECUTABLES",
        host_executor.ALLOWED_EXECUTABLES | {sys.executable},
    )
    monkeypatch.setattr(host_executor, "MAX_SUBPROCESS_OUTPUT_BYTES", 8192)
    command = [
        sys.executable,
        "-c",
        "import os; chunk=b'x'*4096; "
        "[(os.write(1, chunk), os.write(2, chunk)) for _ in range(100)]",
    ]

    ok, stdout, stderr = HostExecutor._run_cmd(command)

    assert not ok
    assert len(stdout.encode("utf-8")) <= host_executor.MAX_SUBPROCESS_OUTPUT_BYTES
    assert "output exceeded" in stderr.lower()
    assert "process group was terminated" in stderr.lower()


def test_run_cmd_keeps_small_successful_output_unchanged(monkeypatch):
    monkeypatch.setattr(
        host_executor,
        "ALLOWED_EXECUTABLES",
        host_executor.ALLOWED_EXECUTABLES | {sys.executable},
    )
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write(' diagnostic-ok '); sys.stderr.write(' harmless-warning ')",
    ]

    ok, stdout, stderr = HostExecutor._run_cmd(command)

    assert ok
    assert stdout == "diagnostic-ok"
    assert stderr == "harmless-warning"


def test_run_cmd_rejects_relative_executable_before_spawning():
    ok, stdout, stderr = HostExecutor._run_cmd(["python", "-c", "pass"])

    assert not ok
    assert stdout == ""
    assert "absolute executable path" in stderr.lower()
