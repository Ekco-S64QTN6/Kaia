"""Regression tests for the privileged execution boundary."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

from security.host_executor import HostExecutor, SYSTEM_BINARIES


@pytest.mark.parametrize("command", [
    SYSTEM_BINARIES["ss"], SYSTEM_BINARIES["ip"], SYSTEM_BINARIES["nft"],
    SYSTEM_BINARIES["sudo"], SYSTEM_BINARIES["systemctl"],
    SYSTEM_BINARIES["unshare"], SYSTEM_BINARIES["bwrap"],
    SYSTEM_BINARIES["systemd-run"], SYSTEM_BINARIES["systemd-nspawn"],
])
def test_registered_host_binaries_are_absolute(command):
    assert os.path.isabs(command)


def test_run_cmd_rejects_path_lookup(monkeypatch):
    monkeypatch.setattr("security.host_executor.subprocess.run", lambda *a, **k: pytest.fail("must not launch"))
    ok, _, error = HostExecutor._run_cmd(["systemctl", "restart", "kaia-policy-gate"])
    assert not ok
    assert "absolute executable path" in error


def test_run_cmd_does_not_inherit_capability_signing_secret(monkeypatch):
    monkeypatch.setenv("KAIA_CAPABILITY_TOKEN_SECRET", "must-not-reach-children")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return MagicMock(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("security.host_executor.subprocess.run", fake_run)
    ok, stdout, _ = HostExecutor._run_cmd([SYSTEM_BINARIES["ss"], "-n"])

    assert ok and stdout == "ok"
    argv, kwargs = calls[0]
    assert os.path.isabs(argv[0])
    assert kwargs["shell"] is False
    assert kwargs["env"]["PATH"] == "/usr/bin:/bin"
    assert "KAIA_CAPABILITY_TOKEN_SECRET" not in kwargs["env"]
    assert "must-not-reach-children" not in kwargs["env"].values()


def test_cgroup_wrapper_uses_absolute_systemd_run(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda path: path == SYSTEM_BINARIES["systemd-run"])
    monkeypatch.setattr("os.access", lambda path, mode: path == SYSTEM_BINARIES["systemd-run"])
    monkeypatch.setattr("os.geteuid", lambda: 0)

    ok, argv, error = HostExecutor._build_cgroup_wrapper()

    assert ok, error
    assert argv[0] == SYSTEM_BINARIES["systemd-run"]
    assert os.path.isabs(argv[0])
