"""Regression tests for the privileged execution boundary."""

import os
import sys

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
    monkeypatch.setattr("security.host_executor.subprocess.Popen", lambda *a, **k: pytest.fail("must not launch"))
    ok, _, error = HostExecutor._run_cmd(["systemctl", "restart", "kaia-policy-gate"])
    assert not ok
    assert "absolute executable path" in error


def test_run_cmd_does_not_inherit_capability_signing_secret(monkeypatch):
    monkeypatch.setenv("KAIA_CAPABILITY_TOKEN_SECRET", "must-not-reach-children")
    calls = []
    real_popen = __import__("subprocess").Popen

    def inspect_popen(argv, **kwargs):
        calls.append((argv, kwargs.copy()))
        return real_popen(argv, **kwargs)

    monkeypatch.setattr("security.host_executor.subprocess.Popen", inspect_popen)
    ok, stdout, stderr = HostExecutor._run_cmd([
        sys.executable,
        "-c",
        "import os; print('leaked' if 'KAIA_CAPABILITY_TOKEN_SECRET' in os.environ else 'ok')",
    ])

    assert ok and stdout == "ok" and stderr == ""
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
