"""Tests for exact, idempotent deletion of Kaia-owned firewall rules."""

import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

from security.host_executor import HostExecutor, SYSTEM_BINARIES


def test_unmitigation_deletes_only_exact_rule_handles(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    rules = {
        "10": "ip saddr 192.0.2.1 drop # handle 10",
        "11": "ip saddr 192.0.2.1 tcp dport 443 drop # handle 11",
        "12": "ip saddr 192.0.2.2 drop # handle 12",
    }
    calls = []

    def fake_run(argv):
        calls.append(argv)
        if argv[1:4] == ["-a", "list", "chain"]:
            return True, "\n".join(rules.values()), ""
        if argv[1:3] == ["delete", "rule"]:
            handle = argv[-1]
            rules.pop(handle, None)
            return True, "", ""
        raise AssertionError(argv)

    monkeypatch.setattr(HostExecutor, "_run_cmd", staticmethod(fake_run))
    ok, message, error = HostExecutor.execute_unmitigation("192.0.2.1", "tcp", 443)

    assert ok, error
    assert "Removed 1" in message
    assert "10" in rules and "12" in rules
    assert "11" not in rules
    delete_calls = [cmd for cmd in calls if cmd[1:3] == ["delete", "rule"]]
    assert delete_calls == [[
        SYSTEM_BINARIES["nft"], "delete", "rule", "inet", "kaia_block", "input", "handle", "11"
    ]]


def test_unmitigation_is_idempotent_when_exact_rule_is_absent(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(
        HostExecutor, "_run_cmd",
        staticmethod(lambda argv: (True, "chain input {\n}", "") if argv[1:4] == ["-a", "list", "chain"] else (False, "", "unexpected")),
    )
    ok, message, error = HostExecutor.execute_unmitigation("192.0.2.1")
    assert ok and "No matching" in message
    assert not error


def test_unmitigation_treats_missing_kaia_table_as_already_unblocked(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr(
        HostExecutor, "_run_cmd",
        staticmethod(lambda argv: (False, "", "Error: No such file or directory")),
    )
    ok, message, error = HostExecutor.execute_unmitigation("192.0.2.1")
    assert ok and "No matching" in message
    assert not error


def test_unmitigation_rejects_invalid_or_ambiguous_rule_scope(monkeypatch):
    monkeypatch.setattr("security.host_executor.os.geteuid", lambda: 0)
    for target, protocol, port in [
        ("not-an-ip", "all", None),
        ("192.0.2.1", "all", 53),
        ("192.0.2.1", "tcp", None),
        ("192.0.2.1", "icmp", None),
    ]:
        ok, _, error = HostExecutor.execute_unmitigation(target, protocol, port)
        assert not ok and error


def test_unmitigation_fails_closed_when_nft_executable_is_missing(monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    calls = []

    def missing_executable(argv):
        calls.append(argv)
        return (
            False, "",
            "Command executable not found: [Errno 2] No such file or directory: '/usr/bin/nft'",
        )

    monkeypatch.setattr(HostExecutor, "_run_cmd", staticmethod(missing_executable))
    ok, message, error = HostExecutor.execute_unmitigation("192.0.2.1")

    assert not ok
    assert "could not run" in error.lower()
    assert "No matching" not in message
    assert calls


def test_unmitigation_rejects_non_string_ip_without_raising():
    ok, _, error = HostExecutor.execute_unmitigation(None)

    assert not ok
    assert "invalid ip" in error.lower()
