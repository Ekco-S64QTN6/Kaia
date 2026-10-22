import pytest

from security.host_executor import HostExecutor


@pytest.mark.parametrize("args", [["del", "default"], ["add", "10.0.0.0/8", "via", "127.0.0.1"], ["flush"]])
def test_ip_route_diagnostics_reject_mutating_subcommands(args, monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", lambda cmd: calls.append(cmd) or (True, "", ""))

    ok, _, err = HostExecutor.execute_diagnostics("ip_route", args)

    assert ok is False
    assert "mutations are forbidden" in err
    assert calls == []


def test_ss_kill_option_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", lambda cmd: calls.append(cmd) or (True, "", ""))

    ok, _, err = HostExecutor.execute_diagnostics("ss", ["-K"])

    assert ok is False
    assert "read-only display flags" in err
    assert calls == []


def test_read_only_diagnostics_options_are_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", lambda cmd: calls.append(cmd) or (True, "ok", ""))

    ok, stdout, _ = HostExecutor.execute_diagnostics("ss", ["-tuln"])
    assert ok is True
    assert stdout == "ok"
    assert calls == [["/usr/bin/ss", "-tuln"]]

    ok, _, _ = HostExecutor.execute_diagnostics("ip_route", ["-4", "show"])
    assert ok is True
    assert calls[-1] == ["/usr/bin/ip", "route", "-4", "show"]


def test_diagnostics_reject_non_string_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", lambda cmd: calls.append(cmd) or (True, "", ""))

    ok, _, err = HostExecutor.execute_diagnostics("ss", ["-n", 123])

    assert ok is False
    assert "list of strings" in err
    assert calls == []


def test_all_diagnostics_launch_absolute_system_binaries(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", lambda cmd: calls.append(cmd) or (True, "", ""))
    monkeypatch.setattr("os.geteuid", lambda: 0)

    assert HostExecutor.execute_diagnostics("ss", ["-n"])[0]
    assert HostExecutor.execute_diagnostics("ip_route", ["show"])[0]
    assert HostExecutor.execute_diagnostics("nft_list", [])[0]
    assert [cmd[0] for cmd in calls] == ["/usr/bin/ss", "/usr/bin/ip", "/usr/bin/nft"]
    assert all(cmd[0].startswith("/") for cmd in calls)
