import pytest

import config
from security.host_executor import HostExecutor


@pytest.mark.parametrize(
    ("target_ip", "protocol", "port", "expected_family"),
    [
        ("192.0.2.10", "all", None, "ip"),
        ("2001:db8::10", "tcp", 443, "ip6"),
    ],
)
def test_mitigation_reuses_existing_table_and_chain(
    target_ip, protocol, port, expected_family, monkeypatch
):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        # Model successful nft -exist calls when the table/chain already exist.
        return True, "", ""

    monkeypatch.setattr(HostExecutor, "_run_cmd", fake_run)
    monkeypatch.setattr(config, "NFT_BLOCK_TABLE", "kaia_block")
    monkeypatch.setattr(config, "NFT_BLOCK_CHAIN", "input")
    monkeypatch.setattr(config, "NFT_BLOCK_PRIORITY", -10)
    monkeypatch.setattr("os.geteuid", lambda: 0)

    ok, _, err = HostExecutor.execute_mitigation(target_ip, protocol, port)

    assert ok is True, err
    assert len(calls) == 3
    assert calls[0][:4] == ["/usr/bin/nft", "-exist", "add", "table"]
    assert calls[1][:4] == ["/usr/bin/nft", "-exist", "add", "chain"]
    assert calls[2][:3] == ["/usr/bin/nft", "add", "rule"]
    assert expected_family in calls[2]


def test_mitigation_rejects_ambiguous_protocol_port_shapes(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", staticmethod(lambda cmd: calls.append(cmd) or (True, "", "")))

    invalid = [
        ("192.0.2.1", "icmp", None),
        ("192.0.2.1", "tcp", None),
        ("192.0.2.1", "udp", None),
        ("192.0.2.1", "all", 53),
        ("192.0.2.1", "tcp", True),
    ]
    for target, protocol, port in invalid:
        ok, _, error = HostExecutor.execute_mitigation(target, protocol, port)
        assert not ok and error
    assert calls == []


def test_mitigation_rejects_invalid_ip_before_nft(monkeypatch):
    calls = []
    monkeypatch.setattr(HostExecutor, "_run_cmd", staticmethod(lambda cmd: calls.append(cmd) or (True, "", "")))

    ok, _, error = HostExecutor.execute_mitigation("203.0.113.1/24", "all")
    assert not ok
    assert "invalid ip" in error.lower()
    assert calls == []
