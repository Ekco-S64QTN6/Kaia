import pytest
import json

from scripts.kaia_restore_blocks import collect_blocks


def _record(ip="192.0.2.10", protocol="all", port=None):
    return {
        "result": "approved",
        "request": {
            "action": "block_ip",
            "target_ip": ip,
            "protocol": protocol,
            "port": port,
        },
    }


def test_collect_blocks_skips_valid_json_with_wrong_shape(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps([]) + "\n" +
        json.dumps({"result": "approved", "request": []}) + "\n" +
        json.dumps(_record()) + "\n"
    )

    blocks = collect_blocks(str(ledger))

    assert len(blocks) == 1
    assert blocks[0]["target_ip"] == "192.0.2.10"


def test_collect_blocks_rejects_invalid_protocol_and_port(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record(protocol="icmp")) + "\n" +
        json.dumps(_record(protocol="tcp", port=70000)) + "\n" +
        json.dumps(_record(ip="not-an-ip")) + "\n" +
        json.dumps(_record(ip="198.51.100.5", protocol="tcp", port=443)) + "\n"
    )

    blocks = collect_blocks(str(ledger))

    assert len(blocks) == 1
    assert blocks[0]["target_ip"] == "198.51.100.5"


def test_collect_blocks_reorders_duplicate_to_newest_position(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record("192.0.2.1")) + "\n" +
        json.dumps(_record("192.0.2.2")) + "\n" +
        json.dumps(_record("192.0.2.1")) + "\n"
    )

    blocks = collect_blocks(str(ledger))

    assert [item["target_ip"] for item in blocks] == ["192.0.2.2", "192.0.2.1"]


def _unblock_record(ip="192.0.2.10", protocol="all", port=None, result="approved"):
    return {
        "result": result,
        "request": {
            "action": "unblock_ip",
            "target_ip": ip,
            "protocol": protocol,
            "port": port,
        },
    }


def test_collect_blocks_successful_unblock_removes_prior_block(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record("192.0.2.1")) + "\n" +
        json.dumps(_record("192.0.2.2")) + "\n" +
        json.dumps(_unblock_record("192.0.2.1")) + "\n"
    )

    assert [item["target_ip"] for item in collect_blocks(str(ledger))] == ["192.0.2.2"]


def test_collect_blocks_failed_unblock_does_not_remove_prior_block(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record("192.0.2.1")) + "\n" +
        json.dumps(_unblock_record("192.0.2.1", result="denied")) + "\n"
    )

    assert [item["target_ip"] for item in collect_blocks(str(ledger))] == ["192.0.2.1"]


def test_collect_blocks_reblock_after_unblock_is_active(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record("192.0.2.1")) + "\n" +
        json.dumps(_unblock_record("192.0.2.1")) + "\n" +
        json.dumps(_record("192.0.2.1")) + "\n"
    )

    assert [item["target_ip"] for item in collect_blocks(str(ledger))] == ["192.0.2.1"]


def test_collect_blocks_canonicalizes_ip_and_keeps_protocol_scopes_separate(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(_record("2001:0db8::1", "tcp", 443)) + "\n" +
        json.dumps(_record("2001:db8::1", "udp", 443)) + "\n" +
        json.dumps(_unblock_record("2001:db8::1", "tcp", 443)) + "\n"
    )

    assert collect_blocks(str(ledger)) == [
        {"target_ip": "2001:db8::1", "protocol": "udp", "port": 443}
    ]


def test_collect_blocks_preserves_legacy_block_rule_semantics(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        # Both of these shapes historically produced an IP-wide rule.
        json.dumps(_record(protocol="all", port=53)) + "\n" +
        json.dumps(_record(protocol="tcp")) + "\n" +
        json.dumps(_record("198.51.100.5", "tcp", 443)) + "\n"
    )

    assert collect_blocks(str(ledger)) == [
        {"target_ip": "192.0.2.10", "protocol": "all", "port": None},
        {"target_ip": "198.51.100.5", "protocol": "tcp", "port": 443}
    ]


def test_collect_blocks_bounds_active_state_and_preserves_newest_order(tmp_path):
    ledger = tmp_path / "ledger.json"
    lines = [
        _record("192.0.2.1"),
        _record("192.0.2.2"),
        _record("192.0.2.3"),
        _record("192.0.2.2"),
    ]
    ledger.write_text(chr(10).join(json.dumps(item) for item in lines) + chr(10))

    assert [
        item["target_ip"] for item in collect_blocks(str(ledger), max_blocks=2)
    ] == ["192.0.2.3", "192.0.2.2"]


def test_collect_blocks_skips_oversized_record_and_keeps_reading(tmp_path):
    from scripts.kaia_restore_blocks import MAX_LEDGER_LINE_BYTES

    ledger = tmp_path / "ledger.json"
    valid = (json.dumps(_record("198.51.100.8")) + chr(10)).encode()
    ledger.write_bytes(b"{" + b"x" * (MAX_LEDGER_LINE_BYTES + 100) + bytes([10]) + valid)

    assert [
        item["target_ip"] for item in collect_blocks(str(ledger))
    ] == ["198.51.100.8"]


def test_max_blocks_configuration_is_bounded_and_invalid_values_fall_back(caplog):
    from scripts.kaia_restore_blocks import (
        DEFAULT_MAX_BLOCKS, HARD_MAX_BLOCKS, _parse_max_blocks,
    )

    assert _parse_max_blocks("23") == 23
    assert _parse_max_blocks("not-a-number") == DEFAULT_MAX_BLOCKS
    assert _parse_max_blocks("0") == DEFAULT_MAX_BLOCKS
    assert _parse_max_blocks(str(HARD_MAX_BLOCKS + 1)) == DEFAULT_MAX_BLOCKS
    assert "Invalid KAIA_MAX_RESTORED_BLOCKS" in caplog.text


def test_collect_blocks_refuses_symlink_ledger(tmp_path):
    target = tmp_path / "real-ledger.json"
    target.write_text(json.dumps(_record()) + chr(10))
    link = tmp_path / "ledger.json"
    link.symlink_to(target)

    with pytest.raises(OSError):
        collect_blocks(str(link))


def test_collect_blocks_logs_when_active_rule_limit_drops_oldest(caplog, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        chr(10).join(
            json.dumps(_record(f"192.0.2.{i}")) for i in range(1, 5)
        ) + chr(10)
    )

    blocks = collect_blocks(str(ledger), max_blocks=2)

    assert [item["target_ip"] for item in blocks] == ["192.0.2.3", "192.0.2.4"]
    assert "older active rules were omitted" in caplog.text


def test_main_returns_failure_when_ledger_cannot_be_safely_read(monkeypatch, caplog):
    import sys
    import types
    import scripts.kaia_restore_blocks as restore

    fake_config = types.ModuleType("config")
    fake_config.AUDIT_LOG_PATH = "/var/lib/kaia/symlinked-ledger.json"
    fake_executor_module = types.ModuleType("security.host_executor")
    fake_executor_module.HostExecutor = object
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "security.host_executor", fake_executor_module)

    def fail_read(_path):
        raise OSError("refusing symlink ledger")

    monkeypatch.setattr(restore, "collect_blocks", fail_read)

    assert restore.main() == 1
    assert "Could not safely reconstruct firewall state" in caplog.text


def test_main_returns_failure_when_any_nft_restore_fails(monkeypatch, caplog):
    import sys
    import types
    import scripts.kaia_restore_blocks as restore

    fake_config = types.ModuleType("config")
    fake_config.AUDIT_LOG_PATH = "/var/lib/kaia/audit_ledger.json"

    class FakeHostExecutor:
        @staticmethod
        def execute_mitigation(ip, protocol, port):
            return False, "", "mock nft failure"

    fake_executor_module = types.ModuleType("security.host_executor")
    fake_executor_module.HostExecutor = FakeHostExecutor
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "security.host_executor", fake_executor_module)
    monkeypatch.setattr(
        restore, "collect_blocks",
        lambda _path: [{"target_ip": "192.0.2.10", "protocol": "all", "port": None}],
    )

    assert restore.main() == 1
    assert "mock nft failure" in caplog.text
