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
