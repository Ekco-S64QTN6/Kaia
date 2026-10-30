"""Packet-level tests for passive NetBIOS Name Service discovery."""

import os
import socket
import struct

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

from security.network_discovery import PassiveDiscoveryEngine


def _encode_netbios_label(name: str, suffix: int = 0x20) -> bytes:
    raw = name.upper().encode("ascii").ljust(15, b" ")[:15] + bytes([suffix])
    encoded = bytearray()
    for byte in raw:
        encoded.extend((0x41 + (byte >> 4), 0x41 + (byte & 0x0F)))
    return bytes(encoded)


def _dns_encoded_name(name: str, *, malformed: bool = False, scope: bytes = b"") -> bytes:
    label = _encode_netbios_label(name)
    if malformed:
        label = b"!" + label[1:]
    return b"\x20" + label + scope + b"\x00"


def _nb_rr(name: str, *, malformed: bool = False, scope: bytes = b"") -> bytes:
    encoded_name = _dns_encoded_name(name, malformed=malformed, scope=scope)
    rdata = b"\x00\x00" + socket.inet_aton("192.0.2.44")
    return (
        encoded_name
        + struct.pack("!HHIH", 0x0020, 0x0001, 0, len(rdata))
        + rdata
    )


def _nbns_packet(name: str, *, response: bool = True, malformed: bool = False) -> bytes:
    # Responses use a resource record (QDCOUNT=0, ANCOUNT=1), not a QNAME.
    flags = 0x8500 if response else 0x0100
    if response:
        header = struct.pack("!HHHHHH", 0x1234, flags, 0, 1, 0, 0)
        return header + _nb_rr(name, malformed=malformed)

    header = struct.pack("!HHHHHH", 0x1234, flags, 1, 0, 0, 0)
    question = _dns_encoded_name(name, malformed=malformed) + struct.pack("!HH", 0x0020, 0x0001)
    return header + question


def _nbstat_rdata(entries: list[tuple[str, int, int]]) -> bytes:
    # RFC 1002: NUM_NAMES, 18-byte NODE_NAME entries, then 46-byte statistics.
    body = bytearray([len(entries)])
    for name, suffix, flags in entries:
        raw_name = name.upper().encode("ascii").ljust(15, b" ")[:15] + bytes([suffix])
        body.extend(raw_name)
        body.extend(struct.pack("!H", flags))
    body.extend(bytes(46))
    return bytes(body)


def _nbstat_rr(rdata: bytes, *, owner: bytes | None = None) -> bytes:
    if owner is None:
        owner = _dns_encoded_name("*")
    return (
        owner
        + struct.pack("!HHIH", 0x0021, 0x0001, 0, len(rdata))
        + rdata
    )


def _nbstat_response(
    rdata: bytes,
    *,
    compressed_owner: bool = False,
) -> bytes:
    flags = 0x8500
    if compressed_owner:
        question_name = _dns_encoded_name("NODESTATUS")
        header = struct.pack("!HHHHHH", 0x2345, flags, 1, 1, 0, 0)
        question = question_name + struct.pack("!HH", 0x0021, 0x0001)
        owner = b"\xc0\x0c"
    else:
        header = struct.pack("!HHHHHH", 0x2345, flags, 0, 1, 0, 0)
        question = b""
        owner = _dns_encoded_name("*")
    rr = (
        owner
        + struct.pack("!HHIH", 0x0021, 0x0001, 0, len(rdata))
        + rdata
    )
    return header + question + rr


def _ipv4_udp_frame(payload: bytes, *, source_port: int, dest_port: int,
                    source_ip: str = "192.0.2.44",
                    dest_ip: str = "192.0.2.255") -> bytes:
    udp_len = 8 + len(payload)
    ip_len = 20 + udp_len
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, ip_len, 0x1234, 0, 64, 17, 0,
        socket.inet_aton(source_ip), socket.inet_aton(dest_ip),
    )
    udp_header = struct.pack("!HHHH", source_port, dest_port, udp_len, 0)
    return ip_header + udp_header + payload


def _capture_assets(monkeypatch):
    engine = PassiveDiscoveryEngine()
    captured = []
    monkeypatch.setattr(
        engine, "_log_asset",
        lambda ip, mac, name, vector: captured.append((ip, mac, name, vector)),
    )
    return engine, captured


def test_netbios_response_from_source_port_137_is_discovered(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    packet = _ipv4_udp_frame(
        _nbns_packet("DESKTOP-01"),
        source_port=137,
        dest_port=53000,
        source_ip="192.0.2.44",
        dest_ip="192.0.2.8",
    )

    engine._parse_ipv4(packet, "02:00:00:00:00:44")

    assert captured == [("192.0.2.44", "02:00:00:00:00:44", "DESKTOP-01", "NetBIOS")]


def test_netbios_query_does_not_mislabel_querier_with_requested_name(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    packet = _ipv4_udp_frame(
        _nbns_packet("FILESERVER", response=False),
        source_port=53000,
        dest_port=137,
        source_ip="192.0.2.20",
    )

    engine._parse_ipv4(packet, "02:00:00:00:00:20")

    assert captured == []


def test_netbios_qname_without_response_flag_is_ignored(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    packet = _ipv4_udp_frame(
        _nbns_packet("WORKSTATION", response=False),
        source_port=137,
        dest_port=53000,
    )

    engine._parse_ipv4(packet, "02:00:00:00:00:44")

    assert captured == []


def test_malformed_netbios_first_level_label_is_ignored(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    packet = _ipv4_udp_frame(
        _nbns_packet("DESKTOP-01", malformed=True),
        source_port=137,
        dest_port=53000,
    )

    engine._parse_ipv4(packet, "02:00:00:00:00:44")

    assert captured == []


def test_netbios_decoder_requires_exact_valid_encoded_length():
    engine = PassiveDiscoveryEngine()

    assert engine._decode_netbios_name(b"A" * 31) == ""
    assert engine._decode_netbios_name(b"!" * 32) == ""
    assert engine._decode_netbios_name(_encode_netbios_label("DESKTOP-01")) == "DESKTOP-01"


def test_ipv4_parser_rejects_non_ipv4_and_inconsistent_lengths(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    valid = _ipv4_udp_frame(
        _nbns_packet("DESKTOP-01"), source_port=137, dest_port=53000
    )

    engine._parse_ipv4(bytes([0x65]) + valid[1:], "02:00:00:00:00:44")
    engine._parse_ipv4(valid[:10], "02:00:00:00:00:44")
    # Contradict the IPv4 total-length field: claims a datagram longer than the frame.
    invalid_length = bytearray(valid)
    invalid_length[2:4] = struct.pack("!H", len(valid) + 100)
    engine._parse_ipv4(bytes(invalid_length), "02:00:00:00:00:44")
    # UDP length must fit inside the enclosing IPv4 payload.
    invalid_udp_length = bytearray(valid)
    invalid_udp_length[24:26] = struct.pack("!H", 0xFFFF)
    engine._parse_ipv4(bytes(invalid_udp_length), "02:00:00:00:00:44")

    assert captured == []


def test_netbios_answer_supports_trailing_scope_labels(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    scope = b"\x05WORKG\x03ORG"
    payload = (
        struct.pack("!HHHHHH", 0x1234, 0x8500, 0, 1, 0, 0)
        + _nb_rr("SCOPEHOST", scope=scope)
    )
    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == [
        ("192.0.2.44", "02:00:00:00:00:44", "SCOPEHOST", "NetBIOS")
    ]


def test_nbstat_response_extracts_active_unique_workstation_name(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    rdata = _nbstat_rdata([
        ("GROUPNAME", 0x00, 0x0021),  # group flag: never a host label
        ("OLDSERVER", 0x20, 0x0000),  # inactive
        ("SERVER01", 0x00, 0x0060),   # permanent, active, unique workstation
    ])
    payload = _nbstat_response(rdata)

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )

    assert captured == [
        ("192.0.2.44", "02:00:00:00:00:44", "SERVER01", "NetBIOS")
    ]


def test_nbstat_response_supports_compressed_owner_name(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    rdata = _nbstat_rdata([("COMPRESSED", 0x00, 0x0060)])
    payload = _nbstat_response(rdata, compressed_owner=True)

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )

    assert captured == [
        ("192.0.2.44", "02:00:00:00:00:44", "COMPRESSED", "NetBIOS")
    ]


def test_nbstat_response_ignores_truncated_name_array_or_statistics(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    valid = _nbstat_rdata([("TRUNCATED", 0x00, 0x0060)])
    payload = _nbstat_response(valid[:-1])

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_nbstat_response_rejects_group_only_and_deregistering_names(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    rdata = _nbstat_rdata([
        ("GROUPNAME", 0x00, 0x0021),
        ("DEREGISTER", 0x00, 0x0028),
    ])
    payload = _nbstat_response(rdata)

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_nbns_compression_pointer_loop_is_ignored(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    # The answer owner points to itself at offset 12.
    payload = (
        struct.pack("!HHHHHH", 0x3456, 0x8500, 0, 1, 0, 0)
        + b"\xc0\x0c"
        + struct.pack("!HHIH", 0x0021, 0x0001, 0, 0)
    )

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_nbns_parser_does_not_persist_partial_records_from_truncated_response(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    valid_record = _nb_rr("VALIDFIRST")
    payload = (
        struct.pack("!HHHHHH", 0x4567, 0x8500, 0, 2, 0, 0)
        + valid_record
        + b"\x20\x05SHORT"
    )

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_nb_answer_rejects_malformed_ipv4_address_list(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    owner = _dns_encoded_name("BADADDR")
    rdata = b"\x00\x00\x01\x02"  # 2-byte NB_FLAGS + invalid 2-byte address
    rr = owner + struct.pack("!HHIH", 0x0020, 0x0001, 0, len(rdata)) + rdata
    payload = struct.pack("!HHHHHH", 0x5678, 0x8500, 0, 1, 0, 0) + rr

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_nbns_additional_section_does_not_create_asset_labels(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    # A syntactically valid NB RR in ADDITIONAL is not the answer to this query.
    payload = (
        struct.pack("!HHHHHH", 0x6789, 0x8500, 0, 0, 0, 1)
        + _nb_rr("UNRELATED")
    )

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []


def test_malformed_nbstat_record_prevents_partial_asset_persistence(monkeypatch):
    engine, captured = _capture_assets(monkeypatch)
    valid_nb = _nb_rr("VALIDFIRST")
    malformed_nbstat = _nbstat_rr(_nbstat_rdata([("TRUNCATED", 0x00, 0x0060)])[:-1])
    payload = (
        struct.pack("!HHHHHH", 0x789A, 0x8500, 0, 2, 0, 0)
        + valid_nb
        + malformed_nbstat
    )

    engine._parse_ipv4(
        _ipv4_udp_frame(payload, source_port=137, dest_port=53000),
        "02:00:00:00:00:44",
    )
    assert captured == []
