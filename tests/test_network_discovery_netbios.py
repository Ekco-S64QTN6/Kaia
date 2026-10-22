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


def _nbns_packet(name: str, *, response: bool = True, malformed: bool = False) -> bytes:
    # NBNS header: transaction ID, flags, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT.
    flags = 0x8500 if response else 0x0100
    header = struct.pack("!HHHHHH", 0x1234, flags, 1, 1 if response else 0, 0, 0)
    label = _encode_netbios_label(name)
    if malformed:
        label = b"!" + label[1:]
    # QNAME followed by QTYPE=NB (0x0020) and QCLASS=IN (0x0001).
    return header + b"\x20" + label + b"\x00\x20\x00\x01"


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
