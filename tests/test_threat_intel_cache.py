"""Tests for thread-safe InternetDB rate limiting and negative caching."""

import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from datetime import datetime, UTC
from types import SimpleNamespace
import threading

import pytest

from security import threat_intel


def _redirect_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(threat_intel, "THREAT_INTEL_DIR", str(tmp_path))
    monkeypatch.setattr(threat_intel, "REPUTATION_DB_PATH", str(tmp_path / "reputation.db"))
    monkeypatch.setattr(threat_intel, "INTERNETDB_PATH", str(tmp_path / "internetdb" / "internetdb.db"))
    monkeypatch.setattr(threat_intel, "CVE_DB_PATH", str(tmp_path / "cvedb" / "cve.db"))
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 0.0)
    monkeypatch.setattr(threat_intel.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(threat_intel.time, "monotonic", lambda: 100.0)
    threat_intel.initialize_intel()


def test_invalid_ip_is_rejected_before_api_request(monkeypatch, tmp_path):
    _redirect_cache(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(threat_intel.requests, "get", lambda *args, **kwargs: calls.append(args) or None)

    result = threat_intel.lookup_internetdb("https://attacker.invalid/path")

    assert result["found"] is False
    assert result["source"] == "invalid"
    assert calls == []


def test_internetdb_404_is_cached_as_a_negative_result(monkeypatch, tmp_path):
    _redirect_cache(monkeypatch, tmp_path)
    calls = []

    def not_found(url, timeout, stream):
        calls.append((url, timeout, stream))
        return SimpleNamespace(status_code=404)

    monkeypatch.setattr(threat_intel.requests, "get", not_found)
    first = threat_intel.lookup_internetdb("8.8.8.8")
    second = threat_intel.lookup_internetdb("8.8.8.8")

    assert first["found"] is False and first["source"] == "api-not-found"
    assert second["found"] is False and second["source"] == "cache"
    assert len(calls) == 1
    assert calls[0][2] is True


def test_malformed_api_payload_is_not_cached_as_authoritative_data(monkeypatch, tmp_path):
    _redirect_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        threat_intel.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            iter_content=lambda chunk_size: iter([b'["not", "an", "object"]']),
            close=lambda: None,
        ),
    )

    result = threat_intel.lookup_internetdb("1.1.1.1")

    assert result["found"] is False
    assert result["source"] == "none"


def test_rate_limit_serializes_real_request_calls_across_threads(monkeypatch):
    """Exercise _fetch_internetdb_api itself, not a copy of its scheduling code."""
    clock_lock = threading.Lock()
    clock = {"now": 100.0}
    request_starts = []
    request_lock = threading.Lock()

    def monotonic():
        with clock_lock:
            return clock["now"]

    def sleep(seconds):
        with clock_lock:
            clock["now"] += seconds

    def fake_get(url, timeout, stream):
        assert stream is True
        with request_lock:
            request_starts.append(monotonic())
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            iter_content=lambda chunk_size: iter([b'{"ports": []}']),
            close=lambda: None,
        )

    monkeypatch.setattr(threat_intel.time, "monotonic", monotonic)
    monkeypatch.setattr(threat_intel.time, "sleep", sleep)
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 99.0)
    monkeypatch.setattr(threat_intel.requests, "get", fake_get)

    threads = [
        threading.Thread(
            target=threat_intel._fetch_internetdb_api,
            args=(ip,),
        )
        for ip in ("8.8.8.8", "1.1.1.1", "9.9.9.9")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(request_starts) == 3
    assert request_starts == sorted(request_starts)
    assert all(
        right - left >= 1.0
        for left, right in zip(request_starts, request_starts[1:])
    )


@pytest.mark.parametrize("ip", [
    "10.0.0.7",
    "192.168.1.10",
    "127.0.0.1",
    "169.254.1.5",
    "192.0.2.25",
    "224.0.0.1",
    "::1",
    "fc00::1",
])
def test_non_global_ip_never_egresses_to_internetdb(monkeypatch, tmp_path, ip):
    _redirect_cache(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        threat_intel.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = threat_intel.lookup_internetdb(ip)

    assert result["found"] is False
    assert result["source"] == "non-global"
    assert result["ip"] == ip
    assert calls == []


def test_direct_internetdb_fetch_rejects_non_global_ip_before_rate_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        threat_intel.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 0.0)

    assert threat_intel._fetch_internetdb_api("192.168.0.4") == {}
    assert calls == []
    assert threat_intel._last_api_call_time == 0.0


def test_internetdb_response_size_is_bounded_before_json_parsing(monkeypatch):
    requested_chunks = []
    closed = []
    chunks = [
        b'{"ports": []}',
        b" " * (threat_intel.INTERNETDB_MAX_RESPONSE_BYTES - len(b'{"ports": []}')),
        b"x",
    ]

    def iter_content(chunk_size):
        requested_chunks.append(chunk_size)
        return iter(chunks)

    response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        iter_content=iter_content,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(threat_intel.requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 0.0)

    result = threat_intel._fetch_internetdb_api("8.8.4.4")

    assert result == {}
    assert requested_chunks == [threat_intel.INTERNETDB_RESPONSE_CHUNK_BYTES]
    assert closed == [True]


def test_internetdb_response_uses_streaming_and_closes_after_valid_json(monkeypatch):
    calls = []
    closed = []
    payload = b'{"ports": [80, 443], "hostnames": ["example.test"]}'

    def fake_get(url, timeout, stream):
        calls.append({"url": url, "timeout": timeout, "stream": stream})
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            iter_content=lambda chunk_size: iter([payload[:7], payload[7:]]),
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(threat_intel.requests, "get", fake_get)
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 0.0)

    result = threat_intel._fetch_internetdb_api("8.8.8.8")

    assert result == {"ports": [80, 443], "hostnames": ["example.test"]}
    assert calls == [{
        "url": f"{threat_intel.INTERNETDB_API_URL}/8.8.8.8",
        "timeout": 10,
        "stream": True,
    }]
    assert closed == [True]

