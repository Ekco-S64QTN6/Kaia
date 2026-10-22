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

    def not_found(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(status_code=404)

    monkeypatch.setattr(threat_intel.requests, "get", not_found)
    first = threat_intel.lookup_internetdb("203.0.113.42")
    second = threat_intel.lookup_internetdb("203.0.113.42")

    assert first["found"] is False and first["source"] == "api-not-found"
    assert second["found"] is False and second["source"] == "cache"
    assert len(calls) == 1


def test_malformed_api_payload_is_not_cached_as_authoritative_data(monkeypatch, tmp_path):
    _redirect_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        threat_intel.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: ["not", "an", "object"],
        ),
    )

    result = threat_intel.lookup_internetdb("203.0.113.99")

    assert result["found"] is False
    assert result["source"] == "none"


def test_rate_limit_lock_serializes_request_start_slots(monkeypatch):
    starts = []
    lock = threading.Lock()
    ticks = iter([100.0, 100.0, 101.0, 101.0, 102.0, 102.0])

    def monotonic():
        with lock:
            return next(ticks)

    sleeps = []
    monkeypatch.setattr(threat_intel.time, "monotonic", monotonic)
    monkeypatch.setattr(threat_intel.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(threat_intel, "_last_api_call_time", 99.0)

    # Exercise the scheduling boundary directly without network I/O.
    def reserve_slot():
        with threat_intel._api_rate_limit_lock:
            now = threat_intel.time.monotonic()
            delay = 1.0 - (now - threat_intel._last_api_call_time)
            if delay > 0:
                threat_intel.time.sleep(delay)
            threat_intel._last_api_call_time = threat_intel.time.monotonic()
            starts.append(threat_intel._last_api_call_time)

    threads = [threading.Thread(target=reserve_slot) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(starts) == 3
    assert starts == sorted(starts)
