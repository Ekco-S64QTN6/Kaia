"""Regression tests for bounded network-discovery resource use."""

import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

from security.network_discovery import PassiveDiscoveryEngine
from queue import Queue


def test_seen_cache_has_a_hard_capacity_and_keeps_recent_deduplication():
    engine = PassiveDiscoveryEngine()
    engine.seen_cache_max_entries = 3

    assert engine._is_recent_duplicate(("aa", "192.0.2.1"), 1000.0) is False
    assert engine._is_recent_duplicate(("aa", "192.0.2.1"), 1001.0) is True

    for number in range(10):
        engine._is_recent_duplicate(("mac", f"192.0.2.{number + 10}"), 1100.0 + number)

    assert len(engine.seen_cache) <= 3
    assert ("aa", "192.0.2.1") not in engine.seen_cache


def test_enrichment_queue_is_bounded_and_nonblocking():
    engine = PassiveDiscoveryEngine()
    engine.enrichment_queue = Queue(maxsize=2)

    assert engine._queue_enrichment("192.0.2.1", "aa:bb:cc:dd:ee:01")
    assert engine._queue_enrichment("192.0.2.2", "aa:bb:cc:dd:ee:02")
    assert not engine._queue_enrichment("192.0.2.3", "aa:bb:cc:dd:ee:03")
    assert engine.enrichment_queue.qsize() == 2
    assert engine.enrichment_dropped == 1


def test_queue_saturation_does_not_spawn_per_asset_threads(monkeypatch):
    engine = PassiveDiscoveryEngine()
    engine.enrichment_queue = Queue(maxsize=1)
    engine._queue_enrichment("192.0.2.1", "aa:bb:cc:dd:ee:01")

    def forbidden_thread(*args, **kwargs):
        raise AssertionError("packet discovery must not spawn per-asset threads")

    monkeypatch.setattr("security.network_discovery.threading.Thread", forbidden_thread)
    assert not engine._queue_enrichment("192.0.2.2", "aa:bb:cc:dd:ee:02")
