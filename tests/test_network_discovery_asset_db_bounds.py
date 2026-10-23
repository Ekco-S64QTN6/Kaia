"""Regression tests for persistent passive-discovery asset retention bounds."""

import os
import sqlite3

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test-discovery-secret")

import config
from security.network_discovery import (
    ASSET_DB_MAX_ENTRIES,
    PassiveDiscoveryEngine,
)


def _create_assets_table(conn):
    conn.execute(
        """
        CREATE TABLE assets (
            ip TEXT,
            mac TEXT,
            hostname TEXT,
            vendor TEXT,
            detection_vector TEXT,
            first_seen TEXT,
            last_seen TEXT,
            PRIMARY KEY (ip, mac)
        )
        """
    )


def test_startup_prune_removes_oldest_assets_from_oversized_database(tmp_path):
    engine = PassiveDiscoveryEngine()
    engine.asset_db_max_entries = 10
    conn = sqlite3.connect(tmp_path / "assets.db")
    try:
        _create_assets_table(conn)
        conn.executemany(
            """
            INSERT INTO assets
                (ip, mac, hostname, vendor, detection_vector, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"192.0.2.{i + 1}",
                    f"02:00:00:00:00:{i:02x}",
                    f"host-{i}",
                    "test",
                    "ARP",
                    f"2026-10-22T00:{i // 60:02d}:{i % 60:02d}Z",
                    f"2026-10-22T00:{i // 60:02d}:{i % 60:02d}Z",
                )
                for i in range(25)
            ],
        )
        conn.commit()

        engine._prune_asset_database(conn)
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 10
        oldest = conn.execute("SELECT hostname FROM assets ORDER BY last_seen ASC LIMIT 1").fetchone()[0]
        newest = conn.execute("SELECT hostname FROM assets ORDER BY last_seen DESC LIMIT 1").fetchone()[0]
        assert oldest == "host-15"
        assert newest == "host-24"
    finally:
        conn.close()


def test_spoofed_asset_stream_triggers_periodic_database_pruning(tmp_path, monkeypatch):
    db_path = tmp_path / "assets.db"
    monkeypatch.setattr(config, "ASSETS_DB_PATH", str(db_path))
    engine = PassiveDiscoveryEngine()
    engine.asset_db_max_entries = 8
    engine.asset_db_prune_interval = 3
    engine._queue_enrichment = lambda ip, mac: True

    conn = sqlite3.connect(db_path)
    try:
        _create_assets_table(conn)
        conn.execute("CREATE INDEX idx_assets_last_seen ON assets(last_seen)")
        conn.commit()
    finally:
        conn.close()

    for i in range(30):
        ip = f"192.168.{i // 250}.{i % 250 + 1}"
        mac = f"02:00:00:00:00:{i:02x}"
        engine._log_asset(ip, mac, f"host-{i}", "ARP")

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        assert count <= engine.asset_db_max_entries + engine.asset_db_prune_interval - 1
        assert count == engine.asset_db_max_entries
        assert conn.execute("SELECT COUNT(*) FROM assets WHERE hostname = 'host-29'").fetchone()[0] == 1
