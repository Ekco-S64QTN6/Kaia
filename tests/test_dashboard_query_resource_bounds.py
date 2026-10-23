"""Tests for bounded dashboard queries over append-only security history."""

import os
import sqlite3

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test-dashboard-query-secret")

from kaia_dashboard import _get_honeypot_trigger_summary


def test_honeypot_summary_aggregates_count_and_fetches_only_latest_match(tmp_path):
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE security_events (
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                actor TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO security_events(timestamp, type, actor) VALUES (?, ?, ?)",
            [
                (f"2026-10-22T00:00:{i % 60:02d}Z", "honeypot_port_trigger", f"Port {i}")
                for i in range(5000)
            ]
            + [
                ("2026-10-22T23:59:59Z", "unrelated_event", "ignore me"),
                ("2026-10-22T23:59:58Z", "honeypot_file_access", "attacker (PID 1234)"),
            ],
        )
        conn.commit()

        total, actor, timestamp = _get_honeypot_trigger_summary(conn)

        assert total == 5001
        assert actor == "attacker (PID 1234)"
        assert timestamp == "2026-10-22T23:59:58Z"
    finally:
        conn.close()


def test_honeypot_summary_handles_empty_history(tmp_path):
    conn = sqlite3.connect(tmp_path / "empty.db")
    try:
        conn.execute(
            "CREATE TABLE security_events(timestamp TEXT, type TEXT, actor TEXT)"
        )
        total, actor, timestamp = _get_honeypot_trigger_summary(conn)
        assert (total, actor, timestamp) == (0, "None", "None")
    finally:
        conn.close()
