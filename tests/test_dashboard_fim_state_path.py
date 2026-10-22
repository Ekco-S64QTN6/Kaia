"""Regression checks for the dashboard's canonical FIM state database path."""

from pathlib import Path

import config


def test_dashboard_uses_configured_fim_database_for_all_queries():
    source = Path("kaia_dashboard.py").read_text()

    assert source.count("fim_db_path = config.FIM_DB_PATH") == 2
    assert '"/var/lib/secdaemon/fim_audit.db"' not in source
    assert "FROM fim_events ORDER BY rowid DESC LIMIT 5" in source
    assert "FROM fim_events ORDER BY rowid DESC LIMIT 10" in source


def test_fim_database_path_tracks_the_configured_state_directory(tmp_path, monkeypatch):
    # The configured path is the contract the service writes to; it must not be
    # shadowed by a dashboard-only constant.
    monkeypatch.setattr(config, "FIM_DB_PATH", str(tmp_path / "fim_audit.db"))
    source = Path("kaia_dashboard.py").read_text()

    assert "fim_db_path = config.FIM_DB_PATH" in source
