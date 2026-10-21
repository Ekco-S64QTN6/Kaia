import os

import pytest

import config
from security.host_executor import HostExecutor


def test_state_write_rejects_sibling_with_common_prefix(tmp_path, monkeypatch):
    workspace = tmp_path / "kaia"
    sibling = tmp_path / "kaia-evil"
    workspace.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    target = sibling / "outside.txt"
    ok, _, err = HostExecutor.execute_state_modification(str(target), "pwned")

    assert ok is False
    assert "Path modification violation" in err
    assert not target.exists()


def test_state_write_rejects_symlink_escape(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "export").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    target = workspace / "export" / "should-not-exist.txt"
    ok, _, err = HostExecutor.execute_state_modification(str(target), "pwned")

    assert ok is False
    assert "Path modification violation" in err
    assert not (outside / "should-not-exist.txt").exists()


def test_state_write_allows_file_inside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    target = workspace / "notes.txt"
    ok, message, err = HostExecutor.execute_state_modification(str(target), "hello")

    assert ok is True
    assert target.read_text() == "hello"
    assert not err
