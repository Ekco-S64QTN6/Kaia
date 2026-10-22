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



def test_state_write_rejects_final_symlink_without_modifying_target(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("original outside content")
    target = workspace / "visible.txt"
    target.symlink_to(outside)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    ok, _, err = HostExecutor.execute_state_modification(str(target), "overwrite attempt")

    assert ok is False
    assert "regular file" in err
    assert target.is_symlink()
    assert outside.read_text() == "original outside content"


def test_state_write_rejects_symlinked_parent_even_when_inside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_dir = workspace / "real"
    real_dir.mkdir()
    alias = workspace / "alias"
    alias.symlink_to(real_dir, target_is_directory=True)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    ok, _, err = HostExecutor.execute_state_modification(str(alias / "note.txt"), "must not write")

    assert ok is False
    assert "Path modification violation" in err
    assert not (real_dir / "note.txt").exists()


def test_state_write_rejects_fifo_target_without_blocking(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "pipe.txt"
    os.mkfifo(target)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    ok, _, err = HostExecutor.execute_state_modification(str(target), "no pipe writes")

    assert ok is False
    assert "regular file" in err
    assert target.is_fifo()


def test_state_write_atomically_replaces_hardlink_without_changing_external_file(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.txt"
    external = tmp_path / "external.txt"
    external.write_text("shared original")
    os.link(external, target)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))

    ok, _, err = HostExecutor.execute_state_modification(str(target), "replacement content")

    assert ok is True, err
    assert target.read_text() == "replacement content"
    assert external.read_text() == "shared original"
    assert target.stat().st_ino != external.stat().st_ino



def test_root_state_write_refuses_untrusted_workspace_ownership(tmp_path, monkeypatch):
    workspace = tmp_path / "user-owned-workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    target = workspace / "notes.txt"
    ok, _, err = HostExecutor.execute_state_modification(str(target), "must not write")

    assert ok is False
    assert "root-owned workspace" in err
    assert not target.exists()


def test_root_state_write_refuses_group_or_world_writable_parent(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = workspace / "public"
    public.mkdir(mode=0o777)
    public.chmod(0o777)
    monkeypatch.setattr(config, "WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    target = public / "notes.txt"
    ok, _, err = HostExecutor.execute_state_modification(str(target), "must not write")

    assert ok is False
    assert "non-root-controlled parent directory" in err
    assert not target.exists()
