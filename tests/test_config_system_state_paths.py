import json
import os
import subprocess
import sys
from pathlib import Path


def test_system_state_mode_keeps_runtime_data_outside_read_only_checkout(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env["KAIA_STATE_DIR"] = str(state_dir)
    env["KAIA_CAPABILITY_TOKEN_SECRET"] = "test-only-secret"
    env["PYTHONPATH"] = str(repo_root / "core")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config, json; print(json.dumps({"
            "'state': config.STATE_DIR, 'storage': str(config.STORAGE_DIR), "
            "'security': str(config.SECURITY_STORAGE_DIR), "
            "'threat_intel': str(config.THREAT_INTEL_DIR), "
            "'system': config.STATE_DIR_IS_SYSTEM}))",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout.strip().splitlines()[-1])
    state = Path(data["state"])

    assert data["system"] is True
    assert Path(data["storage"]) == state
    assert Path(data["security"]) == state
    assert Path(data["threat_intel"]) == state / "threat_intel"
    assert (state / "threat_intel").is_dir()


def test_installed_units_use_configured_state_directory():
    repo_root = Path(__file__).resolve().parents[1]
    policy_unit = (repo_root / "scripts/kaia-policy-gate.service").read_text()
    restore_unit = (repo_root / "scripts/kaia-restore-blocks.service").read_text()
    lockdown_unit = (repo_root / "scripts/kaia-lockdown.service").read_text()
    installer = (repo_root / "scripts/install_services.sh").read_text()

    assert "Environment=KAIA_STATE_DIR=__STATE_DIR__" in policy_unit
    assert "ReadWritePaths=/run/kaiacord __STATE_DIR__" in policy_unit
    assert "Environment=KAIA_STATE_DIR=__STATE_DIR__" in restore_unit
    assert "ReadOnlyPaths=__STATE_DIR__" in restore_unit
    assert "Environment=KAIA_STATE_DIR=__STATE_DIR__" in lockdown_unit
    assert '"s|__STATE_DIR__|${STATE_DIR}|g"' in installer


def test_installer_persists_state_path_for_manual_unlock():
    repo_root = Path(__file__).resolve().parents[1]
    installer = (repo_root / "scripts/install_services.sh").read_text()
    unlock = (repo_root / "scripts/kaia-unlock.sh").read_text()

    assert 'printf \'KAIA_STATE_DIR=%q\\n\' "$STATE_DIR" > /etc/kaia/state.env' in installer
    assert '. /etc/kaia/state.env' in unlock


def test_config_treats_explicit_state_path_as_system_mode(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env = os.environ.copy()
    env["KAIA_STATE_DIR"] = str(state_dir)
    env["KAIA_CAPABILITY_TOKEN_SECRET"] = "test-only-secret"
    env["PYTHONPATH"] = str(repo_root / "core")
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.STATE_DIR_IS_SYSTEM)"],
        cwd=repo_root, env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_installed_units_use_configured_deployment_directory():
    repo_root = Path(__file__).resolve().parents[1]
    policy_unit = (repo_root / "scripts/kaia-policy-gate.service").read_text()
    restore_unit = (repo_root / "scripts/kaia-restore-blocks.service").read_text()
    lockdown_unit = (repo_root / "scripts/kaia-lockdown.service").read_text()

    assert "WorkingDirectory=__DEPLOY_DIR__" in policy_unit
    assert "ExecStart=/usr/bin/python3 __DEPLOY_DIR__/security/policy_gate.py" in policy_unit
    assert "ExecStart=/usr/bin/env python3 __DEPLOY_DIR__/scripts/kaia_restore_blocks.py" in restore_unit
    assert "Environment=KAIA_PROJECT_DIR=__DEPLOY_DIR__" in lockdown_unit
