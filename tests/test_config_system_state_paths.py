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
            "'state': str(config.STATE_DIR), 'storage': str(config.STORAGE_DIR), "
            "'security': str(config.SECURITY_STORAGE_DIR), "
            "'baseline': str(config.TRUSTED_BASELINE_PATH), 'fim_db': str(config.FIM_DB_PATH), "
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
    assert Path(data["baseline"]) == state / "baseline.sha256"
    assert Path(data["fim_db"]) == state / "fim_audit.db"
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
    baseline_script = (repo_root / "scripts/kaia-baseline.sh").read_text()
    assert "_read_env_assignment /etc/kaia/secret.env" in baseline_script
    assert 'if [ -z "${KAIA_STATE_DIR:-}" ]; then' in baseline_script
    assert 'STATE_DIR="$KAIA_STATE_DIR"' in baseline_script
    assert 'BASELINE="${STATE_DIR}/baseline.sha256"' in baseline_script
    assert ". \"\${PROJECT_DIR}/.env\"" not in baseline_script


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


def test_installer_validates_paths_before_privileged_copy():
    repo_root = Path(__file__).resolve().parents[1]
    installer = (repo_root / "scripts/install_services.sh").read_text()
    validation = installer.index('DEPLOY_DIR="$(realpath -m -- "$DEPLOY_DIR")"')
    rsync = installer.index("rsync -a --delete")

    assert validation < rsync
    assert 'must not be the filesystem root' in installer
    assert 'must not be inside DEPLOY_DIR' in installer


def test_fim_audit_storage_is_configured_and_legacy_history_is_migrated():
    repo_root = Path(__file__).resolve().parents[1]
    config = (repo_root / "core/config.py").read_text()
    fim = (repo_root / "security/fim_daemon.py").read_text()
    hud = (repo_root / "core/web_bridge.py").read_text()
    installer = (repo_root / "scripts/install_services.sh").read_text()
    unit = (repo_root / "scripts/kaia-policy-gate.service").read_text()

    assert 'FIM_DB_PATH = str(STATE_DIR / "fim_audit.db")' in config
    assert "self.db_path = config.FIM_DB_PATH" in fim
    assert 'fim_db_path = config.FIM_DB_PATH' in hud
    assert 'legacy_fim_db="/var/lib/secdaemon/fim_audit.db"' in installer
    assert 'cp -a "$legacy_fim_db" "$STATE_DIR/fim_audit.db"' in installer
    assert "/var/lib/secdaemon" not in unit
