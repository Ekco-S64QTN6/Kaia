from pathlib import Path


def test_installer_removes_excluded_secrets_before_chmod():
    script = Path("scripts/install_services.sh").read_text()
    remove_env = script.index('rm -f "$DEPLOY_DIR/.env"')
    remove_git = script.index('rm -rf "$DEPLOY_DIR/.git" "$DEPLOY_DIR/.venv"')
    normalize_permissions = script.index('-type f -exec chmod 644 {} +')

    assert remove_env < normalize_permissions
    assert remove_git < normalize_permissions


def test_installer_preflights_secret_and_stops_gate_before_sync():
    script = Path("scripts/install_services.sh").read_text()
    secret_check = script.index("secret_line=")
    stop_gate = script.index("systemctl stop kaia-policy-gate.service")
    sync = script.index("rsync -a --delete")

    assert secret_check < stop_gate < sync
    assert "systemctl enable --now kaia-policy-gate.service" not in script
    assert "systemctl enable kaia-policy-gate.service" in script
    assert 'sudo ${DEPLOY_DIR}/scripts/kaia-baseline.sh' in script
    assert "sudo systemctl start kaia-policy-gate.service" in script
