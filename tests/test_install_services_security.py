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


def test_installer_normalizes_both_quote_styles_and_writes_real_newlines():
    script = Path("scripts/install_services.sh").read_text()

    assert 'case "$secret_val" in' in script
    assert 'secret_val="${secret_val:1:${#secret_val}-2}"' in script
    assert 'printf \'%s\\n\' "KAIA_CAPABILITY_TOKEN_SECRET=$secret_val"' in script
    assert 'printf \'%s\\n\' "KAIA_STATE_DIR=$STATE_DIR"' in script



def test_installer_restricts_privileged_destinations_and_checkout_overlap():
    script = Path("scripts/install_services.sh").read_text()

    assert "DEPLOY_DIR must be a dedicated directory below /opt." in script
    assert "STATE_DIR must be a dedicated directory below /var/lib, /srv, or /mnt." in script
    assert "DEPLOY_DIR must not be inside the source checkout." in script
    assert "DEPLOY_DIR must not contain the source checkout." in script
    assert "STATE_DIR must not be inside the source checkout." in script
    assert "STATE_DIR must not contain the source checkout." in script


def test_installer_rejects_symlinked_fim_history_migration():
    script = Path("scripts/install_services.sh").read_text()

    assert '[ -L "$fim_db_destination" ]' in script
    assert '[ -L "$legacy_fim_db" ]' in script
    assert '[ -L "$src" ]' in script
    assert '[ -L "$dst" ]' in script
    assert 'cp -- "$src" "$dst"' in script
    assert 'chown root:kaiacord "$dst"' in script
    assert 'chmod 640 "$dst"' in script
    assert 'cp -- "$legacy_fim_db" "$fim_db_destination"' in script
    assert 'chown root:kaiacord "$fim_db_destination"' in script
