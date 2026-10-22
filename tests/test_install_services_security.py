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



def test_installer_requires_32_byte_hex_capability_signing_key():
    script = Path("scripts/install_services.sh").read_text()

    format_check = script.index('[[ ! "$secret_val" =~ ^[A-Fa-f0-9]{64}$ ]]')
    stop_gate = script.index("systemctl stop kaia-policy-gate.service")
    sync = script.index("rsync -a --delete")

    assert format_check < stop_gate < sync
    assert "exactly 64 hexadecimal characters" in script
    assert "openssl rand -hex 32" in script



def test_installer_rejects_unexpected_symlinks_before_root_chown():
    script = Path("scripts/install_services.sh").read_text()

    source_check = script.index('validate_deployment_symlinks "$PROJECT_REAL" source')
    stop_gate = script.index("systemctl stop kaia-policy-gate.service")
    rsync = script.index("rsync -a --delete")
    deployed_check = script.index('validate_deployment_symlinks "$DEPLOY_DIR" deployed')
    chown = script.index('find "$DEPLOY_DIR" -path "$DEPLOY_DIR/storage" -prune -o -exec chown')

    assert source_check < stop_gate < rsync < deployed_check < chown
    assert '[[ "$relative" != "static/dashboard" || "$target" != "../core/web/static" ]]' in script
    assert 'realpath -e -- "$link"' in script
    assert "dashboard symlink escapes deployment tree" in script


def test_installer_allows_only_verified_in_tree_dashboard_link():
    script = Path("scripts/install_services.sh").read_text()

    assert '"$relative" != "static/dashboard"' in script
    assert '"$target" != "../core/web/static"' in script
    assert 'find "$tree_root" -path "$tree_root/storage" -prune -o -type l -print0' in script



def test_installer_does_not_precreate_or_chmod_real_world_honeypot_paths():
    script = Path("scripts/install_services.sh").read_text()

    assert 'touch /etc/api_keys.json /var/backups/credentials.txt /root/.ssh/authorized_keys.bak' not in script
    assert 'chmod 600 /etc/api_keys.json /var/backups/credentials.txt /root/.ssh/authorized_keys.bak' not in script
    assert "exclusive, no-follow" in script


def test_installer_rejects_untrusted_destination_path_ancestors_before_changes():
    script = Path("scripts/install_services.sh").read_text()

    validate_function = script.index("validate_root_controlled_path()")
    deploy_check = script.index('validate_root_controlled_path "$DEPLOY_DIR" DEPLOY_DIR')
    state_check = script.index('validate_root_controlled_path "$STATE_DIR" STATE_DIR')
    stop_gate = script.index("systemctl stop kaia-policy-gate.service")
    rsync = script.index("rsync -a --delete")

    assert validate_function < deploy_check < state_check < stop_gate < rsync
    assert "must be root-owned and not group/world-writable" in script
    assert "has a symlinked path component" in script
    assert "path component is not a directory" in script


def test_geoip_updater_uses_private_temporary_paths_and_cleanup():
    script = Path("scripts/update_geoip.sh").read_text()

    assert 'set -euo pipefail' in script
    assert 'umask 077' in script
    assert 'TEMP_ROOT="$(mktemp -d /tmp/kaia-geoip.XXXXXX)"' in script
    assert 'trap cleanup EXIT' in script
    assert 'TEMP_TAR="${TEMP_ROOT}/download.tar.gz"' in script
    assert 'CURL_CONFIG="${TEMP_ROOT}/curl.conf"' in script
    assert 'curl --config "$CURL_CONFIG" --fail' in script
    assert '/tmp/GeoLite2-City.tar.gz' not in script
    assert '/tmp/GeoLite2-City-extracted' not in script


def test_geoip_updater_validates_archive_paths_and_only_copies_one_regular_database():
    script = Path("scripts/update_geoip.sh").read_text()

    assert 'tarfile.open(archive_path, "r:gz")' in script
    assert 'path.is_absolute() or ".." in path.parts' in script
    assert 'member.isdir() or member.isfile()' in script
    assert 'len(candidates) != 1' in script
    assert 'open(output_path, "xb")' in script
    assert 'max_size = 512 * 1024 * 1024' in script

