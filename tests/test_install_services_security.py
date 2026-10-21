from pathlib import Path


def test_installer_removes_excluded_secrets_before_chmod():
    script = Path("scripts/install_services.sh").read_text()
    remove_env = script.index('rm -f "$DEPLOY_DIR/.env"')
    remove_git = script.index('rm -rf "$DEPLOY_DIR/.git" "$DEPLOY_DIR/.venv"')
    normalize_permissions = script.index('find "$DEPLOY_DIR" -type f -exec chmod 644 {} +')

    assert remove_env < normalize_permissions
    assert remove_git < normalize_permissions
