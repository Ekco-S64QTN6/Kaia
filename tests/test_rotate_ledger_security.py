"""Regression guards for the privileged audit-ledger rotation script."""

from pathlib import Path


def test_rotate_ledger_uses_trusted_state_config_not_environment_override():
    script = Path("scripts/kaia-rotate-ledger.sh").read_text(encoding="utf-8")

    assert 'STATE_CONFIG="/etc/kaia/state.env"' in script
    assert 'STATE_DIR="/var/lib/kaia"' in script
    assert 'STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"' not in script
    assert 'state_config_real="$(realpath -e -- "$STATE_CONFIG")"' in script
    assert "state_config_owner=" in script
    assert "state_config_mode=" in script


def test_rotate_ledger_rejects_symlinks_and_hardlinks_before_mutating():
    script = Path("scripts/kaia-rotate-ledger.sh").read_text(encoding="utf-8")

    symlink_check = script.index('if [ -L "$LEDGER" ]; then')
    hardlink_check = script.index('ledger_links="$(stat -c \'%h\' -- "$LEDGER")"')
    archive_move = script.index('mv -- "$LEDGER" "$archive"')
    archive_chmod = script.index('chmod 600 -- "$archive"')

    assert "set -euo pipefail" in script
    assert symlink_check < hardlink_check < archive_move < archive_chmod
    assert "single-link regular file" in script
    assert "mapfile -t archives < <(" not in script
    assert "mapfile -t old < <(" not in script


def test_rotate_ledger_validates_retention_and_prunes_only_regular_archives():
    script = Path("scripts/kaia-rotate-ledger.sh").read_text(encoding="utf-8")

    assert "KEEP_COUNT < 1 || KEEP_COUNT > 100" in script
    assert "find \"$STATE_DIR\" -maxdepth 1 -type f -name 'audit_ledger.json.*'" in script
    assert 'if ! find "$STATE_DIR"' in script
