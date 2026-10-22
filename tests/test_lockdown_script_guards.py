from pathlib import Path


def test_lockdown_script_validates_and_verifies_atomic_ruleset():
    script = Path("scripts/kaia-lockdown.sh").read_text()
    assert "nft -c -f \"$LOCKDOWN_RULES\"" in script
    assert "nft -f \"$LOCKDOWN_RULES\"" in script
    assert "for chain in input forward output" in script
    assert "Lockdown ruleset verified" in script



def test_lockdown_preserves_operator_state_access_and_private_snapshot_umask():
    script = Path("scripts/kaia-lockdown.sh").read_text()

    assert "umask 077" in script
    assert 'chmod 750 "$STATE_DIR"' in script
    assert 'chown root:kaiacord "$STATE_DIR"' in script
    assert 'chmod 700 "$STATE_DIR"' not in script


def test_unlock_validates_restore_point_and_state_path_before_flushing_firewall():
    script = Path("scripts/kaia-unlock.sh").read_text()

    assert "set -euo pipefail" in script
    assert '. /etc/kaia/state.env' not in script
    assert 'if [ -L "$STATE_CONFIG" ] || [ ! -f "$STATE_CONFIG" ]; then' in script
    assert 'if [ -L "$SNAPSHOT" ]; then' in script
    assert 'snapshot_owner="$(stat -c' in script
    assert 'nft -c -f "$SNAPSHOT"' in script
    assert 'no active UFW manager or validated snapshot is available; firewall unchanged' in script

    preflight = script.index("SNAPSHOT_READY=0")
    snapshot_display = script.index('if [ "$SNAPSHOT_READY" -eq 1 ]; then')
    flush = script.index("nft flush ruleset")
    assert preflight < snapshot_display < flush


def test_unlock_refuses_requested_missing_snapshot_before_firewall_flush():
    script = Path("scripts/kaia-unlock.sh").read_text()

    missing_snapshot_guard = script.index('if [ "$FROM_SNAPSHOT" -eq 1 ] && [ "$SNAPSHOT_READY" -ne 1 ]; then')
    no_restore_guard = script.index('if [ "$UFW_ACTIVE" -ne 1 ] && [ "$SNAPSHOT_READY" -ne 1 ]; then')
    flush = script.index("nft flush ruleset")

    assert missing_snapshot_guard < no_restore_guard < flush
