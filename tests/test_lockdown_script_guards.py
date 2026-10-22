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
