from pathlib import Path


def test_lockdown_script_validates_and_verifies_atomic_ruleset():
    script = Path("scripts/kaia-lockdown.sh").read_text()
    assert "nft -c -f \"$LOCKDOWN_RULES\"" in script
    assert "nft -f \"$LOCKDOWN_RULES\"" in script
    assert "for chain in input forward output" in script
    assert "Lockdown ruleset verified" in script
