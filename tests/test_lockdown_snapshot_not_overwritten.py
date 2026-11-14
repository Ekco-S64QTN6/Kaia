"""A second lockdown must not replace the restore point with the lockdown itself.

On 2026-10-30 the gate locked the host down nine times in a row. Each run
snapshotted the current ruleset, so from the second run on `pre-lockdown.nft`
held only `table inet kaia_lockdown` (256 bytes on the host; the real firewall
was 1109). That file passes `nft -c`, and kaia-unlock.sh would have restored
it: all traffic dropped again, reported as "Firewall restored".
"""
import os
import stat
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKDOWN = (ROOT / "scripts/kaia-lockdown.sh").read_text()
UNLOCK = (ROOT / "scripts/kaia-unlock.sh").read_text()

LOCKDOWN_RULESET = textwrap.dedent("""\
    table inet kaia_lockdown {
    \tchain input {
    \t\ttype filter hook input priority filter; policy drop;
    \t}
    }
""")
HOST_RULESET = textwrap.dedent("""\
    table inet filter {
    \tchain input {
    \t\ttype filter hook input priority filter; policy drop;
    \t\tct state established,related accept
    \t}
    }
""")


def _snapshot_block() -> str:
    """The script's own snapshot section, from the umask to the containment step."""
    start = LOCKDOWN.index("umask 077")
    end = LOCKDOWN.index("# --- Contain")
    return LOCKDOWN[start:end]


def _run_snapshot(tmp_path, already_locked: bool) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    snapshot = state / "pre-lockdown.nft"
    snapshot.write_text(HOST_RULESET)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    current = LOCKDOWN_RULESET if already_locked else HOST_RULESET
    (tmp_path / "current.nft").write_text(current)
    fake = bindir / "nft"
    fake.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        if [ "$1 $2 $3 $4" = "list table inet kaia_lockdown" ]; then
            grep -q kaia_lockdown "{tmp_path}/current.nft"; exit $?
        fi
        if [ "$1 $2" = "list ruleset" ]; then cat "{tmp_path}/current.nft"; exit 0; fi
        exit 1
    """))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    (bindir / "logger").write_text("#!/bin/bash\nexit 0\n")
    (bindir / "logger").chmod(0o755)
    script = (f'set -uo pipefail\nCOLOR_RED=""; COLOR_YELLOW=""; COLOR_RESET=""\n'
              f'STATE_DIR="{state}"\nSNAPSHOT="{snapshot}"\n' + _snapshot_block())
    subprocess.run(["/bin/bash", "-c", script], check=True,
                   env={"PATH": f"{bindir}:/usr/bin:/bin"}, capture_output=True)
    return snapshot


def test_first_lockdown_snapshots_the_host_firewall(tmp_path):
    snapshot = _run_snapshot(tmp_path, already_locked=False)
    assert snapshot.read_text() == HOST_RULESET


def test_repeat_lockdown_keeps_the_existing_snapshot(tmp_path):
    snapshot = _run_snapshot(tmp_path, already_locked=True)
    assert snapshot.read_text() == HOST_RULESET, "the restore point was overwritten by the lockdown ruleset"
    assert not list(snapshot.parent.glob("pre-lockdown-*.nft")), "no history copy of a lockdown ruleset"


def test_unlock_ignores_a_snapshot_that_is_the_lockdown(tmp_path):
    marker = "grep -Eq '^[[:space:]]*table[[:space:]]+inet[[:space:]]+kaia_lockdown([[:space:]]|$)'"
    assert marker in UNLOCK
    # The check sits after validation and before the snapshot is used.
    assert UNLOCK.index("SNAPSHOT_READY=1") < UNLOCK.index(marker) < UNLOCK.index("nft flush ruleset")
    pattern = r"^[[:space:]]*table[[:space:]]+inet[[:space:]]+kaia_lockdown([[:space:]]|$)"
    for text, expected in ((LOCKDOWN_RULESET, 0), (HOST_RULESET, 1)):
        f = tmp_path / "s.nft"
        f.write_text(text)
        assert subprocess.run(["grep", "-Eq", pattern, str(f)]).returncode == expected
