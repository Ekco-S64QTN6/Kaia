"""The one-command deploy keeps the required order and never writes as root into user paths."""

from pathlib import Path

SCRIPT = Path("scripts/kaia-deploy-verify.sh").read_text()


def test_install_then_baseline_then_start():
    i = SCRIPT.index("scripts/install_services.sh")
    b = SCRIPT.index("/opt/kaia/scripts/kaia-baseline.sh")
    s = SCRIPT.index("systemctl start kaia-policy-gate.service")
    assert i < b < s


def test_a_failed_baseline_disables_the_gate():
    block = SCRIPT[SCRIPT.index("kaia-baseline.sh"):SCRIPT.index("systemctl start kaia-policy-gate.service")]
    assert "systemctl disable kaia-policy-gate.service" in block and "exit 1" in block


def test_root_never_writes_into_a_home_directory():
    assert "/home/" not in SCRIPT and "~/" not in SCRIPT
    assert 'OUT="/var/log/kaia/' in SCRIPT
    assert 'mktemp -d /var/lib/kaia-privtest.' in SCRIPT and 'rm -rf "$TDIR"' in SCRIPT


def test_refuses_to_run_without_root_first():
    first_cmd = [l for l in SCRIPT.splitlines() if l and not l.startswith("#")][1]
    assert first_cmd.startswith('[ "$EUID" -eq 0 ]')
