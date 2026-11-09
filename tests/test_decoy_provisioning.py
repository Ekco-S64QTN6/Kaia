"""The SSH decoy is provisioned outside home directories by the installer (#102).

Under ProtectHome=tmpfs the gate could never create ~/.ssh/authorized_keys.bak,
and ProtectSystem=strict stops it creating any new file, so the installer
provisions missing decoys. It never overwrites anything.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security.decoys import DECOYS

ROOT = Path(__file__).resolve().parent.parent


def _installer():
    spec = importlib.util.spec_from_file_location("install_decoys", ROOT / "scripts/install_decoys.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tree(tmp_path):
    for p in DECOYS:
        (tmp_path / os.path.dirname(p).lstrip("/")).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_no_decoy_lives_in_a_home_directory():
    assert not any(p.startswith(("/root", "/home")) or "~" in p for p in DECOYS)
    unit = (ROOT / "scripts/kaia-policy-gate.service").read_text()
    assert "/root/.ssh" not in unit and "-/var/backups/authorized_keys.bak" in unit


def test_missing_decoys_are_created_with_exact_contents_private(tmp_path):
    root = _tree(tmp_path)
    results = _installer().provision(root=str(root))
    assert set(results.values()) == {"created"}
    for path, content in DECOYS.items():
        f = root / path.lstrip("/")
        assert f.read_text() == content and (f.stat().st_mode & 0o777) == 0o600


def test_existing_files_and_symlinks_are_never_touched(tmp_path):
    root = _tree(tmp_path)
    real = root / "var/backups/credentials.txt"
    real.write_text("REAL DATA")
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (root / "var/backups/authorized_keys.bak").symlink_to(victim)
    results = _installer().provision(root=str(root))
    assert results["/var/backups/credentials.txt"] == "exists" and real.read_text() == "REAL DATA"
    assert results["/var/backups/authorized_keys.bak"] == "exists" and victim.read_text() == "keep"


def test_missing_parent_is_an_error_not_a_creation(tmp_path):
    results = _installer().provision(root=str(tmp_path))
    assert all(v.startswith("error") for v in results.values())


def test_coordinator_watches_the_shared_decoys_and_installer_runs_before_the_gate():
    from security.honeypot import HoneypotCoordinator
    assert HoneypotCoordinator().decoys == DECOYS
    script = (ROOT / "scripts/install_services.sh").read_text()
    assert script.index("install_decoys.py") < script.index("systemctl enable kaia-policy-gate.service")
