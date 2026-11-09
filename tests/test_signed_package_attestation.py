"""Privileged files are compared with the signed package archives, not only pacman's records.

The local database is unsigned: root can rewrite an mtree with the file. The
signed archive in pacman's cache is the stronger reference. gpgv and bsdtar are
injected here; the host run used the real tools (149 files, 43 packages, clean).
"""

import hashlib
import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import package_attestation as pa
from tests.test_package_attestation import _install

ROOT = Path(__file__).resolve().parent.parent
SUDO = b"\x7fELF sudo"


@pytest.fixture
def tree(tmp_path):
    _install(tmp_path, "sudo-1.9-1", {"/usr/bin/sudo": (SUDO, 0o4755)})
    (tmp_path / "var/lib/pacman/local/sudo-1.9-1/desc").write_text(
        "%NAME%\nsudo\n\n%VERSION%\n1.9-1\n\n%ARCH%\nx86_64\n")
    cache = tmp_path / "var/cache/pacman/pkg"
    cache.mkdir(parents=True)
    (cache / "sudo-1.9-1-x86_64.pkg.tar.zst").write_bytes(b"archive")
    (cache / "sudo-1.9-1-x86_64.pkg.tar.zst.sig").write_bytes(b"sig")
    return tmp_path


def _members(content):
    return lambda pkg, member: hashlib.sha256(content[member]).hexdigest() if member in content else None


def test_matching_files_are_clean(tree):
    r = pa.signed_attest(str(tree), verify_sig=lambda p, s: True, member_sha256=_members({"usr/bin/sudo": SUDO}))
    assert r["findings"] == [] and r["checked"] == 1 and r["packages"] == 1


def test_disk_and_record_both_rewritten_is_still_caught(tree):
    """The attack the local database cannot see: file and mtree changed together."""
    evil = b"\x7fELF evil"
    _install(tree, "sudo-1.9-1", {"/usr/bin/sudo": (evil, 0o4755)})   # rewrites file and mtree
    assert pa.attest(fs_root=str(tree))["findings"] == []              # local check is fooled
    r = pa.signed_attest(str(tree), verify_sig=lambda p, s: True, member_sha256=_members({"usr/bin/sudo": SUDO}))
    kinds = {f["kind"] for f in r["findings"]}
    assert kinds == {"differs_from_signed_package", "record_differs_from_signed_package"}


def test_bad_signature_is_reported_and_its_contents_not_trusted(tree):
    r = pa.signed_attest(str(tree), verify_sig=lambda p, s: False,
                         member_sha256=lambda *a: pytest.fail("used an unverified archive"))
    assert [f["kind"] for f in r["findings"]] == ["signature_invalid"]


def test_uncached_package_is_listed_not_failed(tree):
    for f in (tree / "var/cache/pacman/pkg").iterdir():
        f.unlink()
    r = pa.signed_attest(str(tree), verify_sig=lambda p, s: True, member_sha256=_members({}))
    assert r["findings"] == [] and r["no_signed_package"] == ["sudo-1.9-1"]


def test_symlinked_archive_is_not_used(tree):
    cache = tree / "var/cache/pacman/pkg"
    real = tree / "elsewhere.pkg"
    real.write_bytes(b"x")
    (cache / "sudo-1.9-1-x86_64.pkg.tar.zst").unlink()
    (cache / "sudo-1.9-1-x86_64.pkg.tar.zst").symlink_to(real)
    assert pa.cached_package("sudo", "1.9-1", "x86_64", str(cache)) is None


def test_tools_are_called_by_absolute_path_with_a_clean_environment():
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return 0, b"data", b""

    assert pa.gpgv_verify("/c/p.pkg", "/c/p.pkg.sig", run=run) is True
    assert pa.archive_member_sha256("/c/p.pkg", "usr/bin/sudo", run=run) == hashlib.sha256(b"data").hexdigest()
    assert calls[0][:3] == ["/usr/bin/gpgv", "--keyring", "/etc/pacman.d/gnupg/pubring.gpg"]
    assert calls[1] == ["/usr/bin/bsdtar", "-xOf", "/c/p.pkg", "usr/bin/sudo"]


def test_unit_is_unprivileged_offline_and_installed():
    unit = (ROOT / "scripts/kaia-package-verify.service").read_text()
    for line in ("DynamicUser=yes", "CapabilityBoundingSet=\n", "PrivateNetwork=true", "kaia_package_attest.py --signed"):
        assert line in unit
    installer = (ROOT / "scripts/install_services.sh").read_text()
    assert "systemctl enable --now kaia-package-verify.timer" in installer
    assert '"kaia-package-verify.service"' in (ROOT / "scripts/kaia_health_check.py").read_text()
