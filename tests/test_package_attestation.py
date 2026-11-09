"""Installed-content attestation against pacman's mtree records.

Fixtures use the mtree format pacman writes (copied from the host's sudo package:
/set defaults, ./path entries, octal escapes), not the parser's own constants.
"""

import gzip
import hashlib
import logging
import os
import stat
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health
from security import package_attestation as pa

UID, GID = os.getuid(), os.getgid()


def _mtree(body: str) -> str:
    return (f"#mtree\n/set type=file uid={UID} gid={GID} mode=644\n"
            "./.BUILDINFO time=1781602302.0 size=6249 sha256digest=" + "a" * 64 + "\n" + body)


def _install(root: Path, pkg: str, files: dict, mtree_extra: str = "") -> None:
    """files: absolute path -> (bytes, mode). Writes the files and a matching mtree."""
    lines = []
    for path, (data, mode) in files.items():
        target = root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        os.chmod(target, mode)
        escaped = path[1:].replace(" ", "\\040")
        lines.append(f"./{escaped} time=1781602302.0 mode={mode:o} size={len(data)} "
                     f"sha256digest={hashlib.sha256(data).hexdigest()}")
    db = root / "var/lib/pacman/local" / pkg
    db.mkdir(parents=True, exist_ok=True)
    with gzip.open(db / "mtree", "wt") as f:
        f.write(_mtree("\n".join(lines) + "\n" + mtree_extra))


@pytest.fixture
def tree(tmp_path):
    _install(tmp_path, "sudo-1.9-1", {
        "/usr/bin/sudo": (b"\x7fELF sudo", 0o4755),
        "/usr/bin/visudo": (b"\x7fELF visudo", 0o755),
    })
    _install(tmp_path, "pam-1.7-1", {"/usr/lib/security/pam_unix.so": (b"\x7fELF pam", 0o755)})
    return tmp_path


# --- parsing ---

def test_parses_the_format_pacman_writes():
    text = _mtree(
        "/set mode=755\n"
        "./usr/bin time=1781602302.0 type=dir\n"
        "./usr/bin/sudo time=1781602302.0 mode=4755 size=261304 sha256digest=" + "e" * 64 + "\n"
        "./usr/bin/sudoedit time=1781602302.0 mode=777 type=link link=sudo\n"
        "./usr/bin/visudo time=1781602302.0 size=242320 sha256digest=" + "b" * 64 + "\n"
        "./usr/share/doc/a\\040b time=1 size=1 sha256digest=" + "c" * 64 + "\n"
        "/unset mode\n"
        "./usr/lib/x time=1 size=1 sha256digest=nothex\n"
    )
    e = pa.parse_mtree(text)
    assert "/.BUILDINFO" not in e
    assert e["/usr/bin/sudo"]["mode"] == 0o4755 and e["/usr/bin/sudo"]["size"] == 261304
    assert e["/usr/bin/visudo"]["mode"] == 0o755          # from /set
    assert e["/usr/bin/sudoedit"]["type"] == "link"
    assert "/usr/share/doc/a b" in e                       # \040
    assert e["/usr/lib/x"]["mode"] is None and e["/usr/lib/x"]["sha256"] is None


def test_mtree_reader_is_bounded_and_refuses_symlinks(tmp_path, monkeypatch):
    big = tmp_path / "mtree"
    with gzip.open(big, "wb") as f:
        f.write(b"#" * 2048)
    monkeypatch.setattr(pa, "MAX_MTREE_BYTES", 1024)
    with pytest.raises(ValueError):
        pa.read_mtree(str(big))
    link = tmp_path / "link"
    link.symlink_to(big)
    with pytest.raises(OSError):
        pa.read_mtree(str(link))


def test_critical_selection():
    f = {"type": "file", "mode": 0o755}
    assert pa.is_critical("/usr/bin/anything", {"type": "file", "mode": 0o4755})
    assert pa.is_critical("/usr/lib/security/pam_unix.so", f)
    assert pa.is_critical("/usr/lib/ld-linux-x86-64.so.2", f)
    assert pa.is_critical("/usr/lib/libc.so.6", f)
    assert pa.is_critical("/usr/bin/nft", f)
    assert not pa.is_critical("/usr/bin/visudo", f)
    assert not pa.is_critical("/usr/bin/sudoedit", {"type": "link", "mode": 0o777})


def test_every_executor_binary_is_critical():
    from security.host_executor import ALLOWED_EXECUTABLES
    assert set(ALLOWED_EXECUTABLES) <= pa.CRITICAL_PATHS


# --- verification ---

def test_untouched_tree_is_clean(tree):
    r = pa.attest(fs_root=str(tree))
    assert r["findings"] == [] and r["unowned_privileged"] == [] and r["unowned_scan_complete"]
    assert r["checked"] == 2 and r["packages"] == 2      # sudo (setuid) + PAM; visudo not critical


@pytest.mark.parametrize("tamper,kind", [
    (lambda p: p.write_bytes(b"\x7fELF sudX"), "content_changed"),          # same size
    (lambda p: p.write_bytes(b"\x7fELF sudo backdoor"), "content_changed"),
    (lambda p: os.chmod(p, 0o755), "mode_changed"),
    (lambda p: p.unlink(), "missing"),
    (lambda p: (p.unlink(), p.symlink_to("/bin/sh")), "type_changed"),
])
def test_tampered_setuid_binary_is_reported(tree, tamper, kind):
    tamper(tree / "usr/bin/sudo")
    (f,) = pa.attest(fs_root=str(tree))["findings"]
    assert f["path"] == "/usr/bin/sudo" and f["kind"] == kind and f["package"] == "sudo-1.9-1"


def test_setuid_bit_added_to_a_pam_module_is_reported(tree):
    os.chmod(tree / "usr/lib/security/pam_unix.so", 0o4755)
    (f,) = pa.attest(fs_root=str(tree))["findings"]
    assert f["kind"] == "mode_changed" and "4755" in f["detail"]


def test_record_without_digest_is_unverifiable_never_clean(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"x")
    assert pa.verify_entry(str(p), {"type": "file", "mode": 0o644, "sha256": None})["kind"] == "unverifiable"


def test_owner_change_is_reported(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"x")
    os.chmod(p, 0o644)
    entry = {"type": "file", "mode": 0o644, "uid": UID + 1, "gid": GID, "size": 1,
             "sha256": hashlib.sha256(b"x").hexdigest()}
    assert pa.verify_entry(str(p), entry)["kind"] == "owner_changed"


def test_planted_setuid_file_owned_by_no_package_is_found(tree):
    planted = tree / "usr/local/bin/helper"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"x")
    os.chmod(planted, 0o4755)
    r = pa.attest(fs_root=str(tree))
    assert r["unowned_privileged"] == ["/usr/local/bin/helper"]


def test_unowned_scan_does_not_follow_directory_symlinks_and_is_bounded(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "s").write_bytes(b"x")
    os.chmod(outside / "s", 0o4755)
    usr = tmp_path / "usr"
    usr.mkdir()
    (usr / "link").symlink_to(outside)
    assert pa.scan_unowned_privileged(set(), (str(usr),)) == ([], True)
    for i in range(5):
        (usr / f"f{i}").write_bytes(b"")
    assert pa.scan_unowned_privileged(set(), (str(usr),), max_entries=3)[1] is False


def test_packaged_tmpfiles_rule_may_change_mode_but_not_content(tree):
    # libutempter ships "z /usr/lib/utempter/utempter 2755 root utmp"; the mtree says 2711.
    _install(tree, "libutempter-1.2-1", {"/usr/lib/utempter/utempter": (b"\x7fELF ut", 0o2711)})
    os.chmod(tree / "usr/lib/utempter/utempter", 0o2755)
    conf = tree / "usr/lib/tmpfiles.d"
    conf.mkdir(parents=True)
    (conf / "libutempter.conf").write_text(f"z /usr/lib/utempter/utempter 2755 {UID} {GID}\n")
    assert pa.attest(fs_root=str(tree))["findings"] == []
    (tree / "usr/lib/utempter/utempter").write_bytes(b"\x7fELF uX")
    (f,) = pa.attest(fs_root=str(tree))["findings"]
    assert f["kind"] == "content_changed"


def test_admin_tmpfiles_rules_are_not_trusted(tree):
    os.chmod(tree / "usr/bin/sudo", 0o6755)
    etc = tree / "etc/tmpfiles.d"
    etc.mkdir(parents=True)
    (etc / "x.conf").write_text("z /usr/bin/sudo 6755 - -\n")
    (f,) = pa.attest(fs_root=str(tree))["findings"]
    assert f["kind"] == "mode_changed"


def test_tmpfiles_parser_ignores_modifiers_and_other_types(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "a.conf").write_text("d /usr/x 0755 - -\nz /usr/y ~0755 - -\nz /etc/z 0600 - -\nZ /usr/w 0700 0 0\n")
    assert pa.tmpfiles_overrides((str(d),)) == {"/usr/w": {"mode": 0o700, "uid": 0, "gid": 0}}


# --- monitor ---

@pytest.fixture
def health(monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    return seen


def test_monitor_reports_each_finding_once_and_again_when_it_changes(tree, health, caplog):
    events = []
    mon = pa.PackageAttestationMonitor(fs_root=str(tree), log_event=lambda **kw: events.append(kw))
    mon.run_once()
    assert events == [] and health["package_attestation"][0] == "active"
    os.chmod(tree / "usr/bin/sudo", 0o755)
    with caplog.at_level(logging.CRITICAL, logger="security.package_attestation"):
        mon.run_once()
        mon.run_once()
    assert [e["event_type"] for e in events] == [pa.EVENT_ALTERED]
    assert events[0]["actor"] == "/usr/bin/sudo" and events[0]["disposition"] == "observed"
    assert "1 altered" in health["package_attestation"][1]
    (tree / "usr/bin/sudo").write_bytes(b"\x7fELF backdoor!")
    mon.run_once()
    assert len(events) == 2 and "content_changed" in events[1]["payload_hash"]


def test_monitor_reports_unowned_privileged_files(tree, health):
    events = []
    planted = tree / "usr/bin/x"
    planted.write_bytes(b"x")
    os.chmod(planted, 0o2755)
    pa.PackageAttestationMonitor(fs_root=str(tree), log_event=lambda **kw: events.append(kw)).run_once()
    assert [e["event_type"] for e in events] == [pa.EVENT_UNOWNED]


def test_monitor_failure_is_failed_not_silent(health):
    def boom():
        raise RuntimeError("cannot list /var/lib/pacman/local")
    pa.PackageAttestationMonitor(attest_fn=boom, log_event=lambda **kw: None).run_once()
    assert health["package_attestation"][0] == "failed"


def test_monitor_never_locks_down(tree, health, monkeypatch):
    import security.policy_gate as gate
    monkeypatch.setattr(gate, "trigger_lockdown", lambda *a, **k: pytest.fail("alert-only"))
    os.chmod(tree / "usr/bin/sudo", 0o755)
    pa.PackageAttestationMonitor(fs_root=str(tree), log_event=lambda **kw: None).run_once()


def test_runs_wait_for_a_pacman_transaction(tmp_path):
    lock = tmp_path / "var/lib/pacman/db.lck"
    lock.parent.mkdir(parents=True)
    assert not pa.pacman_busy(str(lock))
    lock.write_text("")
    assert pa.pacman_busy(str(lock))
    src = Path(pa.__file__).read_text()
    assert "if pacman_busy(self.fs_root + PACMAN_LOCK):" in src


def test_gate_runs_the_monitor_and_it_reports_health():
    gate = Path("security/policy_gate.py").read_text()
    assert "PackageAttestationMonitor()" in gate and "package_attestation.stop()" in gate
    assert "package_attestation" in control_health.GATE_CONTROLS
    assert "import config" not in Path(pa.__file__).read_text()
