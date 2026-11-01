"""Canary flag files and the eBPF honeypot fixes they depend on."""

import importlib.util
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security import ebpf_telemetry
from security.ebpf_telemetry import EBPFTelemetryEngine

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("install_canaries", ROOT / "scripts/install_canaries.py")
install_canaries = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(install_canaries)


# ── eBPF program ───────────────────────────────────────────────────────────
def test_bpf_program_includes_sock_header():
    # Without it struct sock is incomplete on current kernels and the whole
    # program fails to compile, silently disabling every honeypot probe.
    assert "#include <net/sock.h>" in ebpf_telemetry.bpf_text


def test_bpf_program_matches_honeypots_by_inode():
    text = ebpf_telemetry.bpf_text
    assert "kprobe__security_file_open" in text
    assert "BPF_HASH(honeypot_inodes" in text
    assert "0x80000000" in text


def test_kernel_dev_encoding():
    st_dev = os.makedev(259, 1)  # e.g. nvme0n1p1
    assert EBPFTelemetryEngine._kernel_dev(st_dev) == (259 << 20) | 1


class _Table:
    def __init__(self, event):
        self._event = event

    def event(self, _data):
        return self._event


def _hit(engine, pid, filename=b"", flags=0):
    ev = SimpleNamespace(pid=pid, comm=b"agent", filename=filename, flags=flags)
    engine.bpf = {"honeypot_events": _Table(ev)}
    engine._handle_honeypot_event(0, b"", 0)


def test_inode_hit_is_translated_to_the_registered_path():
    engine = EBPFTelemetryEngine()
    engine.honeypot_inode_paths = ["/etc/api_keys.json", "/home/u/flag.txt"]
    _hit(engine, 4242, flags=0x80000000 | 1)  # relative open: no usable filename
    hits = engine.get_honeypot_hits()
    assert [h["filename"] for h in hits] == ["/home/u/flag.txt"]


def test_one_open_seen_by_both_probes_is_reported_once():
    engine = EBPFTelemetryEngine()
    engine.honeypot_inode_paths = ["/home/u/flag.txt"]
    _hit(engine, 77, filename=b"/home/u/flag.txt", flags=0)       # path probe
    _hit(engine, 77, flags=0x80000000 | 0)                          # inode probe
    _hit(engine, 78, flags=0x80000000 | 0)                          # different process
    assert len(engine.get_honeypot_hits()) == 2


# ── coordinator ────────────────────────────────────────────────────────────
class _FakeEngine:
    def __init__(self):
        self.paths, self.inodes = [], []

    def register_honeypot_paths(self, paths):
        self.paths += paths

    def register_honeypot_inodes(self, paths):
        self.inodes += paths
        return len(paths)

    def get_honeypot_hits(self):
        return []


def test_coordinator_watches_installed_canaries_only(tmp_path, monkeypatch):
    real = tmp_path / "flag.txt"
    real.write_text("FLAG{x}\n")
    link = tmp_path / "root.txt"
    link.symlink_to(real)
    missing = tmp_path / "missing.txt"
    monkeypatch.setenv("KAIA_CANARY_PATHS", f"{real}:{link}:{missing}")

    from security.honeypot import HoneypotCoordinator
    coord = HoneypotCoordinator()
    coord.decoys = {}
    engine = _FakeEngine()
    coord.start(ebpf_engine=engine)
    try:
        assert coord.active_canary_paths == [str(real)]
        assert str(real) in engine.paths and str(real) in engine.inodes
        assert str(link) not in engine.inodes
    finally:
        coord.stop()


# ── installer helper ───────────────────────────────────────────────────────
def _template(tmp_path):
    t = tmp_path / "tpl.txt"
    t.write_text("FLAG{stop___CANARY_ID__}\nid: __CANARY_ID__\n")
    return t


def test_new_canary_gets_unique_id_and_reinstall_keeps_it(tmp_path):
    tpl, target = _template(tmp_path), tmp_path / "flag.txt"
    uid, gid = os.getuid(), os.getgid()
    first = install_canaries.install(target, tpl, uid, gid, 0o644)
    assert first.startswith("KAIA-CANARY-") and first in target.read_text()
    assert install_canaries.install(target, tpl, uid, gid, 0o644) == first
    assert oct(target.stat().st_mode & 0o777) == "0o644"


def test_owners_original_flag_is_backed_up_then_replaced(tmp_path):
    tpl, target = _template(tmp_path), tmp_path / "flag.txt"
    original = "AUTONOMOUS AGENT — STOP AND REPORT\nold text\n"
    target.write_text(original)
    cid = install_canaries.install(target, tpl, os.getuid(), os.getgid(), 0o644)
    assert cid and cid in target.read_text()
    assert (tmp_path / "flag.txt.pre-kaia").read_text() == original


def test_unknown_or_symlinked_files_are_never_overwritten(tmp_path):
    tpl = _template(tmp_path)
    real = tmp_path / "flag.txt"
    real.write_text("someone's real notes\n")
    assert install_canaries.install(real, tpl, os.getuid(), os.getgid(), 0o644) == ""
    assert real.read_text() == "someone's real notes\n"
    link = tmp_path / "root.txt"
    link.symlink_to(real)
    assert install_canaries.install(link, tpl, os.getuid(), os.getgid(), 0o600) == ""
    assert link.is_symlink()


def test_shipped_canary_texts_carry_the_id_in_the_flag():
    for name in ("user_flag.txt", "root_flag.txt"):
        text = (ROOT / "security/canaries" / name).read_text()
        assert text.startswith("FLAG{") and "__CANARY_ID__}" in text.splitlines()[0]
        assert text.count("__CANARY_ID__") >= 2


# ── unit + installer wiring ────────────────────────────────────────────────
def test_gate_unit_can_load_ebpf_and_sees_only_the_canaries():
    unit = (ROOT / "scripts/kaia-policy-gate.service").read_text()
    for line in ("CapabilityBoundingSet=", "AmbientCapabilities="):
        caps = next(l for l in unit.splitlines() if l.startswith(line))
        assert "CAP_BPF" in caps and "CAP_PERFMON" in caps
        assert "CAP_SYS_ADMIN" not in caps  # separate, documented decision
    assert "ProtectHome=tmpfs" in unit
    assert "BindReadOnlyPaths=-__OWNER_HOME__/flag.txt -/root/root.txt" in unit


def test_installer_places_canaries_before_enabling_the_gate():
    script = (ROOT / "scripts/install_services.sh").read_text()
    assert 's|__OWNER_HOME__|${OWNER_HOME}|g' in script
    assert script.index("install_canaries.py") < script.index("systemctl enable kaia-policy-gate.service")


# ── who gets blocked ───────────────────────────────────────────────────────
def _fake_proc(tmp_path, pid, ppid, sockets, tcp_rows):
    d = tmp_path / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "net").mkdir()
    for i, inode in enumerate(sockets):
        (d / "fd" / str(i + 3)).symlink_to(f"socket:[{inode}]")
    (d / "stat").write_text(f"{pid} (proc) S {ppid} 0 0\n")
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (d / "net" / "tcp").write_text(header + "".join(tcp_rows))
    (d / "net" / "tcp6").write_text(header)


def _row(remote_hex, inode):
    return f"   0: 0100007F:1F90 {remote_hex} 01 00000000:00000000 00:00000000 00000000  1000        0 {inode} 1\n"


# 1.2.3.4:443 and 5.6.7.8:22 in /proc/net/tcp little-endian notation
UNRELATED = _row("04030201:01BB", 999)
ATTACKER = _row("08070605:0016", 4242)


def test_process_without_sockets_blocks_nobody_even_if_host_has_connections(tmp_path):
    from security.honeypot import get_remote_ip_for_pid
    # The namespace-wide table lists someone else's connection (the old bug blamed it).
    _fake_proc(tmp_path, 500, 1, sockets=[], tcp_rows=[UNRELATED])
    assert get_remote_ip_for_pid(500, proc_root=str(tmp_path)) == "127.0.0.1"


def test_attacker_connection_held_by_parent_shell_is_found(tmp_path):
    from security.honeypot import get_remote_ip_for_pid
    rows = [UNRELATED, ATTACKER]
    _fake_proc(tmp_path, 600, 601, sockets=[], tcp_rows=rows)        # cat flag.txt
    _fake_proc(tmp_path, 601, 1, sockets=[4242], tcp_rows=rows)       # shell on the socket
    assert get_remote_ip_for_pid(600, proc_root=str(tmp_path)) == "5.6.7.8"


def test_attribution_stops_before_desktop_ancestors(tmp_path):
    from security.honeypot import get_remote_ip_for_pid
    rows = [ATTACKER]
    _fake_proc(tmp_path, 700, 701, [], rows)   # cat
    _fake_proc(tmp_path, 701, 702, [], rows)   # bash
    _fake_proc(tmp_path, 702, 703, [], rows)   # terminal
    _fake_proc(tmp_path, 703, 1, [4242], rows) # e.g. plasmashell with its own connection
    assert get_remote_ip_for_pid(700, proc_root=str(tmp_path)) == "127.0.0.1"
