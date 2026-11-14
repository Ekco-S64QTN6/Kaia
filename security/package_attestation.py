"""Installed-content attestation for the files that grant or exercise privilege.

pacman records every installed file's type, mode, owner, size and SHA-256 in
``/var/lib/pacman/local/<pkg>/mtree``. This module re-checks a critical subset of
them against the disk:

* every set-user-ID / set-group-ID file a package installs (sudo, su, pkexec...);
* PAM modules (``/usr/lib/security/``), the dynamic loader, libc, libpam, NSS;
* the binaries Kaia's own executor runs (nft, bwrap, systemd-run, ...) and the
  Python interpreter the gate runs on;

and looks for set-user-ID / set-group-ID files under ``/usr`` that no package
owns. A changed setuid binary, PAM module or loader is how a local root
backdoor survives; an unowned setuid file is a planted one.

Limits, stated plainly: the local package database is not signed, so root can
rewrite an mtree along with the file. This catches tampering by anything short
of a root attacker who also edits pacman's database, and every accidental or
out-of-band change. Verifying against signed packages in the cache is the next
step (ROADMAP). ``/etc`` is excluded: configuration changes legitimately.

Everything here only observes. It must stay importable without ``core.config``.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import re
import stat
import time
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PACMAN_LOCAL_DB = "/var/lib/pacman/local"
PACMAN_LOCK = "/var/lib/pacman/db.lck"
STALE_LOCK_SECONDS = 30 * 60            # no real transaction holds the lock this long

MAX_MTREE_BYTES = 64 * 1024 * 1024      # decompressed, per package
MAX_MTREE_LINE = 64 * 1024
MAX_HASH_BYTES = 512 * 1024 * 1024      # per file
MAX_UNOWNED_SCAN_ENTRIES = 2_000_000
MAX_FINDINGS = 500

# Exact paths (as installed, after /usr merge) always verified when a package owns them.
CRITICAL_PATHS = frozenset({
    # Kaia's executor (security/host_executor.py SYSTEM_BINARIES) and its sandbox.
    "/usr/bin/ss", "/usr/bin/ip", "/usr/bin/nft", "/usr/bin/sudo", "/usr/bin/systemctl",
    "/usr/bin/unshare", "/usr/bin/bwrap", "/usr/bin/systemd-run", "/usr/bin/systemd-nspawn",
    "/usr/bin/bash",
    # Authentication and session entry points.
    "/usr/bin/login", "/usr/bin/su", "/usr/bin/sshd", "/usr/lib/ssh/sshd-session",
    "/usr/lib/systemd/systemd", "/usr/bin/python3",
})
# tmpfiles rules are verified too: they are allowed to re-mode critical files.
CRITICAL_PREFIXES = ("/usr/lib/security/", "/usr/lib/tmpfiles.d/")
# Only package-shipped rules may legitimately change a packaged file's mode or
# owner (libutempter: "z /usr/lib/utempter/utempter 2755 root utmp"). Rules in
# /etc/tmpfiles.d are the admin's and are reported, not trusted.
TMPFILES_DIRS = ("/usr/lib/tmpfiles.d",)
CRITICAL_PATTERNS = (
    re.compile(r"/usr/lib/ld-linux[^/]*\.so[.0-9]*"),
    re.compile(r"/usr/lib/libc\.so[.0-9]*"),
    re.compile(r"/usr/lib/libpam(_misc)?\.so[.0-9]*"),
    re.compile(r"/usr/lib/libnss_[a-z]+\.so[.0-9]*"),
    re.compile(r"/usr/lib/libsystemd\.so[.0-9]*"),
    re.compile(r"/usr/bin/python3\.[0-9]+"),
)

_ESCAPE = re.compile(r"\\([0-7]{3}|\\)")


def _unescape(path: str) -> str:
    """mtree (vis(3)) path escapes: \\ooo octal bytes and \\\\."""
    raw = _ESCAPE.sub(lambda m: "\\" if m.group(1) == "\\" else chr(int(m.group(1), 8)), path)
    try:
        return raw.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def parse_mtree(text: str) -> Dict[str, dict]:
    """Absolute path -> {type, mode, uid, gid, size, sha256, link} for one package.

    Implements the subset libarchive writes for pacman: ``#`` comments, ``/set``
    and ``/unset`` defaults, and ``./path key=value...`` entries. Malformed
    values are dropped from the entry (it is then not verifiable, never "fine").
    """
    defaults: Dict[str, str] = {}
    entries: Dict[str, dict] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or len(line) > MAX_MTREE_LINE:
            continue
        fields = line.split()
        if fields[0] == "/set":
            for kv in fields[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    defaults[k] = v
            continue
        if fields[0] == "/unset":
            for k in fields[1:]:
                defaults.pop(k, None)
            continue
        if not fields[0].startswith("./"):
            continue
        keys = dict(defaults)
        for kv in fields[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                keys[k] = v
        path = "/" + _unescape(fields[0][2:])
        if path.startswith("/."):
            continue  # .PKGINFO, .BUILDINFO, .INSTALL, .MTREE
        entry = {"type": keys.get("type", "file"), "link": keys.get("link")}
        for name, key, base in (("mode", "mode", 8), ("uid", "uid", 10), ("gid", "gid", 10), ("size", "size", 10)):
            try:
                entry[name] = int(keys[key], base) if key in keys else None
            except ValueError:
                entry[name] = None
        digest = keys.get("sha256digest")
        entry["sha256"] = digest if digest and re.fullmatch(r"[0-9a-f]{64}", digest) else None
        entries[path] = entry
    return entries


def read_mtree(path: str) -> str:
    """Decompress one package's mtree, bounded, without following symlinks."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
        data = gz.read(MAX_MTREE_BYTES + 1)
    if len(data) > MAX_MTREE_BYTES:
        raise ValueError("mtree exceeds size limit")
    return data.decode("utf-8", "replace")


def iter_packages(local_db: str = PACMAN_LOCAL_DB) -> Iterable[Tuple[str, str]]:
    """(package dir name, mtree path) for every installed package."""
    try:
        names = sorted(os.listdir(local_db))
    except OSError as e:
        raise RuntimeError(f"cannot list {local_db}: {e}") from e
    for name in names:
        mtree = os.path.join(local_db, name, "mtree")
        if os.path.isfile(mtree):
            yield name, mtree


def _resolve_id(name: str, kind: str) -> Optional[int]:
    if name in ("", "-"):
        return None
    if name.isdigit():
        return int(name)
    try:
        if kind == "user":
            import pwd
            return pwd.getpwnam(name).pw_uid
        import grp
        return grp.getgrnam(name).gr_gid
    except (KeyError, ImportError):
        return None


def tmpfiles_overrides(dirs: Tuple[str, ...] = TMPFILES_DIRS) -> Dict[str, dict]:
    """Path -> {mode, uid, gid} set by ``z``/``Z`` rules in package tmpfiles.d.

    Only plain octal modes are honoured (no ``~``/``:`` modifiers), and ``Z``
    (recursive) only for the named path itself; anything else stays a finding.
    """
    overrides: Dict[str, dict] = {}
    for directory in dirs:
        try:
            names = sorted(n for n in os.listdir(directory) if n.endswith(".conf"))
        except OSError:
            continue
        for name in names:
            try:
                fd = os.open(os.path.join(directory, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(1024 * 1024)
            except OSError:
                continue
            for line in text.splitlines():
                fields = line.split()
                if len(fields) < 3 or fields[0] not in ("z", "Z") or not fields[1].startswith("/usr/"):
                    continue
                override = {}
                if re.fullmatch(r"[0-7]{3,4}", fields[2]):
                    override["mode"] = int(fields[2], 8)
                if len(fields) > 3:
                    uid = _resolve_id(fields[3], "user")
                    if uid is not None:
                        override["uid"] = uid
                if len(fields) > 4:
                    gid = _resolve_id(fields[4], "group")
                    if gid is not None:
                        override["gid"] = gid
                if override:
                    overrides.setdefault(fields[1], {}).update(override)
    return overrides


def is_critical(path: str, entry: dict) -> bool:
    if entry.get("type") != "file":
        return False
    mode = entry.get("mode")
    if mode is not None and mode & 0o6000:
        return True
    if path in CRITICAL_PATHS or path.startswith(CRITICAL_PREFIXES):
        return True
    return any(p.fullmatch(path) for p in CRITICAL_PATTERNS)


def sha256_no_follow(path: str, limit: int = MAX_HASH_BYTES) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        h = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("file exceeds hashing limit")
            h.update(chunk)
        return h.hexdigest()
    finally:
        os.close(fd)


def verify_entry(path: str, entry: dict, lstat=os.lstat, hasher=sha256_no_follow) -> Optional[dict]:
    """None when the file matches its package record, else a finding.

    An entry that lacks a digest or mode cannot be confirmed and is reported as
    ``unverifiable``: absence of evidence is never treated as a match.
    """
    def finding(kind, detail):
        return {"path": path, "kind": kind, "detail": detail}

    if entry.get("sha256") is None or entry.get("mode") is None:
        return finding("unverifiable", "package record has no digest or mode")
    try:
        st = lstat(path)
    except FileNotFoundError:
        return finding("missing", "file recorded by its package is gone")
    except OSError as e:
        return finding("unreadable", str(e))
    if not stat.S_ISREG(st.st_mode):
        return finding("type_changed", f"no longer a regular file (mode {stat.filemode(st.st_mode)})")
    problems, kinds = [], set()
    expected_mode = entry["mode"] & 0o7777
    actual_mode = stat.S_IMODE(st.st_mode)
    if actual_mode != expected_mode:
        problems.append(f"mode {actual_mode:04o}, package says {expected_mode:04o}")
        kinds.add("mode_changed")
    for name, actual in (("uid", st.st_uid), ("gid", st.st_gid)):
        if entry.get(name) is not None and actual != entry[name]:
            problems.append(f"{name} {actual}, package says {entry[name]}")
            kinds.add("owner_changed")
    if entry.get("size") is not None and st.st_size != entry["size"]:
        problems.append(f"size {st.st_size}, package says {entry['size']}")
        kinds.add("content_changed")
    else:
        try:
            digest = hasher(path)
        except OSError as e:
            return finding("unreadable", str(e))
        except ValueError as e:
            return finding("unverifiable", str(e))
        if digest != entry["sha256"]:
            problems.append("SHA-256 differs from the package record")
            kinds.add("content_changed")
    if not problems:
        return None
    # One finding per file, named by its most severe difference.
    kind = next(k for k in ("content_changed", "mode_changed", "owner_changed") if k in kinds)
    return finding(kind, "; ".join(problems))


def scan_unowned_privileged(owned: set, roots: Tuple[str, ...] = ("/usr",),
                            max_entries: int = MAX_UNOWNED_SCAN_ENTRIES) -> Tuple[List[str], bool]:
    """Set-user/group-ID regular files under ``roots`` that no package owns.

    Stays on one filesystem per root and never follows symlinks. Returns the
    paths and whether the scan completed within ``max_entries``.
    """
    found: List[str] = []
    seen = 0
    for root in roots:
        try:
            root_dev = os.lstat(root).st_dev
        except OSError:
            continue
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                it = os.scandir(directory)
            except OSError:
                continue
            with it:
                for entry in it:
                    seen += 1
                    if seen > max_entries:
                        return found, False
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(st.st_mode):
                        if st.st_dev == root_dev:
                            stack.append(entry.path)
                    elif stat.S_ISREG(st.st_mode) and st.st_mode & 0o6000 and entry.path not in owned:
                        found.append(entry.path)
                        if len(found) >= MAX_FINDINGS:
                            return found, False
    return found, True


def attest(fs_root: str = "", verify=verify_entry, scan=scan_unowned_privileged) -> dict:
    """One full pass. Returns counts, findings and unowned privileged files.

    ``fs_root`` prefixes every path read (tests point it at a fixture tree);
    reported paths stay as the package records them.
    """
    started = time.time()
    local_db = fs_root + PACMAN_LOCAL_DB
    overrides = tmpfiles_overrides(tuple(fs_root + d for d in TMPFILES_DIRS))
    owned: set = set()
    findings: List[dict] = []
    errors: List[str] = []
    checked = packages = 0
    for name, mtree_path in iter_packages(local_db):
        packages += 1
        try:
            entries = parse_mtree(read_mtree(mtree_path))
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        for path, entry in entries.items():
            owned.add(path)
            if not is_critical(path, entry):
                continue
            checked += 1
            if path in overrides:
                # Content is still checked against the package; only the
                # declared mode/owner are taken from the packaged rule.
                entry = dict(entry, **overrides[path])
            result = verify(fs_root + path, entry)
            if result is not None:
                result["path"] = path
            if result is not None and len(findings) < MAX_FINDINGS:
                result["package"] = name
                findings.append(result)
    unowned, complete = scan({fs_root + p for p in owned}, (fs_root + "/usr",))
    unowned = [p[len(fs_root):] for p in unowned]
    return {
        "packages": packages,
        "checked": checked,
        "findings": findings,
        "unowned_privileged": unowned,
        "unowned_scan_complete": complete,
        "errors": errors[:50],
        "duration": round(time.time() - started, 1),
    }


def pacman_busy(lock_path: str = PACMAN_LOCK) -> bool:
    """A transaction is running: files and records are changing under us."""
    return os.path.lexists(lock_path)


# --- running inside the Policy Gate ---

ALTERED_KINDS = ("content_changed", "mode_changed", "owner_changed", "type_changed", "missing")
EVENT_ALTERED = "package_file_altered"
EVENT_UNVERIFIABLE = "package_file_unverifiable"
EVENT_UNOWNED = "privileged_file_unowned"


class PackageAttestationMonitor:
    """Re-attest every ``interval`` seconds; report each distinct finding once.

    Alert-only (axiom 5). A finding that persists is not re-reported; one that
    changes (another kind or detail) is. Runs are postponed while a pacman
    transaction holds its lock, since files and records then disagree briefly.
    """

    def __init__(self, interval: float = 6 * 3600, first_delay: float = 600, fs_root: str = "",
                 log_event=None, clock=time.time, attest_fn=None):
        self.interval = interval
        self.first_delay = first_delay
        self.fs_root = fs_root
        self._log_event = log_event
        self._clock = clock
        self._attest = attest_fn or (lambda: attest(fs_root=self.fs_root))
        self._reported: Dict[Tuple[str, str], str] = {}
        self._stop = None
        self._thread = None
        self.last_report: Optional[dict] = None

    def _event(self, **kw) -> None:
        log = self._log_event
        if log is None:
            from security.db import log_security_event as log
        try:
            log(source="package_attestation", disposition="observed",
                session_id="system_protection", **kw)
        except Exception as e:
            logger.error("Could not record package attestation event: %s", e)

    def run_once(self) -> dict:
        from security import control_health
        try:
            report = self._attest()
        except Exception as e:
            logger.error("Package attestation failed: %s", e)
            control_health.report("package_attestation", "failed", f"last run raised: {e}",
                                  heartbeat_interval=60.0)
            return {}
        self.last_report = report
        current: Dict[Tuple[str, str], str] = {}
        for f in report["findings"]:
            current[(f["path"], "file")] = f"{f['kind']}: {f['detail']}"
        for path in report["unowned_privileged"]:
            current[(path, "unowned")] = "set-user/group-ID file owned by no package"
        for key, detail in current.items():
            if self._reported.get(key) == detail:
                continue
            path, what = key
            package = next((f.get("package", "") for f in report["findings"] if f["path"] == path), "")
            kind = detail.split(":", 1)[0]
            if what == "unowned":
                event_type = EVENT_UNOWNED
                logger.critical("Privileged file owned by no package: %s", path)
            elif kind in ALTERED_KINDS:
                event_type = EVENT_ALTERED
                logger.critical("Packaged file altered: %s (%s) %s", path, package, detail)
            else:
                event_type = EVENT_UNVERIFIABLE
                logger.warning("Packaged file not verifiable: %s (%s) %s", path, package, detail)
            self._event(event_type=event_type, actor=path[:200],
                        payload_hash=(f"{detail} [{package}]" if package else detail)[:300])
        self._reported = current

        altered = sum(1 for d in current.values() if d.split(":", 1)[0] in ALTERED_KINDS)
        unowned = len(report["unowned_privileged"])
        detail = (f"{report['checked']} critical files from {report['packages']} packages; "
                  f"{altered} altered, {unowned} unowned privileged, "
                  f"{len(current) - altered - unowned} unverifiable")
        state = "active"
        if report["errors"] or not report["unowned_scan_complete"]:
            state = "degraded"
            detail += f"; {len(report['errors'])} package records unreadable" if report["errors"] else "; unowned scan incomplete"
        control_health.report("package_attestation", state, detail, heartbeat_interval=60.0)
        return report

    def _postponed(self, busy_since: Optional[float]) -> float:
        """Note one postponed run; say so in control health once the lock looks stale.

        pacman leaves db.lck behind when it is killed. Without this the monitor
        waited on that file for ever while its heartbeat kept it "active".
        """
        from security import control_health
        now = self._clock()
        if busy_since is None:
            return now
        waited = now - busy_since
        if waited >= STALE_LOCK_SECONDS:
            control_health.report(
                "package_attestation", "degraded",
                f"postponed for {int(waited // 60)} min: {PACMAN_LOCK} exists "
                "(a pacman transaction, or a stale lock left by a killed pacman; "
                "no attestation runs until it is gone)",
                heartbeat_interval=60.0)
        return busy_since

    def _run(self) -> None:
        from security import control_health
        next_run = self._clock() + self.first_delay
        busy_since = None
        while not self._stop.wait(min(60.0, max(0.0, next_run - self._clock()))):
            control_health.heartbeat("package_attestation")
            if self._clock() < next_run:
                continue
            if pacman_busy(self.fs_root + PACMAN_LOCK):
                next_run = self._clock() + 60
                busy_since = self._postponed(busy_since)
                continue
            busy_since = None
            self.run_once()
            next_run = self._clock() + self.interval

    def start(self) -> None:
        import threading
        from security import control_health
        self._stop = threading.Event()
        control_health.report("package_attestation", "active",
                              f"first pass in {int(self.first_delay)} s, then every {int(self.interval / 3600)} h",
                              heartbeat_interval=60.0)
        self._thread = threading.Thread(target=self._run, daemon=True, name="package-attestation")
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()


# --- verification against the signed package archives ---------------------
# The local database is unsigned: root can rewrite an mtree together with the
# file it describes. The package archives in pacman's cache are signed by Arch
# packagers. This pass verifies each archive's signature with gpgv against
# pacman's public keyring, extracts the critical files from it, and compares
# them with both the disk and the local record. It runs from
# kaia-package-verify.timer as an unprivileged dynamic user, so archive parsing
# never happens inside the root gate.

PACMAN_CACHE = "/var/cache/pacman/pkg"
PACMAN_KEYRING = "/etc/pacman.d/gnupg/pubring.gpg"
GPGV = "/usr/bin/gpgv"
BSDTAR = "/usr/bin/bsdtar"
SIGNED_KINDS = ("differs_from_signed_package", "record_differs_from_signed_package")


def read_desc(path: str) -> dict:
    """%KEY% blocks of a local package desc file (first value of each)."""
    fields, key = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f.read(1024 * 1024).splitlines():
            if line.startswith("%") and line.endswith("%"):
                key = line.strip("%")
            elif key and line:
                fields.setdefault(key, line)
    return fields


def cached_package(name: str, version: str, arch: str, cache: str = PACMAN_CACHE):
    """(archive, signature) paths in the cache, or None when either is missing."""
    for ext in (".pkg.tar.zst", ".pkg.tar.xz"):
        pkg = os.path.join(cache, f"{name}-{version}-{arch}{ext}")
        if os.path.isfile(pkg) and not os.path.islink(pkg) and os.path.isfile(pkg + ".sig"):
            return pkg, pkg + ".sig"
    return None


def _run(argv, timeout=120, limit=MAX_HASH_BYTES):
    import subprocess
    proc = subprocess.run(argv, capture_output=True, timeout=timeout, env={"LC_ALL": "C", "PATH": "/usr/bin"})
    if len(proc.stdout) > limit:
        raise ValueError("output exceeds limit")
    return proc.returncode, proc.stdout, proc.stderr


def gpgv_verify(pkg: str, sig: str, keyring: str = PACMAN_KEYRING, run=_run) -> bool:
    rc, _, _ = run([GPGV, "--keyring", keyring, sig, pkg])
    return rc == 0


def archive_member_sha256(pkg: str, member: str, run=_run):
    """SHA-256 of one archive member, or None if absent."""
    rc, out, _ = run([BSDTAR, "-xOf", pkg, member])
    if rc != 0:
        return None
    return hashlib.sha256(out).hexdigest()


def signed_attest(fs_root: str = "", verify_sig=gpgv_verify, member_sha256=archive_member_sha256,
                  cache: str = PACMAN_CACHE) -> dict:
    """Critical files compared with their signed package archives.

    Findings: ``differs_from_signed_package`` (the file on disk is not what the
    packager signed), ``record_differs_from_signed_package`` (pacman's local
    record was changed), ``signature_invalid``, and ``no_signed_package`` (the
    archive or signature is not cached; informational, the cache may be cleaned).
    """
    started = time.time()
    findings, checked, packages, unverified = [], 0, 0, []
    for name, mtree_path in iter_packages(fs_root + PACMAN_LOCAL_DB):
        try:
            entries = parse_mtree(read_mtree(mtree_path))
        except Exception:
            continue
        critical = {p: e for p, e in entries.items() if is_critical(p, e)}
        if not critical:
            continue
        packages += 1
        try:
            desc = read_desc(os.path.join(os.path.dirname(mtree_path), "desc"))
            located = cached_package(desc["NAME"], desc["VERSION"], desc.get("ARCH", "x86_64"), fs_root + cache)
        except (OSError, KeyError):
            located = None
        if located is None:
            unverified.append(name)
            continue
        pkg, sig = located
        if not verify_sig(pkg, sig):
            findings.append({"path": pkg, "kind": "signature_invalid", "package": name,
                             "detail": "cached package signature does not verify against pacman's keyring"})
            continue
        for path, entry in critical.items():
            checked += 1
            signed = member_sha256(pkg, path.lstrip("/"))
            if signed is None:
                findings.append({"path": path, "kind": "differs_from_signed_package", "package": name,
                                 "detail": "file not present in the signed package"})
                continue
            if entry.get("sha256") and entry["sha256"] != signed:
                findings.append({"path": path, "kind": "record_differs_from_signed_package", "package": name,
                                 "detail": "pacman's local record does not match the signed package"})
            try:
                on_disk = sha256_no_follow(fs_root + path)
            except (OSError, ValueError) as e:
                findings.append({"path": path, "kind": "unreadable", "package": name, "detail": str(e)})
                continue
            if on_disk != signed:
                findings.append({"path": path, "kind": "differs_from_signed_package", "package": name,
                                 "detail": "SHA-256 differs from the file in the signed package"})
    return {"packages": packages, "checked": checked, "findings": findings[:MAX_FINDINGS],
            "no_signed_package": unverified, "duration": round(time.time() - started, 1)}
