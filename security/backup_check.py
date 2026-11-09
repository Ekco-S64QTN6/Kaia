"""Is there a recent, restorable backup? Checked, never inferred from presence.

The owner's backup is a full-system rsync mirror (default ``/mnt/backup_linux``,
override with ``KAIA_BACKUP_PATH``). Every BACKUP_CHECK_INTERVAL the gate:

* confirms the target is a mounted filesystem holding a system tree;
* dates the last backup run from the *ctime* of the mirror's top directories:
  ``rsync -a`` preserves mtimes but cannot preserve ctime, so ctime is when the
  backup last wrote there;
* samples restorability: files that have not changed on the live system since
  that run must be byte-identical in the mirror. A difference means the copy is
  corrupt or the disk is failing.

Reported as the ``backup`` control: active, degraded (older than WARN_DAYS, or
a sample differs), failed (older than FAIL_DAYS, or missing), off (no target
configured or present). Alert-only. Importable without ``core.config``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_TARGET = "/mnt/backup_linux"
BACKUP_CHECK_INTERVAL = 6 * 3600
WARN_DAYS = 14
FAIL_DAYS = 30
MARKER_DIRS = ("etc", "usr", "var")
SAMPLE_FILES = ("/etc/pacman.conf", "/etc/fstab", "/etc/hostname", "/etc/locale.gen",
                "/usr/bin/bash", "/usr/lib/os-release")
MAX_SAMPLE_BYTES = 64 * 1024 * 1024


def _sha256(path: str) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        h, total = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return h.hexdigest()
            total += len(chunk)
            if total > MAX_SAMPLE_BYTES:
                raise ValueError("sample too large")
            h.update(chunk)
    finally:
        os.close(fd)


def check(target: str = DEFAULT_TARGET, live_root: str = "", now: float = None,
          explicit: bool = False) -> tuple:
    """(state, detail) for the backup at ``target``."""
    now = time.time() if now is None else now
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return ("failed" if explicit else "off"), f"no backup target at {target}"
    except OSError as e:
        return "failed", f"cannot inspect {target}: {e}"
    if not stat.S_ISDIR(st.st_mode):
        return "failed", f"{target} is not a directory"
    if not os.path.ismount(target):
        return "failed", f"{target} is not mounted (the backup drive is absent or not mounted)"

    ctimes = []
    for name in MARKER_DIRS:
        try:
            ctimes.append(os.lstat(os.path.join(target, name)).st_ctime)
        except OSError:
            pass
    if not ctimes:
        return "failed", f"{target} holds no system tree ({', '.join(MARKER_DIRS)} missing)"
    last_run = max(ctimes)
    age_days = (now - last_run) / 86400

    mismatched, compared = [], 0
    for path in SAMPLE_FILES:
        live, copy = live_root + path, target + path
        try:
            live_st = os.lstat(live)
            if not stat.S_ISREG(live_st.st_mode) or live_st.st_ctime > last_run:
                continue  # changed since the backup: a difference would be expected
            compared += 1
            if _sha256(live) != _sha256(copy):
                mismatched.append(path)
        except FileNotFoundError:
            if os.path.lexists(live):
                mismatched.append(path)  # live file exists, the copy does not
        except (OSError, ValueError):
            continue

    detail = f"last backup {age_days:.1f} days ago ({time.strftime('%Y-%m-%d', time.localtime(last_run))})"
    detail += f"; {compared - len(mismatched)} of {compared} unchanged sample files match"
    if mismatched:
        return "degraded", detail + f"; differ in the backup: {', '.join(mismatched)} (corrupt copy or failing disk)"
    if age_days > FAIL_DAYS:
        return "failed", detail + f"; older than {FAIL_DAYS} days"
    if age_days > WARN_DAYS:
        return "degraded", detail + f"; older than {WARN_DAYS} days: run the backup"
    return "active", detail


class BackupMonitor:
    def __init__(self, target: str = None, interval: float = BACKUP_CHECK_INTERVAL):
        configured = os.environ.get("KAIA_BACKUP_PATH")
        self.target = target or configured or DEFAULT_TARGET
        self.explicit = bool(target or configured)
        self.interval = interval
        self._stop = threading.Event()

    def run_once(self) -> tuple:
        from security import control_health
        try:
            state, detail = check(self.target, explicit=self.explicit)
        except Exception as e:
            state, detail = "failed", f"backup check raised: {e}"
        control_health.report("backup", state, detail)
        if state in ("degraded", "failed"):
            logger.warning("Backup check: %s", detail)
        return state, detail

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.run_once()

    def start(self) -> None:
        self.run_once()
        threading.Thread(target=self._run, daemon=True, name="backup-check").start()

    def stop(self) -> None:
        self._stop.set()
