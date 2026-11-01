"""Alert-only watch on the files that grant privilege on the host.

The tamper detector's lockdown tier covers Kaia's own code and units: a change
there means Kaia itself can no longer be trusted, so it isolates the host. The
files here are different. They change legitimately -- pacman adds system users
to /etc/passwd and /etc/group, the owner edits sudo rules -- so they must never
block startup or lock the host down. But they are exactly where an attacker
who reached root, or a hijacked agent with a sudo foothold, makes access
permanent: a NOPASSWD rule, a second uid-0 account, a user added to wheel, a
library injected through /etc/ld.so.preload.

So every change is recorded once, as an ``observed`` security event plus a log
line naming the lines added and removed. Nothing is blocked, nothing is locked.

The last-seen state is persisted under the state directory, so a change made
while the gate was stopped (or the host was off) is still reported at the next
start instead of silently becoming the new normal.
"""

import hashlib
import json
import logging
import os
import re
import stat
import tempfile

logger = logging.getLogger(__name__)

WATCH_FILES = ("/etc/sudoers", "/etc/passwd", "/etc/group", "/etc/ld.so.preload")
# Directories whose *members* are watched (a new file is a change).
WATCH_DIRS = ("/etc/sudoers.d",)
STATE_FILENAME = "privilege_watch.json"
STATE_VERSION = 1
MAX_FILE_BYTES = 256 * 1024
MAX_LINES_REPORTED = 12
PRIVILEGED_GROUPS = frozenset({"root", "wheel", "sudo", "admin"})

_NOPASSWD = re.compile(r"\bNOPASSWD\s*:")


def sudo_reads(name: str) -> bool:
    """Whether sudo's @includedir would read this file name.

    sudo skips names containing '.' or ending in '~' (sudoers(5)); visudo's
    temporary files are among them. Watching only what sudo reads keeps those
    transient files from raising alerts without opening a blind spot.
    """
    return bool(name) and "." not in name and not name.endswith("~")


class PrivilegeConfigWatcher:
    def __init__(self, state_path, files=WATCH_FILES, dirs=WATCH_DIRS, log_event=None):
        self.state_path = str(state_path)
        self.files = tuple(files)
        self.dirs = tuple(dirs)
        self._log_event = log_event
        self._state = None

    # ── reading ──────────────────────────────────────────────────────────
    @staticmethod
    def _read(path):
        """Return {"sha256", "lines"} for a regular file, None if absent.

        Opened without following symlinks; a symlink, directory or other
        non-regular object at a watched path is recorded as such (so replacing
        a file with a symlink is itself a change).
        """
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError as e:
            try:
                st = os.lstat(path)
            except OSError:
                return {"sha256": f"unreadable:{e.errno}", "lines": None}
            if not stat.S_ISLNK(st.st_mode):
                # Exists but cannot be opened (permissions): a coverage gap, not a type.
                return {"sha256": f"unreadable:{e.errno}", "lines": None}
            try:
                target = os.readlink(path)
            except OSError:
                target = "?"
            return {"sha256": f"symlink:{target}", "lines": None}
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return {"sha256": "special:", "lines": None}
            data = b""
            while len(data) <= MAX_FILE_BYTES:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(fd)
        digest = hashlib.sha256(data).hexdigest()
        if len(data) > MAX_FILE_BYTES:
            return {"sha256": f"oversize:{digest}", "lines": None}
        return {"sha256": digest, "lines": data.decode("utf-8", "replace").splitlines()}

    def snapshot(self) -> dict:
        snap = {}
        for path in self.files:
            entry = self._read(path)
            if entry is not None:
                snap[path] = entry
        for d in self.dirs:
            try:
                names = sorted(os.listdir(d))
            except FileNotFoundError:
                continue
            except OSError as e:
                snap[d + "/"] = {"sha256": f"unreadable:{e.errno}", "lines": None}
                continue
            for name in names:
                if sudo_reads(name):
                    entry = self._read(os.path.join(d, name))
                    if entry is not None:
                        snap[os.path.join(d, name)] = entry
        return snap

    # ── state persistence ────────────────────────────────────────────────
    def _load_state(self):
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.state_path, flags)
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning(f"Privilege watch: cannot open saved state {self.state_path}: {e}; starting fresh")
            return None
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get("version") != STATE_VERSION or not isinstance(doc.get("files"), dict):
                raise ValueError("unexpected format")
            return doc["files"]
        except Exception as e:
            logger.warning(f"Privilege watch: saved state {self.state_path} unreadable ({e}); starting fresh")
            return None

    def _save_state(self, snap):
        d = os.path.dirname(os.path.abspath(self.state_path)) or "."
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".privilege_watch.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"version": STATE_VERSION, "files": snap}, fh)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.state_path)
        except Exception as e:
            logger.error(f"Privilege watch: could not save state to {self.state_path}: {e}")

    # ── change reporting ─────────────────────────────────────────────────
    @staticmethod
    def _line_diff(old, new):
        old, new = old or [], new or []
        old_set, new_set = set(old), set(new)
        added = [l for l in new if l not in old_set]
        removed = [l for l in old if l not in new_set]
        return added, removed

    @staticmethod
    def _role(path):
        """What a watched file is, by name (works for any root, e.g. in tests)."""
        name = os.path.basename(path)
        if name == "sudoers" or os.path.basename(os.path.dirname(path)) == "sudoers.d":
            return "sudoers"
        return {"passwd": "passwd", "group": "group", "ld.so.preload": "preload"}.get(name, "other")

    @classmethod
    def _critical_reasons(cls, path, kind, added, new_entry):
        reasons = []
        role = cls._role(path)
        if role == "preload" and kind != "deleted":
            if new_entry and (new_entry.get("lines") is None or any(l.strip() and not l.lstrip().startswith("#") for l in new_entry["lines"])):
                reasons.append("library preload configured (every program will load it)")
        if role == "sudoers":
            for l in added:
                s = l.strip()
                if s and not s.startswith("#") and _NOPASSWD.search(s):
                    reasons.append(f"new passwordless sudo rule: {s[:160]}")
        if role == "passwd":
            for l in added:
                f = l.split(":")
                if len(f) >= 3 and f[2] == "0" and f[0] != "root":
                    reasons.append(f"new uid-0 account: {f[0]}")
        if role == "group":
            for l in added:
                f = l.split(":")
                if len(f) >= 4 and f[0] in PRIVILEGED_GROUPS and f[3].strip():
                    reasons.append(f"privileged group {f[0]} members now: {f[3].strip()[:120]}")
        return reasons

    def _report(self, old_snap, new_snap, while_stopped=False):
        changes = []
        when = " while Kaia was not running" if while_stopped else ""
        for path in sorted(set(old_snap) | set(new_snap)):
            a, b = old_snap.get(path), new_snap.get(path)
            if (a or {}).get("sha256") == (b or {}).get("sha256"):
                continue
            kind = "created" if a is None else "deleted" if b is None else "modified"
            added, removed = self._line_diff((a or {}).get("lines"), (b or {}).get("lines"))
            critical = self._critical_reasons(path, kind, added, b)
            summary = f"{path} {kind}{when} (+{len(added)}/-{len(removed)} lines)"
            detail = [f"    + {l[:200]}" for l in added[:MAX_LINES_REPORTED]]
            detail += [f"    - {l[:200]}" for l in removed[:MAX_LINES_REPORTED]]
            if len(added) > MAX_LINES_REPORTED or len(removed) > MAX_LINES_REPORTED:
                detail.append("    (more lines changed than shown)")
            if critical:
                logger.critical("PRIVILEGE CONFIG CHANGED (alert only, no lockdown): " + summary
                                + "".join(f"\n  !! {r}" for r in critical) + ("\n" + "\n".join(detail) if detail else ""))
            else:
                logger.warning("Privilege config changed (alert only): " + summary
                               + ("\n" + "\n".join(detail) if detail else ""))
            if self._log_event is not None:
                try:
                    self._log_event(
                        event_type="privilege_config_critical" if critical else "privilege_config_changed",
                        source="privilege_watch",
                        actor=f"{path} ({kind}{', while stopped' if while_stopped else ''})",
                        payload_hash=((b or {}).get("sha256") or "deleted")[:32],
                        disposition="observed",
                        session_id="system_protection",
                    )
                except Exception as e:
                    logger.error(f"Privilege watch: failed to record event for {path}: {e}")
            changes.append((path, kind, critical))
        return changes

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self):
        """Compare against the state saved last run, then begin from the current disk."""
        current = self.snapshot()
        previous = self._load_state()
        changes = []
        if previous is not None:
            changes = self._report(previous, current, while_stopped=True)
        else:
            logger.info(f"Privilege watch: first run, recording {len(current)} files as the starting state")
        self._save_state(current)
        self._state = current
        logger.info(f"Privilege watch active (alert only) on {len(current)} files: "
                    + ", ".join(sorted(current)))
        blind = sorted(p for p, e in current.items() if e["sha256"].startswith("unreadable:"))
        if blind:
            logger.warning("Privilege watch cannot read (changes there go unseen): " + ", ".join(blind))
        return changes

    def check(self):
        if self._state is None:
            return []
        current = self.snapshot()
        changes = self._report(self._state, current)
        if changes:
            self._save_state(current)
            self._state = current
        return changes
