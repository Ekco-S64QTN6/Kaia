"""Behavioural baseline for executables: what has this host run before?

An intruder's tooling is usually a binary this host has never executed, often
from a writable location. The baseline learns every executed path for
LEARNING_SECONDS (silently), then reports:

* ``exec_unusual_location`` (high): a first-seen executable under /tmp, /var/tmp,
  /dev/shm, /run/user, a home directory, or given by a relative path;
* ``exec_first_seen`` (low): any other first-seen executable (a new package
  binary, typically).

Each path is reported once and then joins the baseline. Reports are rate-capped
so a burst cannot flood the event database. State persists in
``$STATE_DIR/exec_baseline.json`` so a restart does not restart learning.
Alert-only (axiom 5). Importable without ``core.config``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

FILE_NAME = "exec_baseline.json"
LEARNING_SECONDS = 7 * 86400
MAX_KNOWN = 50_000
MAX_REPORTS_PER_MINUTE = 30
PERSIST_INTERVAL = 60.0
UNUSUAL_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/", "/run/user/", "/home/", "/root/")
EVENT_FIRST_SEEN = "exec_first_seen"
EVENT_UNUSUAL = "exec_unusual_location"


import re

# Build tools (makepkg, pip, configure scripts) run executables from randomly
# named temporary directories. Collapse random-looking components so a build
# pattern is learned once instead of alerting on every build.
_RANDOM_COMPONENT = re.compile(r"(?=.*\d)[A-Za-z0-9_.+-]{6,}|tmp[A-Za-z0-9_]{4,}|[A-Za-z0-9]{8,}")


def baseline_key(path: str) -> str:
    """The path as the baseline remembers it."""
    if not path.startswith(UNUSUAL_PREFIXES):
        return path
    parts = path.split("/")
    # Keep the leading prefix (e.g. /home/<user>) and the final executable name.
    head = 3 if path.startswith(("/home/", "/run/user/")) else 2
    middle = ["*" if _RANDOM_COMPONENT.fullmatch(c) else c for c in parts[head:-1]]
    return "/".join(parts[:head] + middle + parts[-1:])


def classify(path: str) -> str:
    if not path.startswith("/") or "/../" in path or path.startswith(UNUSUAL_PREFIXES):
        return EVENT_UNUSUAL
    return EVENT_FIRST_SEEN


class ExecBaseline:
    def __init__(self, state_path: Optional[str], log_event=None, clock=time.time,
                 learning_seconds: float = LEARNING_SECONDS):
        self.state_path = state_path
        self._log_event = log_event
        self._clock = clock
        self.learning_seconds = learning_seconds
        self._lock = threading.Lock()
        self.known = set()
        self.started_at = clock()
        self._dirty = False
        self._last_persist = 0.0
        self._window = (0.0, 0)
        self.suppressed = 0
        self.reported_today = 0
        self._load()

    # --- persistence ---
    def _load(self) -> None:
        if not self.state_path:
            return
        try:
            fd = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                doc = json.loads(f.read(16 * 1024 * 1024))
            started = doc.get("started_at")
            known = doc.get("known")
            if isinstance(started, (int, float)) and not isinstance(started, bool) and isinstance(known, list):
                self.started_at = float(started)
                self.known = {p for p in known if isinstance(p, str)}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Exec baseline %s unreadable (%s); learning starts again", self.state_path, e)

    def persist(self, force: bool = False) -> None:
        if not self.state_path:
            return
        now = self._clock()
        with self._lock:
            if not (self._dirty or force) or (not force and now - self._last_persist < PERSIST_INTERVAL):
                return
            doc = {"started_at": self.started_at, "known": sorted(self.known)}
            self._dirty, self._last_persist = False, now
        from security.control_health import write_snapshot  # atomic, no-follow, 0640
        try:
            write_snapshot(self.state_path, doc, max_bytes=16 * 1024 * 1024)
        except Exception as e:
            logger.error("Could not persist exec baseline: %s", e)

    # --- observation ---
    @property
    def learning(self) -> bool:
        return self._clock() - self.started_at < self.learning_seconds

    def observe(self, path: str, comm: str = "", pid: int = 0) -> Optional[str]:
        """Record one exec; returns the event type reported, if any."""
        if not path:
            return None
        key = baseline_key(path)
        with self._lock:
            if key in self.known:
                return None
            if len(self.known) >= MAX_KNOWN:
                return None
            self.known.add(key)
            self._dirty = True
            if self._clock() - self.started_at < self.learning_seconds:
                return None
            event_type = classify(path)
            now = self._clock()
            start, count = self._window
            if now - start >= 60:
                start, count = now, 0
            if count >= MAX_REPORTS_PER_MINUTE:
                self.suppressed += 1
                self._window = (start, count)
                return None
            self._window = (start, count + 1)
            self.reported_today += 1
        if event_type == EVENT_UNUSUAL:
            logger.warning("First-seen executable from an unusual location: %s (%s, PID %d)", path, comm, pid)
        log = self._log_event
        if log is None:
            from security.db import log_security_event as log
        try:
            log(event_type=event_type, source="exec_baseline", actor=f"{comm} (PID {pid})"[:200],
                payload_hash=path[:300], disposition="observed", session_id="system_protection")
        except Exception as e:
            logger.error("Could not record %s: %s", event_type, e)
        return event_type

    def health(self) -> tuple:
        if self.learning:
            left = self.learning_seconds - (self._clock() - self.started_at)
            return "active", f"learning: {len(self.known)} executables, {int(left // 3600)} h left"
        detail = f"{len(self.known)} known executables; {self.reported_today} first-seen reported"
        if self.suppressed:
            detail += f"; {self.suppressed} suppressed by the rate cap"
        return "active", detail
