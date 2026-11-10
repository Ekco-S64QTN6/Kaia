"""Recent eBPF telemetry, published by the gate for the HUD.

Only the root Policy Gate can load eBPF probes. The web HUD used to create its
own engine anyway: as an unprivileged user that meant a clang compile on every
start, logged errors, and a process sampler instead of real exec/open/unlinkat
events. The gate now writes its newest events (already sanitised by the engine)
to ``$STATE_DIR/telemetry_recent.json`` every PUBLISH_INTERVAL seconds (atomic,
no-follow, 0640, group kaiacord), and the HUD reads them through
``PublishedTelemetry``, which offers the engine's read API.

Importable without ``core.config``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

FILE_NAME = "telemetry_recent.json"
PUBLISH_INTERVAL = 2.0
MAX_AGE = 10.0
PER_STREAM = 60
MAX_BYTES = 2 * 1024 * 1024
STREAMS = {
    "execs": "get_recent_execs",
    "opens": "get_recent_opens",
    "deletes": "get_recent_deletes",
    "connections": "get_recent_connections",
}


class TelemetryPublisher:
    """In the gate: copy the engine's recent events to the published file."""

    def __init__(self, engine, path: str, interval: float = PUBLISH_INTERVAL):
        self.engine, self.path, self.interval = engine, path, interval
        self._stop = threading.Event()
        self._failing = False

    def publish_once(self, now: Optional[float] = None) -> None:
        from security.control_health import write_snapshot
        doc = {"written_at": time.time() if now is None else now,
               "live": bool(getattr(self.engine, "bpf", None))}
        for key, getter in STREAMS.items():
            try:
                doc[key] = list(getattr(self.engine, getter)(PER_STREAM))
            except Exception:
                doc[key] = []
        try:
            write_snapshot(self.path, doc, max_bytes=MAX_BYTES)
            self._failing = False
        except Exception as e:
            if not self._failing:
                logger.error("Could not publish telemetry to %s: %s", self.path, e)
                self._failing = True

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.publish_once()

    def start(self) -> None:
        self.publish_once()
        threading.Thread(target=self._run, daemon=True, name="telemetry-publish").start()

    def stop(self) -> None:
        self._stop.set()


class PublishedTelemetry:
    """In the HUD: the engine's read API, served from the gate's published file."""

    def __init__(self, path: str, max_age: float = MAX_AGE, clock=time.time):
        self.path, self.max_age, self._clock = path, max_age, clock
        self._cache, self._cache_at = None, -1.0

    def _doc(self) -> Optional[dict]:
        now = self._clock()
        if self._cache_at >= 0 and now - self._cache_at < 1.0:
            return self._cache
        self._cache_at, self._cache = now, None
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:
                return None
            raw = os.read(fd, MAX_BYTES + 1)
        finally:
            os.close(fd)
        try:
            doc = json.loads(raw)
        except ValueError:
            return None
        written = doc.get("written_at") if isinstance(doc, dict) else None
        if isinstance(written, bool) or not isinstance(written, (int, float)) or not -5 <= now - written <= self.max_age:
            return None
        self._cache = doc
        return doc

    @property
    def bpf(self):
        """Truthy when the gate's probes are live and it published recently."""
        doc = self._doc()
        return bool(doc and doc.get("live"))

    def _stream(self, key: str, n: int) -> list:
        doc = self._doc()
        items = doc.get(key) if doc else None
        return [e for e in items if isinstance(e, dict)][-n:] if isinstance(items, list) else []

    def get_recent_execs(self, n=20):
        return self._stream("execs", n)

    def get_recent_opens(self, n=20):
        return self._stream("opens", n)

    def get_recent_deletes(self, n=20):
        return self._stream("deletes", n)

    def get_recent_connections(self, n=20):
        return self._stream("connections", n)

    def get_privilege_escalations(self, n=10):
        return []

    def start(self):
        return False

    def stop(self):
        pass
