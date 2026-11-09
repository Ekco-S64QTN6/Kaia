"""Live state of each security control, published for the operator and a checker.

Every outage this host has had was a control that stopped without anyone being
told: the eBPF program did not compile for a week (honeypot watching with it),
kaia-restore-blocks failed on every boot, the HUD could not show a lockdown.
Each subsystem now reports what it is actually doing, the Policy Gate publishes
that to ``$STATE_DIR/control_health.json`` every few seconds, and
``scripts/kaia_health_check.py`` (run by ``kaia-health-check.timer``, outside the
gate) checks it is fresh.

States:

* ``active``   working as designed;
* ``degraded`` running with less coverage than designed (the detail says what);
* ``off``      not provisioned on this host, deliberately: not an alarm;
* ``failed``   meant to run and not running.

A control that declares a heartbeat interval and stops beating (its thread died
or hung) is evaluated as ``stale``, which is treated like ``failed``.

This module must stay importable without ``core.config`` (which exits without
the signing secret): the checker imports it to evaluate the file.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

STATES = ("active", "degraded", "off", "failed")
SEVERITY = {"active": 0, "off": 0, "degraded": 1, "failed": 2, "stale": 2}

PUBLISH_INTERVAL = 10.0
# The checker's limit for the file itself: three missed publishes.
MAX_FILE_AGE = 3 * PUBLISH_INTERVAL + 5.0
MAX_FILE_BYTES = 256 * 1024
MAX_DETAIL_CHARS = 300
MAX_CONTROLS = 64
FILE_NAME = "control_health.json"
SCHEMA = 1

# Controls the Policy Gate daemon runs; each must report its own state.
GATE_CONTROLS = (
    "policy_gate", "tamper_detection", "privilege_watch", "fim", "yara", "ebpf",
    "honeypot", "decoy_listeners", "discovery", "pacman_monitor", "package_attestation",
    "secret_exposure", "security_db", "audit_ledger", "firewall",
    "exec_baseline", "conn_baseline",
)


def stale_after(interval: float) -> float:
    """Seconds without a heartbeat before a control counts as stale."""
    return max(3.0 * interval, interval + 10.0)


class ControlHealth:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._controls: Dict[str, dict] = {}

    def report(self, name: str, state: str, detail: str = "",
               heartbeat_interval: Optional[float] = None) -> None:
        """Set a control's state. ``since`` changes only when the state does."""
        if state not in STATES:
            raise ValueError(f"unknown control state {state!r}")
        now = self._clock()
        with self._lock:
            entry = self._controls.get(name)
            if entry is None:
                if len(self._controls) >= MAX_CONTROLS:
                    logger.error("Control health registry full; not tracking %s", name)
                    return
                entry = {"since": now, "last_heartbeat": None, "heartbeat_interval": None}
                self._controls[name] = entry
            elif entry["state"] != state:
                entry["since"] = now
            entry["state"] = state
            entry["detail"] = str(detail)[:MAX_DETAIL_CHARS]
            if heartbeat_interval is not None:
                entry["heartbeat_interval"] = float(heartbeat_interval)
                # A new beating period starts now, not at the last beat.
                entry["last_heartbeat"] = now

    def heartbeat(self, name: str) -> None:
        """Called from a control's loop. Unknown names are ignored."""
        now = self._clock()
        with self._lock:
            entry = self._controls.get(name)
            if entry is not None:
                entry["last_heartbeat"] = now

    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            controls = {name: dict(entry) for name, entry in self._controls.items()}
        return {
            "schema": SCHEMA,
            "written_at": now,
            "pid": os.getpid(),
            "publish_interval": PUBLISH_INTERVAL,
            "controls": controls,
        }


_registry = ControlHealth()


def registry() -> ControlHealth:
    return _registry


def report(name: str, state: str, detail: str = "",
           heartbeat_interval: Optional[float] = None) -> None:
    """Never raises: health reporting must not take a control down with it."""
    try:
        _registry.report(name, state, detail, heartbeat_interval)
    except Exception as e:
        logger.error("Control health report for %s failed: %s", name, e)


def heartbeat(name: str) -> None:
    try:
        _registry.heartbeat(name)
    except Exception:
        pass


# --- publishing ---

def write_snapshot(path: str, snapshot: dict, max_bytes: Optional[int] = None) -> None:
    """Atomically replace ``path``: same-directory temp file, no-follow, 0640.

    Group read is for the dashboards (group kaiacord: the gate runs with
    Group=kaiacord, so new files carry that group).
    """
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    data = json.dumps(snapshot, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(data) > (MAX_FILE_BYTES if max_bytes is None else max_bytes):
        raise ValueError("snapshot exceeds size limit")
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(tmp, flags, 0o640)
    try:
        os.fchmod(fd, 0o640)
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class HealthPublisher:
    def __init__(self, path: str, health: Optional[ControlHealth] = None,
                 interval: float = PUBLISH_INTERVAL):
        self.path = path
        self.health = health or _registry
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failing = False

    def publish_once(self) -> bool:
        try:
            write_snapshot(self.path, self.health.snapshot())
        except Exception as e:
            if not self._failing:
                logger.error("Could not publish control health to %s: %s", self.path, e)
                self._failing = True
            return False
        if self._failing:
            logger.info("Control health publishing to %s recovered.", self.path)
            self._failing = False
        return True

    def start(self, publish_now: bool = True) -> None:
        if publish_now:
            self.publish_once()
        self._thread = threading.Thread(target=self._run, daemon=True, name="control-health")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.publish_once()

    def stop(self) -> None:
        self._stop.set()


# --- reading and evaluation (shared by the HUD and the checker) ---

def read_snapshot(path: str) -> dict:
    """Read the published file without following symlinks, size-bounded."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("control health file is not a regular file")
        if st.st_size > MAX_FILE_BYTES:
            raise ValueError("control health file is too large")
        chunks = []
        remaining = MAX_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("control health file is too large")
    snapshot = json.loads(raw.decode("utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("controls"), dict):
        raise ValueError("control health file has no controls")
    return snapshot


def _num(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate(snapshot: Optional[dict], now: float, error: str = "",
             max_age: float = MAX_FILE_AGE) -> dict:
    """Effective state of every control, plus the file's own freshness.

    Returns ``{"overall": state, "fresh": bool, "age": s|None, "problem": str,
    "controls": [{"name", "state", "detail", "since"}...]}`` where ``state`` is
    one of STATES or ``stale``. ``overall`` is the worst of them: a missing,
    unreadable or stale file is ``failed`` (the gate is down, hung, or cannot
    write), because then no control's state is known.
    """
    if snapshot is None:
        return {"overall": "failed", "fresh": False, "age": None,
                "problem": error or "no control health published", "controls": []}

    written = _num(snapshot.get("written_at"))
    age = None if written is None else now - written
    controls = []
    for name, entry in sorted(snapshot.get("controls", {}).items())[:MAX_CONTROLS]:
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        if state not in STATES:
            state = "failed"
        interval = _num(entry.get("heartbeat_interval"))
        beat = _num(entry.get("last_heartbeat"))
        detail = str(entry.get("detail") or "")[:MAX_DETAIL_CHARS]
        if state in ("active", "degraded") and interval and interval > 0 and written is not None:
            silent = written - (beat if beat is not None else written)
            if silent > stale_after(interval):
                state = "stale"
                detail = f"no heartbeat for {int(silent)} s (expected every {interval:g} s)"
                detail += f"; last report: {entry.get('detail') or ''}"[:MAX_DETAIL_CHARS]
        controls.append({
            "name": str(name)[:64],
            "state": state,
            "detail": detail,
            "since": _num(entry.get("since")),
        })

    problem = ""
    fresh = age is not None and -5.0 <= age <= max_age
    if age is None:
        problem = "control health file has no timestamp"
    elif age > max_age:
        problem = f"control health is {int(age)} s old: the Policy Gate is down or hung"
    elif age < -5.0:
        problem = "control health timestamp is in the future"

    worst = max((SEVERITY[c["state"]] for c in controls), default=0)
    overall = {0: "active", 1: "degraded", 2: "failed"}[worst]
    if not fresh:
        overall = "failed"
    return {"overall": overall, "fresh": fresh, "age": age, "problem": problem,
            "controls": controls}


def load_and_evaluate(path: str, now: Optional[float] = None) -> dict:
    now = time.time() if now is None else now
    try:
        snapshot = read_snapshot(path)
    except FileNotFoundError:
        return evaluate(None, now, f"{path} does not exist: the Policy Gate has not published")
    except Exception as e:
        return evaluate(None, now, f"cannot read {path}: {e}")
    return evaluate(snapshot, now)
