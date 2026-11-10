"""The firewall as the gate sees it, published so the dashboards need not ask.

The dashboards read the ruleset (rule counts, lockdown state) every few seconds.
Each read was a Policy Gate request, and every request is audited, so an open
dashboard wrote thousands of ledger rows a day (#105). Summarising those rows
would make the audit trail incomplete; instead the gate reads the ruleset itself
every FIREWALL_STATE_INTERVAL seconds (an internal observation, like its other
telemetry, not a request) and writes it to ``$STATE_DIR/firewall_state.json``
(atomic, no-follow, 0640, group kaiacord). Every request that is made is still
audited.

Pure parsing helpers live here too. Importable without ``core.config``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

FILE_NAME = "firewall_state.json"
FIREWALL_STATE_INTERVAL = 10.0
MAX_AGE = 3 * FIREWALL_STATE_INTERVAL
MAX_RULESET_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = MAX_RULESET_BYTES + 64 * 1024

LOCKDOWN_TABLE_RE = re.compile(r"^\s*table\s+inet\s+kaia_lockdown\s*\{")
KAIA_BLOCK_TABLE_RE = re.compile(r"^\s*table\s+inet\s+kaia_block\s*\{")


def lockdown_state_from_ruleset(success: bool, ruleset: str) -> str:
    """'active' / 'inactive' from `nft list ruleset` output, 'unknown' without it.

    The lockdown is the presence of `table inet kaia_lockdown` (kaia-lockdown.sh).
    `systemctl is-active kaia-lockdown.service` cannot show it: the unit is a
    oneshot without RemainAfterExit.
    """
    if not success:
        return "unknown"
    for line in (ruleset or "").splitlines():
        if LOCKDOWN_TABLE_RE.match(line):
            return "active"
    return "inactive"


def kaia_block_rule_count(ruleset: str) -> int:
    """Drop rules inside `table inet kaia_block`."""
    count, depth, inside = 0, 0, False
    for line in (ruleset or "").splitlines():
        if not inside and KAIA_BLOCK_TABLE_RE.match(line):
            inside, depth = True, 0
        if inside:
            depth += line.count("{") - line.count("}")
            if re.search(r"\b(drop|reject)\b", line) and "policy" not in line:
                count += 1
            if depth <= 0 and "}" in line:
                inside = False
    return count


_HOOK = re.compile(r"\bhook\s+(input|forward|output)\b")


def drop_rule_counts(ruleset: str) -> dict:
    """Drop/reject *rules* per hook (input, forward, output), across all tables.

    A chain is classified by its ``type ... hook <name>`` line, not its name,
    and the ``policy drop;`` on that line is a chain default, not a rule: the
    old counter included it, so an empty forward chain read as one "block".
    """
    counts = {"input": 0, "forward": 0, "output": 0}
    hook = None
    for line in (ruleset or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("chain "):
            hook = None
            continue
        if stripped == "}":
            hook = None
            continue
        m = _HOOK.search(stripped)
        if m and stripped.startswith("type "):
            hook = m.group(1)
            continue
        if hook and not stripped.startswith("#") and re.search(r"\b(drop|reject)\b", stripped):
            counts[hook] += 1
    return counts


def write_state(path: str, success: bool, ruleset: str, now: Optional[float] = None) -> None:
    """Atomic same-directory replace; never follows a symlink at either path."""
    data = (ruleset or "").encode("utf-8", "replace")[:MAX_RULESET_BYTES].decode("utf-8", "ignore")
    body = json.dumps({"written_at": time.time() if now is None else now,
                       "success": bool(success), "ruleset": data},
                      separators=(",", ":")).encode("utf-8")
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o640)
    try:
        os.fchmod(fd, 0o640)
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
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


def read_state(path: str, now: Optional[float] = None, max_age: float = MAX_AGE) -> Optional[Tuple[bool, str]]:
    """(success, ruleset) if the gate published it within ``max_age``, else None."""
    now = time.time() if now is None else now
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE_BYTES:
            return None
        raw = b""
        while len(raw) <= MAX_FILE_BYTES:
            chunk = os.read(fd, 256 * 1024)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    written = doc.get("written_at")
    if isinstance(written, bool) or not isinstance(written, (int, float)) or not -5 <= now - written <= max_age:
        return None
    ruleset = doc.get("ruleset")
    if not isinstance(ruleset, str) or not isinstance(doc.get("success"), bool):
        return None
    return doc["success"], ruleset


class FirewallStateMonitor:
    """Inside the gate: read the ruleset every interval and publish it."""

    def __init__(self, path: str, read_ruleset, interval: float = FIREWALL_STATE_INTERVAL):
        self.path = path
        self._read = read_ruleset   # () -> (success, stdout, stderr)
        self.interval = interval
        self._stop = threading.Event()
        self._failing = False

    def run_once(self) -> None:
        from security import control_health
        try:
            success, stdout, stderr = self._read()
        except Exception as e:
            success, stdout, stderr = False, "", str(e)
        try:
            write_state(self.path, success, stdout if success else "")
        except Exception as e:
            if not self._failing:
                logger.error("Could not publish firewall state to %s: %s", self.path, e)
                self._failing = True
            control_health.report("firewall", "failed", f"cannot publish state: {e}", heartbeat_interval=self.interval)
            return
        self._failing = False
        if not success:
            control_health.report("firewall", "failed", f"nft list ruleset failed: {str(stderr)[:200]}",
                                  heartbeat_interval=self.interval)
            return
        lockdown = lockdown_state_from_ruleset(True, stdout)
        detail = (("LOCKDOWN ACTIVE (table inet kaia_lockdown)" if lockdown == "active" else "no lockdown")
                  + f"; {kaia_block_rule_count(stdout)} Kaia block rules")
        control_health.report("firewall", "active", detail, heartbeat_interval=self.interval)
        control_health.heartbeat("firewall")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.run_once()

    def start(self) -> None:
        self.run_once()
        threading.Thread(target=self._run, daemon=True, name="firewall-state").start()

    def stop(self) -> None:
        self._stop.set()
