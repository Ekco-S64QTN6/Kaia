"""Passive Arch Linux package-transaction monitoring.

This module observes completed pacman transactions from /var/log/pacman.log.
It never runs package-manager commands or automatically blocks upgrades. First
startup establishes a cursor at EOF so historical installs are not re-reported.
"""

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import config
from security.db import log_security_event

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "/var/log/pacman.log"
MAX_BYTES_PER_POLL = 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_PENDING_PACKAGES = 256

_ALPM_PREFIX = re.compile(r"^\[([^\]]+)\]\s+\[ALPM\]\s+(.*)$")
_PACKAGE_ACTION = re.compile(
    r"^(installed|upgraded|downgraded|reinstalled|removed) "
    r"([A-Za-z0-9@._+-]{1,128}) \((.{1,256})\)$"
)
_SAFE_VERSION = re.compile(r"[^A-Za-z0-9._+~:@<>/-]")


class PacmanLogMonitor:
    """Tail pacman.log safely and emit events only for completed transactions."""

    def __init__(
        self,
        log_path: Optional[str] = None,
        cursor_path: Optional[str] = None,
        poll_interval: float = 5.0,
        max_bytes_per_poll: int = MAX_BYTES_PER_POLL,
    ):
        self.log_path = os.path.abspath(
            log_path or os.environ.get("KAIA_PACMAN_LOG_PATH", DEFAULT_LOG_PATH)
        )
        self.cursor_path = os.path.abspath(
            cursor_path or os.path.join(str(config.STATE_DIR), "pacman-log.cursor.json")
        )
        self.poll_interval = max(0.25, float(poll_interval))
        self.max_bytes_per_poll = max(4096, min(int(max_bytes_per_poll), 16 * MAX_BYTES_PER_POLL))
        self._stop = threading.Event()
        self._poll_lock = threading.Lock()
        self._thread = None
        self._cursor = None
        self._cursor_error = False
        self._missing_warned = False
        self._load_cursor()

    def _load_cursor(self):
        try:
            with open(self.cursor_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError("cursor must be a JSON object")
            for field in ("device", "inode", "offset"):
                if isinstance(state.get(field), bool) or not isinstance(state.get(field), int):
                    raise ValueError(f"invalid cursor field: {field}")
            if min(state["device"], state["inode"], state["offset"]) < 0:
                raise ValueError("negative cursor field")
            pending = state.get("pending", [])
            if not isinstance(pending, list) or len(pending) > MAX_PENDING_PACKAGES:
                raise ValueError("invalid pending transaction")
            cleaned_pending = []
            for item in pending:
                if not isinstance(item, dict):
                    raise ValueError("invalid pending package record")
                operation = item.get("operation")
                package = item.get("package")
                version = item.get("version")
                payload_hash = item.get("payload_hash")
                if (
                    operation not in ("installed", "upgraded", "downgraded", "reinstalled", "removed")
                    or not isinstance(package, str)
                    or not re.fullmatch(r"[A-Za-z0-9@._+-]{1,128}", package)
                    or not isinstance(version, str)
                    or len(version) > 256
                    or not isinstance(payload_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", payload_hash)
                ):
                    raise ValueError("invalid pending package fields")
                cleaned_pending.append({
                    "operation": operation,
                    "package": package,
                    "version": version,
                    "payload_hash": payload_hash,
                })
            transaction_start = state.get("transaction_start")
            if transaction_start is not None and not isinstance(transaction_start, str):
                raise ValueError("invalid transaction_start")
            self._cursor = {
                "device": state["device"],
                "inode": state["inode"],
                "offset": state["offset"],
                "transaction_start": transaction_start,
                "pending": cleaned_pending,
            }
        except FileNotFoundError:
            self._cursor = None
        except Exception as exc:
            logger.error("Pacman monitor cursor is invalid; establishing a fresh tail: %s", exc)
            self._cursor = None
            self._cursor_error = True

    def _save_cursor(self):
        parent = os.path.dirname(self.cursor_path)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        fd = -1
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".pacman-cursor-", dir=parent)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1
                json.dump(self._cursor, f, separators=(",", ":"), sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.cursor_path)
            tmp_path = None
            dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

    def _emit(self, event_type: str, actor: str, payload_hash: str, session_id: str):
        log_security_event(
            event_type=event_type,
            source="pacman_log",
            actor=actor[:256],
            payload_hash=payload_hash,
            disposition="observed",
            session_id=session_id[:128] or "pacman_transaction",
        )

    def _consume_line(self, raw_line: bytes) -> int:
        """Update transaction state and return count of committed package events."""
        if len(raw_line) > MAX_LINE_BYTES:
            logger.warning("Skipping overlong pacman log line (%d bytes)", len(raw_line))
            return 0

        text = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        match = _ALPM_PREFIX.match(text)
        if not match:
            return 0
        timestamp, message = match.groups()
        timestamp = timestamp[:64]

        if message == "transaction started":
            self._cursor["transaction_start"] = timestamp
            self._cursor["pending"] = []
            return 0

        if "transaction failed" in message.lower() or "failed to commit transaction" in message.lower():
            self._cursor["transaction_start"] = None
            self._cursor["pending"] = []
            return 0

        if message == "transaction completed":
            pending = list(self._cursor["pending"])
            session = self._cursor["transaction_start"] or timestamp
            for item in pending:
                actor_version = _SAFE_VERSION.sub("?", item["version"])[:128]
                actor = f'{item["package"]} ({actor_version})'
                self._emit(
                    f'pacman_package_{item["operation"]}',
                    actor,
                    item["payload_hash"],
                    f"pacman:{session}",
                )
            self._cursor["transaction_start"] = None
            self._cursor["pending"] = []
            return len(pending)

        if not self._cursor["transaction_start"]:
            return 0
        action = _PACKAGE_ACTION.match(message)
        if not action:
            return 0
        operation, package, version = action.groups()
        pending = self._cursor["pending"]
        if len(pending) >= MAX_PENDING_PACKAGES:
            # Do not silently commit a partial transaction as complete evidence.
            logger.error(
                "Pacman transaction exceeds %d package actions; discarding pending audit batch",
                MAX_PENDING_PACKAGES,
            )
            self._cursor["transaction_start"] = None
            self._cursor["pending"] = []
            return 0
        pending.append({
            "operation": operation,
            "package": package,
            "version": version[:256],
            "payload_hash": hashlib.sha256(raw_line).hexdigest(),
        })
        return 0

    def poll_once(self) -> int:
        """Consume a bounded log slice and return the number of package events emitted."""
        with self._poll_lock:
            try:
                st = os.stat(self.log_path, follow_symlinks=False)
            except FileNotFoundError:
                if not self._missing_warned:
                    logger.warning("Pacman log not found at %s; package monitoring is unavailable", self.log_path)
                    self._missing_warned = True
                return 0
            except OSError as exc:
                logger.error("Could not stat pacman log %s: %s", self.log_path, exc)
                return 0

            if not stat.S_ISREG(st.st_mode):
                logger.error("Pacman log path is not a regular file: %s", self.log_path)
                return 0
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.log_path, flags)
            except OSError as exc:
                logger.error("Could not open pacman log %s: %s", self.log_path, exc)
                return 0

            try:
                current = os.fstat(fd)
                if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
                    logger.warning("Pacman log changed during open; retrying on next poll")
                    return 0

                if self._cursor is None:
                    # First run never replays all historic package operations as new alerts.
                    self._cursor = {
                        "device": current.st_dev,
                        "inode": current.st_ino,
                        "offset": current.st_size,
                        "transaction_start": None,
                        "pending": [],
                    }
                    self._save_cursor()
                    if self._cursor_error:
                        self._emit(
                            "pacman_monitor_cursor_reset",
                            "pacman-log.cursor.json",
                            "",
                            "pacman_monitor",
                        )
                        self._cursor_error = False
                    self._emit(
                        "pacman_monitor_baseline_established",
                        os.path.basename(self.log_path),
                        "",
                        "pacman_monitor",
                    )
                    return 0

                reset = (
                    self._cursor["device"] != current.st_dev
                    or self._cursor["inode"] != current.st_ino
                    or current.st_size < self._cursor["offset"]
                )
                if reset:
                    logger.warning("Pacman log rotated or truncated; resetting read offset")
                    self._cursor = {
                        "device": current.st_dev,
                        "inode": current.st_ino,
                        "offset": 0,
                        "transaction_start": None,
                        "pending": [],
                    }

                fd_offset = self._cursor["offset"]
                os.lseek(fd, fd_offset, os.SEEK_SET)
                data = os.read(fd, self.max_bytes_per_poll)
                if not data:
                    return 0
                final_newline = data.rfind(b"\n")
                if final_newline < 0:
                    # Keep the offset stable until a complete log record arrives.
                    return 0
                complete = data[:final_newline + 1]
                lines = complete.splitlines(keepends=True)
                emitted = 0
                for raw_line in lines:
                    emitted += self._consume_line(raw_line)
                self._cursor["offset"] = fd_offset + len(complete)
                self._cursor["device"] = current.st_dev
                self._cursor["inode"] = current.st_ino
                self._save_cursor()
                return emitted
            except Exception:
                logger.exception("Pacman log poll failed")
                return 0
            finally:
                os.close(fd)

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pacman-log-monitor")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_interval)
