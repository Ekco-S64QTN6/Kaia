import os
import sys
import time
import ctypes
import sqlite3
import hashlib
import logging
import threading
import select
import stat
import config
from security.db import log_security_event

logger = logging.getLogger(__name__)

# fanotify constants
FAN_CLASS_NOTIF = 0x00000000
FAN_CLOEXEC = 0x00000001
FAN_NONBLOCK = 0x00000002

FAN_MARK_ADD = 0x00000001
FAN_MARK_MOUNT = 0x00000010

FAN_MODIFY = 0x00000002
FAN_CLOSE_WRITE = 0x00000008
FAN_ONDIR = 0x40000000
FAN_CREATE = 0x00000100
FAN_ATTRIB = 0x00000004

O_RDONLY = 0x00000000
O_LARGEFILE = 0x00008000

# fanotify_mark(2) takes a dirfd. The code previously passed -1, which is not a
# valid descriptor; AT_FDCWD is the documented "relative to cwd" value. It is
# ignored when pathname is absolute, but passing a bogus fd is a latent trap.
AT_FDCWD = -100

# Bound the in-memory byte snapshot used for YARA. Oversized files are
# explicitly reported as skipped instead of silently reopening a pathname.
MAX_FIM_SCAN_BYTES = 64 * 1024 * 1024

class FanotifyEventMetadata(ctypes.Structure):
    _fields_ = [
        ("event_len", ctypes.c_uint32),
        ("vers", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("metadata_len", ctypes.c_uint16),
        ("mask", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("pid", ctypes.c_int32),
    ]

class FIMDaemon:
    def __init__(self, yara_scanner=None):
        self.yara_scanner = yara_scanner
        self._stop_event = threading.Event()
        self._thread = None
        self.fan_fd = -1
        self.db_path = config.FIM_DB_PATH
        self._alerts_cache = []
        self._alerts_lock = threading.Lock()
        
        # Load libc for fanotify syscalls
        try:
            self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
            self.libc.fanotify_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
            self.libc.fanotify_init.restype = ctypes.c_int
            self.libc.fanotify_mark.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p]
            self.libc.fanotify_mark.restype = ctypes.c_int
        except Exception as e:
            logger.warning(f"Could not load libc for fanotify: {e}")
            self.libc = None

        # Resolve exclusions
        self.exclude_prefixes = [
            "/var/log",
            "/tmp",
            "/var/tmp",
            "/proc",
            "/sys",
            "/dev",
            os.path.expanduser("~/.cache")
        ]
        
    def start(self):
        """Starts the FIM daemon thread."""
        if not self.libc:
            logger.warning("FIMDaemon cannot start: libc fanotify binding unavailable.")
            return False
            
        conn = None
        try:
            # Init DB directory
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fim_events (
                    timestamp TEXT,
                    event_type TEXT,
                    pid INTEGER,
                    comm TEXT,
                    path TEXT,
                    yara_matches TEXT,
                    sha256 TEXT
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"FIM audit database initialization failed: {e}")
            return False
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

        # Initialize fanotify fd
        # We monitor FAN_CLASS_NOTIF. We must close event fds ourselves.
        fd = self.libc.fanotify_init(FAN_CLASS_NOTIF | FAN_CLOEXEC | FAN_NONBLOCK, O_RDONLY | O_LARGEFILE)
        if fd < 0:
            errno = ctypes.get_errno()
            logger.warning(f"fanotify_init failed (errno={errno}). Requires CAP_SYS_ADMIN.")
            return False
            
        self.fan_fd = fd

        # Mark the mount containing WORKSPACE_DIR.
        #
        # This previously failed with EINVAL on every start, so mount-wide FIM has
        # never actually run and the daemon silently fell back to the inotify
        # watchdog -- which is the polling-style blind spot INV-009 forbids.
        #
        # Both attempts failed because both masks contained events that the kernel
        # only permits when fanotify_init() was given FAN_REPORT_FID: the first had
        # FAN_CREATE/FAN_ONDIR, and the "basic" retry still kept FAN_ATTRIB.
        #
        # Rather than assume exactly which flag a given kernel objects to, degrade
        # progressively and record which tier actually took. FAN_MODIFY and
        # FAN_CLOSE_WRITE are the classic notification events and need no FID.
        workspace_path = config.WORKSPACE_DIR.encode("utf-8")
        mark_flags = FAN_MARK_ADD | FAN_MARK_MOUNT
        mask_tiers = [
            ("full (modify, close-write, create, attrib, ondir)",
             FAN_MODIFY | FAN_CLOSE_WRITE | FAN_CREATE | FAN_ATTRIB | FAN_ONDIR),
            ("no directory events (modify, close-write, attrib)",
             FAN_MODIFY | FAN_CLOSE_WRITE | FAN_ATTRIB),
            ("file content only (modify, close-write)",
             FAN_MODIFY | FAN_CLOSE_WRITE),
            ("minimal (modify)",
             FAN_MODIFY),
        ]

        res = -1
        for label, mask in mask_tiers:
            res = self.libc.fanotify_mark(self.fan_fd, mark_flags, mask, AT_FDCWD, workspace_path)
            if res >= 0:
                logger.info(f"fanotify mount mark established: {label}.")
                break
            err = ctypes.get_errno()
            logger.warning(f"fanotify_mark mount failed for mask '{label}' (errno={err}); degrading.")
            if err != 22:  # only EINVAL is worth retrying with a narrower mask
                break

        if res < 0:
            errno = ctypes.get_errno()
            logger.error(
                f"fanotify_mark mount failed at every mask tier (errno={errno}). "
                "Mount-wide FIM unavailable; falling back to the watchdog sentinel, "
                "which cannot see changes made between scans (see INV-009)."
            )
            os.close(self.fan_fd)
            self.fan_fd = -1
            return False
            
        self._thread = threading.Thread(target=self._run, daemon=True, name="fim-daemon")
        self._thread.start()
        logger.info(f"FIMDaemon started successfully monitoring mount of {config.WORKSPACE_DIR}")
        return True

    def stop(self):
        self._stop_event.set()
        if self.fan_fd >= 0:
            try:
                os.close(self.fan_fd)
            except Exception:
                pass
            self.fan_fd = -1

    def reload_rules(self, new_scanner):
        """Allows reload of compiled YARA ruleset dynamically."""
        self.yara_scanner = new_scanner
        logger.info("FIMDaemon YARA ruleset reloaded.")

    def get_recent_alerts(self, n=10) -> list:
        alerts = []
        if not os.path.exists(self.db_path):
            return alerts
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, pid, comm, path, yara_matches, sha256
                FROM fim_events
                ORDER BY rowid DESC LIMIT ?
            """, (n,))
            rows = cursor.fetchall()
            for row in rows:
                alerts.append({
                    "timestamp": row[0],
                    "event_type": row[1],
                    "pid": row[2],
                    "comm": row[3],
                    "path": row[4],
                    "yara_matches": row[5],
                    "sha256": row[6]
                })
        except Exception:
            pass
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        return alerts

    def _run(self):
        try:
            while not self._stop_event.is_set():
                if self.fan_fd < 0:
                    break
                r, w, x = select.select([self.fan_fd], [], [], 0.5)
                if self.fan_fd in r:
                    try:
                        data = os.read(self.fan_fd, 8192)
                    except BlockingIOError:
                        continue
                    except Exception as e:
                        logger.error(f"FIM read error: {e}")
                        break
                        
                    offset = 0
                    while offset + ctypes.sizeof(FanotifyEventMetadata) <= len(data):
                        metadata = FanotifyEventMetadata.from_buffer_copy(data[offset:])
                        if metadata.event_len == 0 or metadata.event_len < ctypes.sizeof(FanotifyEventMetadata):
                            break
                            
                        event_fd = metadata.fd
                        event_pid = metadata.pid
                        event_mask = metadata.mask
                        
                        if event_fd >= 0:
                            filepath = ""
                            try:
                                # Keep the kernel-supplied event descriptor alive
                                # until its contents have been captured and scanned.
                                # Reopening its pathname after closing this FD lets a
                                # rename/symlink replacement substitute different bytes.
                                filepath = os.readlink(f"/proc/self/fd/{event_fd}")
                                self._handle_event(filepath, event_pid, event_mask, event_fd)
                            except Exception as e:
                                logger.critical("FIM event processing failed for %s: %s", filepath or "<unknown>", e)
                                try:
                                    log_security_event(
                                        event_type="fim_event_processing_failure",
                                        source="fim_daemon",
                                        actor=filepath or "<unknown>",
                                        payload_hash="",
                                        disposition="blocked",
                                        session_id="system_protection",
                                    )
                                except Exception:
                                    logger.exception("Could not record FIM event processing failure")
                            finally:
                                try:
                                    os.close(event_fd)
                                except OSError:
                                    pass

                        offset += metadata.event_len
        except Exception as ex:
            logger.critical(f"FATAL Exception in FIMDaemon: {ex}")
            # Raise critical log to security DB to notify Policy Gate
            try:
                log_security_event(
                    event_type="fim_daemon_crash",
                    source="fim_daemon",
                    actor="FIMDaemon",
                    payload_hash="",
                    disposition="blocked",
                    session_id="system_protection"
                )
            except Exception:
                pass
            raise ex

    def _is_excluded_path(self, filepath: str) -> bool:
        """Check exclusions by path-component boundaries rather than prefixes."""
        normalized = os.path.abspath(filepath.removesuffix(" (deleted)"))
        for prefix in self.exclude_prefixes:
            root = os.path.abspath(prefix)
            try:
                if os.path.commonpath((normalized, root)) == root:
                    return True
            except ValueError:
                continue
        return False

    def _is_security_critical_path(self, filepath: str) -> bool:
        """Identify trust/control files whose skipped or failed scan is critical."""
        normalized = os.path.abspath(filepath.removesuffix(" (deleted)"))
        workspace = os.path.abspath(config.WORKSPACE_DIR)
        try:
            if os.path.commonpath((normalized, workspace)) != workspace:
                return normalized in {
                    "/etc/kaia/secret.env",
                    "/etc/systemd/system/kaia-policy-gate.service",
                    "/etc/systemd/system/kaia-lockdown.service",
                }
        except ValueError:
            return False
        relative = os.path.relpath(normalized, workspace).replace(os.sep, "/")
        return (
            relative == ".env"
            or relative == "core/config.py"
            or relative.startswith("security/")
            or relative in {
                "scripts/kaia-lockdown.sh",
                "scripts/kaia-baseline.sh",
                "scripts/kaia-unlock.sh",
            }
        )

    @staticmethod
    def _read_event_snapshot(event_fd: int) -> tuple:
        """Read bytes from the kernel-supplied FD under a strict memory ceiling."""
        metadata = os.fstat(event_fd)
        if not stat.S_ISREG(metadata.st_mode):
            return None, metadata.st_size, "not_regular_file"
        if metadata.st_size > MAX_FIM_SCAN_BYTES:
            return None, metadata.st_size, "size_limit_exceeded"

        content = bytearray()
        offset = 0
        while True:
            remaining = MAX_FIM_SCAN_BYTES + 1 - len(content)
            if remaining <= 0:
                return None, metadata.st_size, "size_limit_exceeded"
            chunk = os.pread(event_fd, min(1024 * 1024, remaining), offset)
            if not chunk:
                break
            content.extend(chunk)
            offset += len(chunk)
            if len(content) > MAX_FIM_SCAN_BYTES:
                return None, metadata.st_size, "size_limit_exceeded"
        return bytes(content), metadata.st_size, None

    def _handle_event(self, filepath: str, pid: int, mask: int, event_fd: int):
        if self._is_excluded_path(filepath):
            return

        comm = "unknown"
        try:
            with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="replace") as f:
                comm = f.read().strip()
        except Exception:
            pass

        if mask & FAN_CREATE:
            event_type = "create"
        elif mask & FAN_ATTRIB:
            event_type = "attrib"
        elif mask & FAN_MODIFY:
            event_type = "modify"
        else:
            event_type = "close_write"

        content, observed_size, skip_reason = self._read_event_snapshot(event_fd)
        sha = ""
        yara_matches = []

        if skip_reason:
            logger.warning(
                "FIM scan skipped for %s (%s; observed size=%s)",
                filepath, skip_reason, observed_size,
            )
            event_type = f"scan_skipped_{skip_reason}"
            try:
                log_security_event(
                    event_type="fim_scan_skipped",
                    source="fim_daemon",
                    actor=filepath,
                    payload_hash="",
                    disposition="blocked",
                    session_id="system_protection",
                )
            except Exception:
                logger.exception("Failed to record FIM scan skip for %s", filepath)
            if self._is_security_critical_path(filepath):
                try:
                    from security.policy_gate import trigger_lockdown
                    trigger_lockdown(f"FIM scan unavailable for protected file: {filepath} ({skip_reason})")
                except Exception as e:
                    logger.critical("Lockdown failed after FIM scan skip for %s: %s", filepath, e)
        else:
            # Header detection, signature scan and digest all consume precisely
            # the same captured content; none reopens the mutable pathname.
            sha = hashlib.sha256(content).hexdigest()
            is_executable = content.startswith(b"\x7fELF") or content.startswith(b"#!")
            if is_executable and self.yara_scanner:
                try:
                    matches = self.yara_scanner.match(data=content)
                    if matches:
                        yara_matches = [m.rule for m in matches]
                except Exception as e:
                    logger.error("YARA matching failed for %s: %s", filepath, e)
                    yara_matches = ["SCAN_ERROR"]
                    try:
                        log_security_event(
                            event_type="fim_yara_scan_error",
                            source="fim_daemon",
                            actor=filepath,
                            payload_hash=sha[:32] if sha else "",
                            disposition="blocked",
                            session_id="system_protection",
                        )
                    except Exception:
                        logger.exception("Failed to record YARA scan error for %s", filepath)
                    if self._is_security_critical_path(filepath):
                        try:
                            from security.policy_gate import trigger_lockdown
                            trigger_lockdown(f"YARA scanning failed for protected file: {filepath}")
                        except Exception as lockdown_error:
                            logger.critical(
                                "Lockdown failed after YARA scan error on %s: %s",
                                filepath, lockdown_error,
                            )

        yara_str = ",".join(yara_matches) if yara_matches else ""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fim_events (timestamp, event_type, pid, comm, path, yara_matches, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, event_type, pid, comm, filepath, yara_str, sha),
            )
            conn.commit()
        except Exception as e:
            logger.error("FIM audit database insert failed for %s: %s", filepath, e)
            try:
                log_security_event(
                    event_type="fim_audit_write_failure",
                    source="fim_daemon",
                    actor=filepath,
                    payload_hash=sha[:32] if sha else "",
                    disposition="blocked",
                    session_id="system_protection",
                )
            except Exception:
                logger.exception("Failed to report FIM audit write failure")
            if self._is_security_critical_path(filepath):
                try:
                    from security.policy_gate import trigger_lockdown
                    trigger_lockdown(f"FIM audit persistence failed for protected file: {filepath}")
                except Exception as lockdown_error:
                    logger.critical("Lockdown failed after FIM persistence error: %s", lockdown_error)
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

        if yara_matches:
            logger.critical("YARA result for %s: %s", filepath, yara_str)
            try:
                log_security_event(
                    event_type="fim_yara_match" if "SCAN_ERROR" not in yara_matches else "fim_yara_scan_error",
                    source="fim_daemon",
                    actor=filepath,
                    payload_hash=sha[:32] if sha else "",
                    disposition="blocked",
                    session_id="system_protection",
                )
            except Exception as e:
                logger.error("Failed to log YARA result for %s: %s", filepath, e)

            if "SCAN_ERROR" not in yara_matches and self._is_security_critical_path(filepath):
                try:
                    from security.policy_gate import trigger_lockdown
                    trigger_lockdown(f"FIM YARA Match on core file: {filepath}")
                except Exception as e:
                    logger.error("Failed to trigger lockdown for YARA match: %s", e)


