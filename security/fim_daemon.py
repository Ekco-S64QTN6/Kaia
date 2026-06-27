import os
import sys
import time
import ctypes
import sqlite3
import hashlib
import logging
import threading
import select
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

O_RDONLY = 0x00000000
O_LARGEFILE = 0x00008000

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
        self.db_path = "/var/lib/secdaemon/fim_audit.db"
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
            
        # Init DB directory
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
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
            conn.close()
        except Exception as e:
            logger.error(f"FIM audit database initialization failed: {e}")
            return False

        # Initialize fanotify fd
        # We monitor FAN_CLASS_NOTIF. We must close event fds ourselves.
        fd = self.libc.fanotify_init(FAN_CLASS_NOTIF | FAN_CLOEXEC | FAN_NONBLOCK, O_RDONLY | O_LARGEFILE)
        if fd < 0:
            errno = ctypes.get_errno()
            logger.warning(f"fanotify_init failed (errno={errno}). Requires CAP_SYS_ADMIN.")
            return False
            
        self.fan_fd = fd

        # Mark mount containing WORKSPACE_DIR
        workspace_path = config.WORKSPACE_DIR.encode("utf-8")
        mask = FAN_MODIFY | FAN_CLOSE_WRITE | FAN_ONDIR
        res = self.libc.fanotify_mark(self.fan_fd, FAN_MARK_ADD | FAN_MARK_MOUNT, mask, -1, workspace_path)
        if res < 0:
            errno = ctypes.get_errno()
            logger.error(f"fanotify_mark mount failed (errno={errno}).")
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
                            try:
                                proc_path = f"/proc/self/fd/{event_fd}"
                                filepath = os.readlink(proc_path)
                                os.close(event_fd)
                                
                                # Process path
                                self._handle_event(filepath, event_pid, event_mask)
                            except Exception as e:
                                logger.error(f"Error handling FIM event metadata: {e}")
                                try:
                                    os.close(event_fd)
                                except Exception:
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

    def _handle_event(self, filepath: str, pid: int, mask: int):
        # Exclude directories/files by prefix
        for prefix in self.exclude_prefixes:
            if filepath.startswith(prefix):
                return
                
        # Gather comm
        comm = "unknown"
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                comm = f.read().strip()
        except Exception:
            pass

        # Check mask type
        event_type = "modify" if (mask & FAN_MODIFY) else "close_write"
        
        # Check ELF or shebang
        is_executable = False
        try:
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    hdr = f.read(4)
                    if hdr.startswith(b"\x7fELF") or hdr.startswith(b"#!"):
                        is_executable = True
        except Exception:
            pass
            
        yara_matches = []
        if is_executable and self.yara_scanner:
            try:
                matches = self.yara_scanner.match(filepath)
                if matches:
                    yara_matches = [m.rule for m in matches]
            except Exception as e:
                logger.error(f"YARA matching failed for {filepath}: {e}")

        # Compute SHA-256
        sha = ""
        if os.path.isfile(filepath):
            try:
                h = hashlib.sha256()
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                sha = h.hexdigest()
            except Exception:
                pass

        yara_str = ",".join(yara_matches) if yara_matches else ""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Write to local database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fim_events (timestamp, event_type, pid, comm, path, yara_matches, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ts, event_type, pid, comm, filepath, yara_str, sha))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"FIM audit database insert failed: {e}")

        # Trigger security alert on YARA matches
        if yara_matches:
            logger.critical(f"YARA Match on path {filepath}: {yara_str}")
            try:
                log_security_event(
                    event_type="fim_yara_match",
                    source="fim_daemon",
                    actor=filepath,
                    payload_hash=sha[:32] if sha else "",
                    disposition="blocked",
                    session_id="system_protection"
                )
            except Exception as e:
                logger.error(f"Failed to log fim_yara_match security event: {e}")
                
            # If yara match occurs in core security or configs, trigger lockdown
            if "security/" in filepath or "core/config.py" in filepath:
                try:
                    from security.policy_gate import trigger_lockdown
                    trigger_lockdown(f"FIM YARA Match on core file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to trigger lockdown for YARA match: {e}")
