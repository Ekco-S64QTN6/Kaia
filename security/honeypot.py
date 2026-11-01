import os
import sys
import time
import stat
import socket
import struct
import ctypes
import sqlite3
import hashlib
import logging
import threading
import subprocess
import config
from security.db import log_security_event


logger = logging.getLogger(__name__)

CLONE_NEWNET = 0x40000000

def _socket_inodes(pid: int, proc_root: str = "/proc") -> set:
    """Inodes of the sockets this process holds open (from /proc/<pid>/fd)."""
    inodes = set()
    fd_dir = os.path.join(proc_root, str(pid), "fd")
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(int(target[8:-1]))
    except OSError:
        pass  # no permission / process gone: attribute nothing
    return inodes


def _decode_proc_ip(hex_addr: str) -> str:
    """/proc/net/tcp{,6} address field (host-endian 32-bit words) -> IP string."""
    raw = bytes.fromhex(hex_addr)
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw[::-1])
    words = b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))
    ip = socket.inet_ntop(socket.AF_INET6, words)
    return ip[7:] if ip.startswith("::ffff:") and "." in ip else ip


def _parent_pid(pid: int, proc_root: str = "/proc") -> int:
    try:
        with open(os.path.join(proc_root, str(pid), "stat")) as f:
            content = f.read()
        return int(content[content.rfind(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def get_remote_ip_for_pid(pid: int, proc_root: str = "/proc") -> str:
    """Remote peer of a connection held by ``pid`` or one of its ancestors.

    /proc/<pid>/net/tcp lists *every* socket in the network namespace, not the
    process's own, so reading its first remote address blamed (and auto-blocked)
    an unrelated connection — an API, Discord, GitHub. Only rows whose socket
    inode the process actually holds are considered. When ownership cannot be
    established (no sockets, no permission to read the fd table) the answer is
    "127.0.0.1", which logs the event without blocking anyone.
    """
    try:
        curr_pid = pid
        # The process, its parent and grandparent: covers a reverse shell (socket
        # on the shell itself), ssh (cat -> bash -> sshd-session) and an AI agent
        # (cat -> tool shell -> agent holding its API connection), without
        # climbing into desktop processes (terminal, plasmashell) whose unrelated
        # connections would be blamed.
        for _ in range(3):
            if curr_pid <= 1:
                break
            owned = _socket_inodes(curr_pid, proc_root)
            if owned:
                for table in ("tcp", "tcp6"):
                    path = os.path.join(proc_root, str(curr_pid), "net", table)
                    try:
                        with open(path) as f:
                            rows = f.readlines()[1:]
                    except OSError:
                        continue
                    for row in rows:
                        parts = row.split()
                        if len(parts) < 10 or int(parts[9]) not in owned:
                            continue
                        hex_ip, _, hex_port = parts[2].partition(":")
                        if int(hex_port or "0", 16) == 0:
                            continue  # listening socket, no peer
                        ip = _decode_proc_ip(hex_ip)
                        if ip not in ("0.0.0.0", "::") and not ip.startswith("127.") and ip != "::1":
                            return ip
            curr_pid = _parent_pid(curr_pid, proc_root)
    except Exception as e:
        logger.error(f"Error resolving remote IP for PID {pid}: {e}")
    return "127.0.0.1"

class HoneypotCoordinator:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.stop_event = threading.Event()
        self.yara_scanner = None
        
        # Filesystem decoys
        self.decoys = {
            "/etc/api_keys.json": '{"aws_key": "AKIAIOSFODNN7EXAMPLE", "note": "decoy"}',
            "/var/backups/credentials.txt": 'db_password=example_decoy_do_not_use',
            os.path.expanduser("~/.ssh/authorized_keys.bak"): '# Authorized keys backup\n'
        }
        self.decoy_hashes = {}
        self.active_decoy_paths = []
        # Canary "flag" files (e.g. ~/flag.txt, /root/root.txt) are installed by
        # scripts/install_canaries.py and only *watched* here: never created,
        # modified or removed by the daemon. The unit passes them in.
        self.canary_paths = [
            p for p in os.environ.get("KAIA_CANARY_PATHS", "").split(":") if p.strip()
        ]
        self.active_canary_paths = []
        # Only files exclusively created by this coordinator may be cleaned up.
        self._managed_decoys = {}
        
        # Network namespace decoy setup
        self.ns_name = "ns_decoy"
        self.decoy_ports = [22, 443, 3306, 5432, 8080]
        self.status = {
            port: {
                "active": False,
                "last_trigger_ip": "None",
                "last_trigger_time": "None",
                "total_triggers": 0
            }
            for port in self.decoy_ports
        }
        
        self.threads = []
        self.libc = None
        try:
            self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except Exception as e:
            logger.warning(f"Could not load libc for netns setns call: {e}")

    @staticmethod
    def _hash_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _read_regular_file_no_follow(path: str, max_bytes: int) -> tuple:
        """Read a small regular file without following a final-component symlink."""
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                return None, metadata
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                amount = min(4096, max_bytes + 1 - len(chunks))
                if amount <= 0:
                    return None, metadata
                chunk = os.read(fd, amount)
                if not chunk:
                    break
                chunks.extend(chunk)
            if len(chunks) > max_bytes:
                return None, metadata
            return bytes(chunks), metadata
        finally:
            os.close(fd)

    def _prepare_decoy(self, path: str, content: str) -> bool:
        """Create a decoy exclusively; never adopt a real file based on size alone."""
        expected = content.encode("utf-8")
        try:
            # Parent paths are fixed system directories. Creating a missing parent
            # does not alter permissions or ownership on any existing directory.
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            # A previous run may have left a decoy behind. Reuse it only when its
            # contents exactly equal our known harmless payload; never own it for
            # cleanup because this coordinator did not create the file.
            try:
                existing, metadata = self._read_regular_file_no_follow(path, len(expected))
                if existing == expected and metadata is not None:
                    self.decoy_hashes[path] = self._hash_bytes(existing)
                    self.active_decoy_paths.append(path)
                    logger.info(
                        "Reusing verified existing decoy content at %s (not owned for cleanup).",
                        path,
                    )
                    return True
                logger.warning(
                    "Skipping existing non-decoy file at honeypot path %s; preserving it.",
                    path,
                )
            except (OSError, ValueError) as e:
                logger.warning("Skipping existing unsafe honeypot path %s: %s", path, e)
            return False
        except OSError as e:
            logger.warning("Could not create filesystem decoy %s: %s", path, e)
            return False

        created_metadata = None
        try:
            created_metadata = os.fstat(fd)
            if not stat.S_ISREG(created_metadata.st_mode):
                raise OSError("new decoy path is not a regular file")
            offset = 0
            while offset < len(expected):
                written = os.write(fd, expected[offset:])
                if written <= 0:
                    raise OSError("short write while creating decoy")
                offset += written
            os.fsync(fd)
            final_metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(final_metadata.st_mode)
                or (final_metadata.st_dev, final_metadata.st_ino)
                != (created_metadata.st_dev, created_metadata.st_ino)
            ):
                raise OSError("decoy file identity changed during creation")
            digest = self._hash_bytes(expected)
            self._managed_decoys[path] = {
                "device": final_metadata.st_dev,
                "inode": final_metadata.st_ino,
                "sha256": digest,
            }
            self.decoy_hashes[path] = digest
            self.active_decoy_paths.append(path)
            logger.info("Created filesystem decoy: %s", path)
            return True
        except Exception:
            # This descriptor refers to a file we exclusively created. Remove it
            # only if the pathname still names the same regular-file inode.
            if created_metadata is not None:
                try:
                    current = os.lstat(path)
                    if (
                        stat.S_ISREG(current.st_mode)
                        and (current.st_dev, current.st_ino)
                        == (created_metadata.st_dev, created_metadata.st_ino)
                    ):
                        os.unlink(path)
                except OSError:
                    logger.exception("Could not remove partially-created decoy %s", path)
            raise
        finally:
            os.close(fd)

    def start(self, ebpf_engine=None):
        logger.info("Starting Honeypot Coordinator...")
        if any(thread.is_alive() for thread in self.threads):
            logger.error("Honeypot workers from an earlier lifecycle are still active; refusing to start again.")
            return False
        self.threads.clear()
        self.stop_event.clear()
        self.active_decoy_paths = []
        self.decoy_hashes = {}
        self._managed_decoys = {}

        # A real file may already occupy a proposed decoy path. Never truncate,
        # overwrite, or register it just because it is empty or non-empty.
        for path, content in self.decoys.items():
            try:
                self._prepare_decoy(path, content)
            except Exception as e:
                logger.error("Failed to create filesystem decoy %s: %s", path, e)

        self._register_canaries()

        # Register only exact verified decoys, never every proposed path.
        if ebpf_engine:
            watched = list(self.active_decoy_paths) + list(self.active_canary_paths)
            if watched:
                ebpf_engine.register_honeypot_paths(watched)
                # Inode matching catches relative paths, ./x, grep -r, etc.
                register_inodes = getattr(ebpf_engine, "register_honeypot_inodes", None)
                if register_inodes:
                    count = register_inodes(watched)
                    logger.info("Honeypot inode watches active on %d of %d decoy/canary files.",
                                count, len(watched))
            t = threading.Thread(
                target=self._check_ebpf_honeypots,
                args=(ebpf_engine,),
                daemon=True,
                name="honeypot-file-check",
            )
            t.start()
            self.threads.append(t)

        # Network namespace listeners remain host-dependent. Defining port numbers
        # alone does not provision the namespace or activate network decoys.
        if self.libc and os.path.exists(f"/var/run/netns/{self.ns_name}"):
            for port in self.decoy_ports:
                t = threading.Thread(
                    target=self._run_port_listener,
                    args=(port,),
                    daemon=True,
                    name=f"decoy-listener-{port}",
                )
                t.start()
                self.threads.append(t)
        else:
            logger.warning(
                "Network namespace decoy listeners disabled (namespace '%s' not found or libc unavailable).",
                self.ns_name,
            )

    def _remove_managed_decoy_if_unchanged(self, path: str, expected: dict) -> bool:
        """Unlink only a file this process created whose identity and bytes are intact."""
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning("Preserving managed decoy %s; cannot safely open it: %s", path, e)
            return False
        try:
            metadata = os.fstat(fd)
            identity = (metadata.st_dev, metadata.st_ino)
            if not stat.S_ISREG(metadata.st_mode) or identity != (
                expected["device"], expected["inode"]
            ):
                logger.warning("Preserving replaced or non-regular decoy path %s.", path)
                return False
            chunks = bytearray()
            max_bytes = 4096
            while len(chunks) <= max_bytes:
                amount = min(1024, max_bytes + 1 - len(chunks))
                if amount <= 0:
                    return False
                chunk = os.read(fd, amount)
                if not chunk:
                    break
                chunks.extend(chunk)
            if len(chunks) > max_bytes or self._hash_bytes(bytes(chunks)) != expected["sha256"]:
                logger.warning("Preserving modified decoy path %s.", path)
                return False
        finally:
            os.close(fd)

        # Recheck pathname identity immediately before unlink; system parent
        # directories are root-controlled, and a replaced file is never removed.
        try:
            current = os.lstat(path)
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev, current.st_ino
            ) != (expected["device"], expected["inode"]):
                logger.warning("Preserving swapped decoy path %s.", path)
                return False
            os.unlink(path)
            logger.info("Removed coordinator-owned decoy: %s", path)
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning("Could not clean up coordinator-owned decoy %s: %s", path, e)
            return False

    def stop(self):
        self.stop_event.set()
        # Stop consumers before cleanup to avoid racing a background reader.
        for thread in list(self.threads):
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        still_alive = [
            thread for thread in self.threads
            if thread is not threading.current_thread() and thread.is_alive()
        ]
        if still_alive:
            # A worker may still be reading/logging an event. Preserve all files
            # rather than unlinking under a live consumer; a later stop can retry.
            logger.error(
                "Preserving managed honeypots because %d worker thread(s) did not stop.",
                len(still_alive),
            )
            return
        self.threads.clear()

        for path, metadata in list(self._managed_decoys.items()):
            self._remove_managed_decoy_if_unchanged(path, metadata)
        self._managed_decoys.clear()
        self.active_decoy_paths = []

    def _register_canaries(self) -> None:
        """Verify installed canary files (regular, not symlinks) and hash them."""
        self.active_canary_paths = []
        for path in self.canary_paths:
            try:
                content, metadata = self._read_regular_file_no_follow(path, 64 * 1024)
            except (OSError, ValueError) as e:
                logger.warning("Canary %s not watched: %s", path, e)
                continue
            if content is None:
                logger.warning("Canary %s not watched: not a small regular file.", path)
                continue
            self.decoy_hashes[path] = self._hash_bytes(content)
            self.active_canary_paths.append(path)
            logger.info("Watching canary flag file: %s", path)

    def get_decoy_status(self) -> dict:
        return dict(self.status)

    def _check_ebpf_honeypots(self, ebpf_engine):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            hits = ebpf_engine.get_honeypot_hits()
            for hit in hits:
                path = hit["filename"]
                pid = hit["pid"]
                comm = hit["comm"]
                
                logger.critical(f"HONEYPOT FILE ACCESS DETECTED: {path} accessed by {comm} (PID {pid})")
                
                # Derive source IP
                src_ip = get_remote_ip_for_pid(pid)
                
                # Log critical event
                try:
                    log_security_event(
                        event_type=(
                            "honeypot_canary_read"
                            if path in self.active_canary_paths
                            else "honeypot_file_access"
                        ),
                        source="honeypot_coordinator",
                        actor=f"{comm} (PID {pid})",
                        payload_hash=self.decoy_hashes.get(path, ""),
                        disposition="blocked",
                        session_id="system_protection"
                    )
                except Exception as e:
                    logger.error(f"Failed to log honeypot_file_access: {e}")
                    
                # Mitigate source IP if remote
                if src_ip != "127.0.0.1":
                    self._dispatch_block_ip(src_ip)

    def _run_port_listener(self, port: int):
        # Enter network namespace inside this thread only
        try:
            ns_path = f"/var/run/netns/{self.ns_name}"
            fd = os.open(ns_path, os.O_RDONLY)
            res = self.libc.setns(fd, CLONE_NEWNET)
            os.close(fd)
            if res < 0:
                logger.error(f"setns failed for {self.ns_name} on port {port}: errno={ctypes.get_errno()}")
                return
        except Exception as e:
            logger.error(f"Failed to enter namespace {self.ns_name} for port {port}: {e}")
            return

        # Start listening on the port
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.listen(10)
            s.settimeout(1.0)
            
            self.status[port]["active"] = True
            logger.info(f"Decoy listener running inside {self.ns_name} on port {port}")
            
            while not self.stop_event.is_set():
                try:
                    conn, addr = s.accept()
                    src_ip = addr[0]
                    conn.close()
                    
                    self._handle_port_trigger(port, src_ip)
                except socket.timeout:
                    continue
                except Exception:
                    break
        except Exception as e:
            logger.error(f"Decoy listener on port {port} crashed: {e}")
        finally:
            self.status[port]["active"] = False

    def _handle_port_trigger(self, port: int, src_ip: str):
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.critical(f"HONEYPOT DECOY PORT HIT: Port {port} accessed by {src_ip}!")
        
        # Update metrics
        self.status[port]["last_trigger_ip"] = src_ip
        self.status[port]["last_trigger_time"] = now_str
        self.status[port]["total_triggers"] += 1
        
        # Log to events DB
        try:
            log_security_event(
                event_type="honeypot_port_trigger",
                source="honeypot_coordinator",
                actor=f"Port {port}",
                payload_hash=hashlib.sha256(f"{src_ip}:{port}".encode()).hexdigest()[:32],
                disposition="blocked",
                session_id="system_protection"
            )
        except Exception as e:
            logger.error(f"Failed to log honeypot_port_trigger: {e}")

        # Dispatch immediate block_ip payload Exception
        if src_ip != "127.0.0.1":
            self._dispatch_block_ip(src_ip)

    def _dispatch_block_ip(self, src_ip: str):
        logger.info(f"Auto-mitigating honeypot offender IP: {src_ip}")
        from security.policy_gate import generate_capability_token
        
        token = generate_capability_token("block_ip", src_ip, duration_seconds=30)
        
        # Send IPC connection request to policy gate socket
        import json
        import uuid
        
        req_id = str(uuid.uuid4())
        payload = {
            "request_id": req_id,
            "action": "block_ip",
            "payload": {"target_ip": src_ip, "protocol": "all", "session_id": "honeypot_auto_block"},
            "capability_token": token
        }
        
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(config.POLICY_GATE_SOCKET)
            payload_bytes = json.dumps(payload).encode('utf-8')
            header = len(payload_bytes).to_bytes(4, byteorder='big')
            s.sendall(header + payload_bytes)
            s.close()
            logger.info(f"Auto-block IPC request dispatched successfully for {src_ip}.")
        except Exception as e:
            logger.error(f"Failed to dispatch auto-block IPC request for {src_ip}: {e}")
