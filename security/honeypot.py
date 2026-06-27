import os
import sys
import time
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

def get_remote_ip_for_pid(pid: int) -> str:
    try:
        curr_pid = pid
        # Max depth of 5 to avoid infinite loops
        for _ in range(5):
            if curr_pid <= 1:
                break
            path = f"/proc/{curr_pid}/net/tcp"
            if os.path.exists(path):
                with open(path, "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            remote_addr = parts[2]
                            if remote_addr != "00000000" and not remote_addr.endswith(":0000"):
                                hex_ip = remote_addr.split(":")[0]
                                ip_bytes = bytes.fromhex(hex_ip)
                                # Little endian parse
                                ip = socket.inet_ntoa(ip_bytes)
                                if ip != "127.0.0.1":
                                    return ip
            # Walk up to parent
            stat_path = f"/proc/{curr_pid}/stat"
            if os.path.exists(stat_path):
                with open(stat_path, "r") as f:
                    content = f.read()
                    # Find last parenthesis (end of command name) to parse fields correctly
                    parent_idx = content.rfind(")")
                    if parent_idx != -1:
                        fields = content[parent_idx+2:].split()
                        curr_pid = int(fields[1]) # ppid is index 1 of remaining fields
                    else:
                        break
            else:
                break
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

    def start(self, ebpf_engine=None):
        logger.info("Starting Honeypot Coordinator...")
        
        # 1. Create file decoys
        for path, content in self.decoys.items():
            try:
                # Ensure directories exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        f.write(content)
                    logger.info(f"Deployed filesystem decoy: {path}")
                    
                # Compute and store SHA-256
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    h.update(f.read())
                self.decoy_hashes[path] = h.hexdigest()
            except Exception as e:
                logger.error(f"Failed to deploy honeypot file {path}: {e}")

        # 2. Register with eBPF engine
        if ebpf_engine:
            ebpf_engine.register_honeypot_paths(list(self.decoys.keys()))
            
            # Start eBPF openat check thread
            t = threading.Thread(target=self._check_ebpf_honeypots, args=(ebpf_engine,), daemon=True, name="honeypot-file-check")
            t.start()
            self.threads.append(t)

        # 3. Start namespace decoy listeners
        if self.libc and os.path.exists(f"/var/run/netns/{self.ns_name}"):
            for port in self.decoy_ports:
                t = threading.Thread(target=self._run_port_listener, args=(port,), daemon=True, name=f"decoy-listener-{port}")
                t.start()
                self.threads.append(t)
        else:
            logger.warning(f"Network namespace decoy listeners disabled (namespace '{self.ns_name}' not found or libc unavailable).")

    def stop(self):
        self.stop_event.set()
        
        # Remove file decoys
        for path in self.decoys.keys():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed filesystem decoy: {path}")
                except Exception as e:
                    logger.error(f"Failed to remove decoy file {path}: {e}")

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
                        event_type="honeypot_file_access",
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
