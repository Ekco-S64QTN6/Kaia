#!/usr/bin/env python3
"""
Kaia Cyber-Sentinel Web Operations Center: Zero-Trust Bridge Daemon
===================================================================

High-performance aiohttp WebSocket and HTTP telemetry bridge streaming
immutable KaiamonSnapshots to the Cyber-Sentinel HUD interface.

Zero-Trust Invariants:
1. Strict Loopback Binding: Binds exclusively to 127.0.0.1. Rejects 0.0.0.0.
2. Air-Gapped Local Assets: Zero external CDN requests.
3. Content Security Policy (CSP): Enforced via HTTP middleware.
4. One-Way Telemetry Push: WebSocket stream is read-only outbound broadcast.
5. Deterministic Policy Gate Enforcement: Commands sent to /api/command serialize
   into Pydantic intents and validate capability tokens via Policy Gate Unix socket.
"""

import os
import sys
import time
import json
import uuid
import socket
import stat
import asyncio
import logging
import argparse
import threading
from datetime import datetime, timezone, timedelta, UTC
from typing import Dict, Any, List, Optional, Set, Tuple

import aiohttp
from aiohttp import web
import psutil

# Ensure repo root and core are in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
CORE_DIR = os.path.join(BASE_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

logger = logging.getLogger("kaia.web_bridge")

# Import Kaia collectors and snapshot definitions
from kaia_dashboard import (
    CollectorManager,
    KaiamonSnapshot,
    LogEntry,
    ServiceCollector,
    TelemetryCollector,
    query_gate_diagnostics,
)
from security.policy_gate import generate_capability_token
from security.ebpf_telemetry import EBPFTelemetryEngine


# ==================== CONSTANTS & INVARIANTS ====================

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TELEMETRY_BROADCAST_INTERVAL_SEC = 0.20  # 5 updates per second
ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

STATIC_DIR = os.path.join(CORE_DIR, "web", "static")


# ==================== PROCESS / SYSCALL FALLBACK SAMPLER ====================

class ProcessSyscallSampler:
    """
    Supplements eBPF kernel telemetry with userspace process activity sampling
    when BCC kernel probes are not available (e.g. running unprivileged/dev).
    """
    def __init__(self, max_history: int = 100) -> None:
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._known_pids: Set[int] = set()
        self._init_pids()

    def _init_pids(self) -> None:
        try:
            for p in psutil.process_iter(["pid"]):
                self._known_pids.add(p.info["pid"])
        except Exception:
            pass

    def sample(self) -> List[Dict[str, Any]]:
        """Detect newly launched processes as sys_enter_execve equivalents."""
        new_events = []
        now = time.time()
        try:
            current_pids = set()
            for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "username"]):
                pid = proc.info["pid"]
                current_pids.add(pid)
                if pid not in self._known_pids:
                    self._known_pids.add(pid)
                    comm = proc.info.get("name") or "unknown"
                    exe = proc.info.get("exe") or (proc.info.get("cmdline")[0] if proc.info.get("cmdline") else comm)
                    new_events.append({
                        "syscall": "sys_enter_execve",
                        "pid": pid,
                        "comm": comm,
                        "filename": exe,
                        "disposition": "observed",   # sampled, not evaluated
                        "timestamp": now,
                    })
            # Prune departed pids
            if len(self._known_pids) > 1000:
                self._known_pids = current_pids
        except Exception:
            pass

        with self._lock:
            self._history.extend(new_events)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            return list(self._history)


# ==================== HOST INTROSPECTION (top processes, listening sockets) ====================

class HostIntrospector:
    """Cheap, cached host views for the dashboard: top processes and listening sockets."""
    PROC_TTL_SEC = 2.0
    SOCK_TTL_SEC = 5.0
    LOOPBACK = ("127.", "::1", "localhost")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[int, psutil.Process] = {}
        self._top: List[Dict[str, Any]] = []
        self._top_at = 0.0
        self._listen: List[Dict[str, Any]] = []
        self._listen_at = 0.0

    def top_processes(self, n: int = 12) -> List[Dict[str, Any]]:
        with self._lock:
            if time.time() - self._top_at < self.PROC_TTL_SEC:
                return self._top
            rows, alive = [], {}
            ncpu = psutil.cpu_count() or 1
            for proc in psutil.process_iter(["pid", "name", "username", "memory_info"]):
                pid = proc.info["pid"]
                cached = self._procs.get(pid, proc)
                try:
                    cpu = cached.cpu_percent(None) / ncpu   # first sample per process reads 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                alive[pid] = cached
                mem = proc.info.get("memory_info")
                rows.append({
                    "pid": pid,
                    "name": proc.info.get("name") or "?",
                    "user": proc.info.get("username") or "?",
                    "cpu": round(cpu, 1),
                    "memory_mb": round(mem.rss / (1024 ** 2), 1) if mem else None,
                })
            self._procs = alive
            rows.sort(key=lambda r: (r["cpu"], r["memory_mb"] or 0), reverse=True)
            self._top, self._top_at = rows[:n], time.time()
            return self._top

    def listening_sockets(self) -> List[Dict[str, Any]]:
        with self._lock:
            if time.time() - self._listen_at < self.SOCK_TTL_SEC:
                return self._listen
            seen, rows = set(), []
            try:
                conns = psutil.net_connections(kind="inet")
            except (psutil.AccessDenied, OSError):
                conns = []
            for c in conns:
                is_tcp = c.type == socket.SOCK_STREAM
                if is_tcp and c.status != psutil.CONN_LISTEN:
                    continue
                if not is_tcp and c.raddr:
                    continue
                if not c.laddr:
                    continue
                key = (c.laddr.ip, c.laddr.port, is_tcp)
                if key in seen:
                    continue
                seen.add(key)
                name = None
                if c.pid:
                    try:
                        name = psutil.Process(c.pid).name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        name = None
                ip = c.laddr.ip
                rows.append({
                    "proto": "tcp" if is_tcp else "udp",
                    "ip": ip,
                    "port": c.laddr.port,
                    "pid": c.pid,
                    "process": name,
                    "exposed": not ip.startswith(self.LOOPBACK),
                })
            rows.sort(key=lambda r: (not r["exposed"], r["port"]))
            self._listen, self._listen_at = rows, time.time()
            return self._listen


# ==================== SENTINEL STATE TRACKER ====================

class SentinelStateTracker:
    """
    Derives the avatar/threat posture from *recent* activity rather than
    lifetime counters: tokens_rejected and sentinel_alerts only ever grow, so a
    single rejection last month must not pin the dashboard in ContainmentAlert.
    """
    ALERT_HOLD_SEC = 30.0       # posture stays alerted this long after a new rejection/alert
    EVAL_HOLD_SEC = 6.0         # posture shows PolicyEvaluating after a newly approved intent
    WINDOW_SEC = 600.0          # "recent" counters cover the last 10 minutes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_rejected: Optional[int] = None
        self._last_alerts: Optional[int] = None
        self._last_log_key: Optional[Tuple[str, str]] = None
        self._last_alert_at = 0.0
        self._last_eval_at = 0.0
        self._rejections: List[Tuple[float, int]] = []
        self._alerts: List[Tuple[float, int]] = []

    def observe(self, snap: "KaiamonSnapshot", now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        sec = snap.system_security
        rejected = int(sec.get("tokens_rejected", 0) or 0)
        alerts = int(snap.containment.get("sentinel_alerts", 0) or 0)
        with self._lock:
            if self._last_rejected is not None and rejected > self._last_rejected:
                self._rejections.append((now, rejected - self._last_rejected))
                self._last_alert_at = now
            if self._last_alerts is not None and alerts > self._last_alerts:
                self._alerts.append((now, alerts - self._last_alerts))
                self._last_alert_at = now
            self._last_rejected, self._last_alerts = rejected, alerts

            # New log lines since the last observation (logs is a rolling buffer).
            logs = list(snap.logs)
            if logs:
                newest = (logs[-1].timestamp, logs[-1].message)
                if self._last_log_key is not None and newest != self._last_log_key:
                    fresh = []
                    for log in reversed(logs):
                        if (log.timestamp, log.message) == self._last_log_key:
                            break
                        fresh.append(log)
                    for log in fresh:
                        msg = log.message.lower()
                        if "denied" in msg or "violation" in msg:
                            self._last_alert_at = now
                        elif "approved" in msg:
                            self._last_eval_at = now
                self._last_log_key = newest

            cutoff = now - self.WINDOW_SEC
            self._rejections = [r for r in self._rejections if r[0] >= cutoff]
            self._alerts = [a for a in self._alerts if a[0] >= cutoff]
            return {
                "alerting": now - self._last_alert_at <= self.ALERT_HOLD_SEC,
                "evaluating": now - self._last_eval_at <= self.EVAL_HOLD_SEC,
                "rejections_10m": sum(n for _, n in self._rejections),
                "alerts_10m": sum(n for _, n in self._alerts),
            }


# ==================== SNAPSHOT SERIALIZER ====================

def serialize_snapshot(
    snap: KaiamonSnapshot,
    telemetry_collector: Optional[TelemetryCollector] = None,
    service_collector: Optional[ServiceCollector] = None,
    ebpf_engine: Optional[EBPFTelemetryEngine] = None,
    proc_sampler: Optional[ProcessSyscallSampler] = None,
    state_tracker: Optional[SentinelStateTracker] = None,
    introspector: Optional[HostIntrospector] = None,
) -> Dict[str, Any]:
    """
    Converts KaiamonSnapshot and host sensors into a JSON-serializable dictionary.
    Pass a long-lived ``state_tracker`` so posture reflects recent activity; without
    one, each call is a fresh baseline (no recent activity).
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Logs formatting
    formatted_logs = [
        {
            "timestamp": log.timestamp,
            "level": log.level,
            "message": log.message,
        }
        for log in snap.logs
    ]

    # 2. Extract active token cards from audit_ledger
    tokens_detail: List[Dict[str, Any]] = []
    seen_sigs: Set[str] = set()
    audit_path = config.AUDIT_LOG_PATH if HAS_CONFIG else os.path.join(BASE_DIR, "storage", "security", "audit_ledger.json")

    if os.path.exists(audit_path):
        try:
            with open(audit_path, "r") as f:
                content = f.read().strip()
            records = []
            if content.startswith("["):
                records = json.loads(content)
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            # Process most recent 50 records in reverse
            for rec in reversed(records[-50:]):
                cap_token_str = rec.get("capability_token")
                if not cap_token_str:
                    req_dict = rec.get("request", {})
                    if isinstance(req_dict, dict):
                        cap_token_str = req_dict.get("capability_token")
                if not cap_token_str:
                    continue

                try:
                    tok_obj = json.loads(cap_token_str) if isinstance(cap_token_str, str) else cap_token_str
                    sig = tok_obj.get("signature", "")
                    if not sig or sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)

                    expires_str = tok_obj.get("expires", "")
                    ttl_seconds = 0
                    is_expired = True

                    if expires_str:
                        exp_iso = expires_str[:-1] + "+00:00" if expires_str.endswith("Z") else expires_str
                        exp_dt = datetime.fromisoformat(exp_iso)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=UTC)
                        diff = (exp_dt - datetime.now(UTC)).total_seconds()
                        if diff > 0:
                            ttl_seconds = int(diff)
                            is_expired = False

                    tokens_detail.append({
                        "id": sig[:8],
                        "signature": sig,
                        "capability": tok_obj.get("capability", "UNKNOWN"),
                        "target": tok_obj.get("target", "*"),
                        "issued_at": tok_obj.get("issued_at", ""),
                        "expires": expires_str,
                        "ttl_seconds": ttl_seconds,
                        "is_expired": is_expired,
                        "disposition": rec.get("result", "observed"),
                    })
                    if len(tokens_detail) >= 8:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # 3. Hardware readings from TelemetryCollector or psutil
    hw_telemetry = {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_pct": psutil.virtual_memory().percent,
        "cpu_temp": 0.0,
        "cpu_power": 0.0,
        "cpu_throttled": False,
        "gpu_temp": 0.0,
        "gpu_power": 0.0,
        "gpu_fan": 0.0,
        "gpu_mem_used_mb": 0.0,
        "split_lock_events": 0,
        "split_lock_last_app": "None",
    }

    if telemetry_collector:
        try:
            t_data = telemetry_collector.get_data()
            hw_telemetry["cpu_temp"] = round(t_data.get("cpu_temp", 0.0), 1)
            hw_telemetry["cpu_power"] = round(t_data.get("cpu_power", 0.0), 1)
            hw_telemetry["cpu_throttled"] = t_data.get("cpu_throttled", False)
            hw_telemetry["gpu_temp"] = round(t_data.get("gpu_temp", 0.0), 1)
            hw_telemetry["gpu_power"] = round(t_data.get("gpu_power", 0.0), 1)
            hw_telemetry["gpu_fan"] = round(t_data.get("gpu_fan", 0.0), 1)
            hw_telemetry["gpu_mem_used_mb"] = round(t_data.get("gpu_mem_used", 0.0), 1)
            hw_telemetry["split_lock_events"] = t_data.get("split_lock_events", 0)
            hw_telemetry["split_lock_last_app"] = t_data.get("split_lock_last_app", "None")
        except Exception:
            pass

    # 4. Service status readings
    services_list = []
    if service_collector:
        try:
            for s in service_collector.get_data():
                services_list.append({
                    "name": s.name.replace(".service", ""),
                    "active": s.active,
                    "substate": s.substate,
                    "pid": s.pid,
                    "cpu": round(s.cpu, 1),
                    "memory_mb": round(s.memory_mb, 1),
                })
        except Exception:
            pass
    if not services_list:
        # Service collector unavailable: report only what the snapshot actually knows.
        gate_running = snap.system_security.get("gate_running", False)
        services_list = [{
            "name": "PolicyGate", "active": gate_running,
            "substate": "running" if gate_running else "inactive",
            "pid": snap.system_security.get("gate_pid", 0), "cpu": None, "memory_mb": None,
        }]

    # 5. eBPF Syscall Waterfall events
    waterfall_events = []
    ebpf_live = bool(ebpf_engine and getattr(ebpf_engine, "bpf", None))
    if ebpf_engine:
        try:
            raw_execs = ebpf_engine.get_recent_execs(30)
            for ev in raw_execs:
                waterfall_events.append({
                    "syscall": "sys_enter_execve",
                    "pid": ev.get("pid", 0),
                    "comm": ev.get("comm", ""),
                    "filename": ev.get("filename", ""),
                    "disposition": "observed",
                    "source": "ebpf",
                    "timestamp": ev.get("timestamp", time.time()),
                })
        except Exception:
            pass

        # openat/openat2 probes existed in the BPF program but their perf buffer
        # was never registered, so file activity was silently absent from the HUD.
        try:
            raw_opens = ebpf_engine.get_recent_opens(30)
            for ev in raw_opens:
                waterfall_events.append({
                    "syscall": ev.get("syscall", "sys_enter_openat"),
                    "pid": ev.get("pid", 0),
                    "comm": ev.get("comm", ""),
                    "filename": ev.get("filename", ""),
                    "disposition": "observed",
                    "source": "ebpf",
                    "timestamp": ev.get("timestamp", time.time()),
                })
        except Exception:
            pass

        # unlinkat is a syscall-entry observation: it records an attempt, not
        # proof that the kernel completed the deletion.
        try:
            raw_deletes = ebpf_engine.get_recent_deletes(30)
            for ev in raw_deletes:
                waterfall_events.append({
                    "syscall": ev.get("syscall", "sys_enter_unlinkat"),
                    "pid": ev.get("pid", 0),
                    "comm": ev.get("comm", ""),
                    "filename": ev.get("filename", ""),
                    "disposition": "observed",
                    "source": "ebpf",
                    "timestamp": ev.get("timestamp", time.time()),
                })
        except Exception:
            pass

    if proc_sampler:
        try:
            sampled = proc_sampler.sample()
            waterfall_events.extend(dict(ev, source="sampler") for ev in sampled)
        except Exception:
            pass

    # Sort and cap waterfall events
    waterfall_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    waterfall_events = waterfall_events[:60]

    # 6. Determine Kaia Sentinel Avatar Reactive State
    sentinel_state = "SentinelWatch"
    threat_level = "LOW"

    lockdown_active = snap.system_security.get("lockdown_active", False)
    gate_running = snap.system_security.get("gate_running", False)
    sentinel_alerts = snap.containment.get("sentinel_alerts", 0)
    tokens_rejected = snap.system_security.get("tokens_rejected", 0)
    lps = snap.lps

    recent = state_tracker.observe(snap) if state_tracker else {
        "alerting": False, "evaluating": False, "rejections_10m": 0, "alerts_10m": 0,
    }

    if lockdown_active:
        sentinel_state = "LockdownMode"
        threat_level = "CRITICAL"
    elif recent["alerting"]:
        sentinel_state = "ContainmentAlert"
        threat_level = "ELEVATED"
    elif not gate_running:
        sentinel_state = "AnomalyDetected"
        threat_level = "ELEVATED"
    elif lps > 30.0:
        sentinel_state = "AuditSurge"
        threat_level = "ELEVATED"
    elif recent["evaluating"]:
        sentinel_state = "PolicyEvaluating"
        threat_level = "LOW"
    elif hw_telemetry["cpu_pct"] < 5.0 and lps < 0.5:
        sentinel_state = "SleepyIdle"
        threat_level = "LOW"
    else:
        sentinel_state = "SentinelWatch"
        threat_level = "LOW"

    # Merge token details into system_security
    sys_sec = dict(snap.system_security)
    sys_sec["token_cards"] = tokens_detail

    return {
        "timestamp_iso": now_iso,
        "timestamp_unix": snap.snapshot_time,
        "lps": round(snap.lps, 2),
        "lps_scale": round(snap.lps_scale, 2),
        "filter_mode": snap.filter_mode,
        "paused": snap.paused,
        "threat_level": threat_level,
        "sentinel_state": sentinel_state,
        "logs": formatted_logs,
        "threat_intel": snap.threat_intel,
        "containment": snap.containment,
        "system_security": sys_sec,
        "host_telemetry": hw_telemetry,
        "services": services_list,
        "ebpf_waterfall": waterfall_events,
        "process_source": "eBPF tracepoints" if ebpf_live else "process sampler (eBPF unavailable)",
        "recent_activity": recent,
        "host": socket.gethostname(),
        "top_processes": introspector.top_processes() if introspector else [],
        "listening": introspector.listening_sockets() if introspector else [],
    }


# ==================== POLICY GATE COMMAND DISPATCHER ====================

def _is_trusted_policy_gate_fallback_socket(sock_path: str) -> bool:
    """Accept a fallback only inside a current-user-owned private directory."""
    try:
        parent = os.path.dirname(os.path.abspath(sock_path))
        directory_stat = os.lstat(parent)
        socket_stat = os.lstat(sock_path)
    except OSError:
        return False

    directory_mode = stat.S_IMODE(directory_stat.st_mode)
    return (
        stat.S_ISDIR(directory_stat.st_mode)
        and directory_stat.st_uid == os.geteuid()
        and not (directory_mode & 0o077)
        and (directory_mode & 0o300) == 0o300
        and stat.S_ISSOCK(socket_stat.st_mode)
        and socket_stat.st_uid == os.geteuid()
        and socket_stat.st_nlink == 1
    )


def send_policy_gate_ipc(action: str, payload: Dict[str, Any], cap_token: Optional[str] = None) -> Tuple[bool, Any]:
    """
    Length-framed IPC client communicating with Policy Gate over Unix Domain Socket.
    The production socket is preferred; a fallback is accepted only in a private
    per-user directory and only when it is a socket owned by the current user.
    """
    req_id = str(uuid.uuid4())
    nested = {
        "request_id": req_id,
        "action": action,
        "payload": payload,
        "capability_token": cap_token,
    }

    fallback_default = os.path.join(
        "/tmp", f"kaia-policy-gate-{os.getuid()}", "policy_gate.sock"
    )
    if HAS_CONFIG:
        sockets_to_try = [config.POLICY_GATE_SOCKET, config.POLICY_GATE_SOCKET_FALLBACK]
    else:
        sockets_to_try = ["/run/kaiacord/policy_gate.sock", fallback_default]

    last_error = "Policy Gate socket unavailable"
    for index, sock_path in enumerate(sockets_to_try):
        if index > 0 and not _is_trusted_policy_gate_fallback_socket(sock_path):
            last_error = "Untrusted or unavailable fallback socket path"
            continue
        if not os.path.exists(sock_path):
            continue
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.5)
            s.connect(sock_path)

            payload_bytes = json.dumps(nested).encode("utf-8")
            header = len(payload_bytes).to_bytes(4, byteorder="big")
            s.sendall(header + payload_bytes)

            header_resp = bytearray()
            while len(header_resp) < 4:
                p = s.recv(4 - len(header_resp))
                if not p:
                    raise RuntimeError("IPC connection closed before response header")
                header_resp.extend(p)
            length = int.from_bytes(header_resp, byteorder="big")

            payload_resp = bytearray()
            while len(payload_resp) < length:
                p = s.recv(length - len(payload_resp))
                if not p:
                    raise RuntimeError("IPC connection closed before response payload")
                payload_resp.extend(p)
            s.close()

            resp = json.loads(payload_resp.decode("utf-8"))
            approved = resp.get("approved", False)
            exec_res = resp.get("executor_response", {})
            return approved, exec_res
        except Exception as e:
            last_error = str(e)
            continue

    return False, {"message": f"Policy Gate offline or unreachable: {last_error}"}


def dispatch_operator_command(cmd_line: str) -> Dict[str, Any]:
    """
    Executes an operator command via schema validation and Policy Gate dispatch.
    Returns structured results for UI terminal display.
    """
    raw_cmd = cmd_line.strip()
    if not raw_cmd:
        return {"success": False, "command": raw_cmd, "lines": ["Empty command."]}

    parts = raw_cmd.split()
    cmd = parts[0].lower()
    args = parts[1:]
    output_lines: List[str] = []

    try:
        if cmd == "help":
            output_lines = [
                "KAIA CYBER-SENTINEL COMMAND INTERFACE",
                "=====================================",
                "  block <ip>                    - Block threat IP via nftables & Policy Gate",
                "  unblock <ip>                  - Remove IP from firewall block table",
                "  show rules                    - Inspect active nftables DROP rules",
                "  show audit [--since <Nh>]     - Query recent security audit records",
                "  show assets                   - List passive LAN discovered entities",
                "  show fim alerts               - View file integrity monitoring events",
                "  check threat <ip>             - Enrich IP via Shodan InternetDB & CVEs",
                "  restart <service>             - Issue signed restart for allowed service",
                "  add rule <name> <ioc> [mitre] - Compile and register YARA rule",
                "  list recent blocks            - View recent 10 firewall mitigations",
                "  lockdown                      - Initiate emergency network containment",
                "  status                        - Overview of security daemons and sockets",
                "  clear                         - Clear command console buffer",
            ]
            return {"success": True, "command": raw_cmd, "lines": output_lines}

        elif cmd == "status":
            success, stdout = query_gate_diagnostics("nft_list")
            gate_status = "ONLINE / ACTIVE" if success else "OFFLINE"
            drop_count = 0
            if success:
                drop_count = sum(1 for line in stdout.splitlines() if "drop" in line.lower() or "reject" in line.lower())
            output_lines.append(f"Policy Gate Status: {gate_status}")
            output_lines.append(f"Active Firewall Drops: {drop_count} detected")
            output_lines.append(f"Audit Ledger: {config.AUDIT_LOG_PATH if HAS_CONFIG else 'storage/security/audit_ledger.json'}")
            return {"success": True, "command": raw_cmd, "lines": output_lines}

        elif cmd == "block" and len(args) == 1:
            ip = args[0]
            token = generate_capability_token("block_ip", ip, duration_seconds=15)
            payload = {
                "target_ip": ip,
                "protocol": "all",
                "port": None,
                "justification": "operator requested firewall block",
                "session_id": "sentinel_ui",
            }
            approved, resp = send_policy_gate_ipc("block_ip", payload, token)
            if approved:
                output_lines.append(f"[APPROVED] Threat IP {ip} successfully blocked.")
                detail = resp.get("stdout", "") if isinstance(resp, dict) else ""
                if detail:
                    output_lines.append(detail)
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            msg = resp.get("message") if isinstance(resp, dict) else str(resp)
            output_lines.append(f"[DENIED] block_ip rejected: {msg}")
            return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "block":
            output_lines.append("Usage: block <ip>")
            return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "unblock" and len(args) == 1:
            ip = args[0]
            token = generate_capability_token("unblock_ip", ip, duration_seconds=15)
            payload = {
                "target_ip": ip,
                "protocol": "all",
                "port": None,
                "justification": "operator requested firewall rule removal",
                "session_id": "sentinel_ui",
            }
            approved, resp = send_policy_gate_ipc("unblock_ip", payload, token)
            if approved:
                detail = resp.get("stdout", "") if isinstance(resp, dict) else ""
                output_lines.append(f"[APPROVED] Firewall block for {ip} removed or already absent.")
                if detail:
                    output_lines.append(detail)
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            msg = resp.get("message") if isinstance(resp, dict) else str(resp)
            output_lines.append(f"[DENIED] unblock_ip rejected: {msg}")
            return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "unblock":
            output_lines.append("Usage: unblock <ip>")
            return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "show" and len(args) >= 1 and args[0] == "rules":
            approved, resp = query_gate_diagnostics("nft_list")
            if approved:
                lines = resp.splitlines()
                output_lines.extend(lines[:35])
                if len(lines) > 35:
                    output_lines.append(f"... ({len(lines) - 35} lines truncated)")
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            else:
                output_lines.append(f"[ERROR] Failed to query rules: {resp}")
                return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "show" and len(args) >= 1 and args[0] == "assets":
            try:
                from security.network_discovery import PassiveDiscoveryEngine
                discovery = PassiveDiscoveryEngine.get_instance()
                assets = discovery.get_recent_assets(20)
                if assets:
                    output_lines.append(f"{'IP':<16} {'MAC':<18} {'VENDOR':<16} {'VECTOR':<12} {'LAST SEEN'}")
                    output_lines.append("-" * 75)
                    for a in assets:
                        output_lines.append(f"{a['ip']:<16} {a['mac']:<18} {a.get('vendor', 'Unknown')[:15]:<16} {a.get('detection_vector', 'arp'):<12} {a.get('last_seen', '')}")
                else:
                    output_lines.append("No LAN assets discovered yet.")
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            except Exception as e:
                output_lines.append(f"[ERROR] Discovery query failed: {e}")
                return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "show" and len(args) >= 2 and args[0] == "fim" and args[1] == "alerts":
            fim_db_path = config.FIM_DB_PATH
            if os.path.exists(fim_db_path):
                import sqlite3
                conn = sqlite3.connect(fim_db_path, timeout=1.0)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, event_type, pid, comm, path, yara_matches
                    FROM fim_events ORDER BY rowid DESC LIMIT 15
                """)
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    for r in rows:
                        lbl = f"[{r[0][:19]}] {r[1].upper()} | {os.path.basename(r[4])} | {r[3]} (PID {r[2]})"
                        if r[5]:
                            lbl += f" [YARA: {r[5]}]"
                        output_lines.append(lbl)
                else:
                    output_lines.append("No FIM integrity alerts recorded.")
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            else:
                output_lines.append(f"FIM audit database ({fim_db_path}) not found.")
                return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "check" and len(args) >= 2 and args[0] == "threat":
            ip = args[1]
            from security import threat_intel
            rep = threat_intel.get_ip_reputation(ip)
            shodan = threat_intel.lookup_internetdb(ip)
            output_lines.append(f"Threat Intelligence Report for {ip}:")
            output_lines.append(f"  • IP Reputation Score: {rep.get('score', 0)}/100 ({rep.get('threat_level', 'unknown')})")
            if shodan.get("ports"):
                output_lines.append(f"  • Open Ports: {shodan.get('ports')}")
            if shodan.get("tags"):
                output_lines.append(f"  • Shodan Tags: {shodan.get('tags')}")
            if shodan.get("vulns"):
                output_lines.append(f"  • Vulnerabilities: {len(shodan['vulns'])} CVEs identified")
                for cve in shodan["vulns"][:3]:
                    details = threat_intel.lookup_cve_details(cve)
                    output_lines.append(f"    - {cve} (CVSS {details.get('cvss', 'N/A')}): {str(details.get('details', ''))[:60]}")
            return {"success": True, "command": raw_cmd, "lines": output_lines}

        elif cmd == "restart" and len(args) >= 1:
            srv = args[0]
            token = generate_capability_token("restart_service", srv, duration_seconds=15)
            payload = {"service_name": srv, "justification": "sentinel_ui request", "session_id": "sentinel_ui"}
            approved, resp = send_policy_gate_ipc("restart_service", payload, token)
            if approved:
                output_lines.append(f"[APPROVED] Service '{srv}' restart triggered.")
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            else:
                msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                output_lines.append(f"[DENIED] Restart rejected: {msg}")
                return {"success": False, "command": raw_cmd, "lines": output_lines}

        elif cmd == "lockdown":
            token = generate_capability_token("diagnostics", "lockdown_confirm", duration_seconds=15)
            payload = {"query_type": "lockdown_trigger", "args": [], "justification": "operator emergency", "session_id": "sentinel_ui"}
            approved, resp = send_policy_gate_ipc("lockdown", payload, token)
            if approved:
                output_lines.append("[CRITICAL] Emergency Network Lockdown Triggered.")
                return {"success": True, "command": raw_cmd, "lines": output_lines}
            else:
                msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                output_lines.append(f"[DENIED] Lockdown failed: {msg}")
                return {"success": False, "command": raw_cmd, "lines": output_lines}

        else:
            output_lines.append(f"Unknown command: '{raw_cmd}'. Type 'help' for available commands.")
            return {"success": False, "command": raw_cmd, "lines": output_lines}

    except Exception as e:
        logger.exception("Error executing command: %s", raw_cmd)
        return {"success": False, "command": raw_cmd, "lines": [f"Execution error: {e}"]}


# ==================== AIOHTTP WEB BRIDGE SERVER ====================

class KaiaWebBridge:
    """
    Asynchronous web daemon hosting the Cyber-Sentinel UI and streaming live telemetry.
    """
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        collector_mgr: Optional[CollectorManager] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        service_collector: Optional[ServiceCollector] = None,
    ) -> None:
        # Enforce Zero-Trust Invariant 1: Loopback only
        if host not in ALLOWED_LOOPBACK_HOSTS:
            raise ValueError(
                f"Security Invariant Violation (INV-Loopback): Kaia Web Bridge must only bind to "
                f"loopback ({', '.join(ALLOWED_LOOPBACK_HOSTS)}). Attempted to bind to '{host}'."
            )

        self.host = host
        self.port = port
        self.owns_collectors = False

        if collector_mgr is not None:
            self.collector_mgr = collector_mgr
        else:
            self.collector_mgr = CollectorManager()
            self.owns_collectors = True

        self._stop_event = threading.Event()
        self.telemetry_collector = telemetry_collector or TelemetryCollector(self._stop_event)
        self.service_collector = service_collector or ServiceCollector(self._stop_event)
        self.ebpf_engine = EBPFTelemetryEngine.get_instance()
        self.proc_sampler = ProcessSyscallSampler()
        self.state_tracker = SentinelStateTracker()
        self.introspector = HostIntrospector()

        self.ws_clients: Set[web.WebSocketResponse] = set()
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._broadcast_task: Optional[asyncio.Task] = None

    @web.middleware
    async def security_headers_middleware(self, request: web.Request, handler) -> web.StreamResponse:
        """Enforce loopback authority/origin checks and security headers on all routes."""
        # DNS rebinding is not limited to state-changing routes: same-origin script
        # on an attacker-controlled hostname could read the host telemetry from
        # /api/snapshot. Apply the same authority check at the application boundary
        # so health, snapshot, index, static assets, commands, and WebSockets agree.
        request_error = self._local_request_error(request)
        if request_error:
            resp = web.json_response(
                {"success": False, "error": request_error}, status=403
            )
        else:
            resp = await handler(request)
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
            "font-src 'self';"
        )
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        resp.headers["Server"] = "Kaia-Sentinel-Bridge/1.0"
        return resp

    def create_app(self) -> web.Application:
        """Constructs the configured aiohttp web application."""
        app = web.Application(middlewares=[self.security_headers_middleware])

        # Routes
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/api/health", self.handle_health)
        app.router.add_get("/api/snapshot", self.handle_snapshot)
        app.router.add_post("/api/command", self.handle_command)
        app.router.add_get("/ws", self.handle_websocket)
        app.router.add_get("/ws/telemetry", self.handle_websocket)

        # Static assets
        if os.path.exists(STATIC_DIR):
            app.router.add_static("/static", STATIC_DIR, show_index=False)

        return app

    def _local_request_error(self, request: web.Request) -> Optional[str]:
        """Reject non-loopback or authority/origin mismatches before dispatch.

        Do not assume an omitted port means the HUD's port: URL defaults are 80
        for HTTP and 443 for HTTPS. The Host authority must also name the actual
        local listener, preventing a forged Host and matching Origin from being
        accepted on a non-default port.
        """
        from urllib.parse import urlsplit

        host_header = request.headers.get("Host", "")
        if not host_header or host_header != host_header.strip():
            return "Invalid Host header"
        try:
            parsed_host = urlsplit("//" + host_header)
            host = (parsed_host.hostname or "").lower()
            explicit_host_port = parsed_host.port
        except ValueError:
            return "Invalid Host header"

        if (
            not host
            or host not in ALLOWED_LOOPBACK_HOSTS
            or parsed_host.username is not None
            or parsed_host.password is not None
            or parsed_host.path
            or parsed_host.query
            or parsed_host.fragment
            or "," in host_header
        ):
            return "Invalid Host header"

        scheme = request.scheme.lower()
        if scheme not in ("http", "https"):
            return "Invalid request scheme"
        default_port = 443 if scheme == "https" else 80
        host_effective_port = explicit_host_port if explicit_host_port is not None else default_port

        # Prefer the actual listener socket over self.port: test servers and
        # socket activation may select a port different from the configured one.
        local_port = None
        transport = request.transport
        if transport is not None:
            sockname = transport.get_extra_info("sockname")
            if isinstance(sockname, tuple) and len(sockname) >= 2:
                candidate = sockname[1]
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    local_port = candidate
        if local_port is None:
            local_port = self.port
        if host_effective_port != local_port:
            return "Host port mismatch"

        origin = request.headers.get("Origin")
        if origin:
            if origin != origin.strip():
                return "Invalid Origin"
            try:
                parsed_origin = urlsplit(origin)
                explicit_origin_port = parsed_origin.port
            except ValueError:
                return "Invalid Origin"

            origin_scheme = parsed_origin.scheme.lower()
            origin_host = (parsed_origin.hostname or "").lower()
            if (
                origin_scheme not in ("http", "https")
                or origin_host not in ALLOWED_LOOPBACK_HOSTS
                or parsed_origin.username is not None
                or parsed_origin.password is not None
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
                or not parsed_origin.netloc
            ):
                return "Invalid Origin"

            origin_default_port = 443 if origin_scheme == "https" else 80
            origin_effective_port = (
                explicit_origin_port if explicit_origin_port is not None else origin_default_port
            )
            if origin_scheme != scheme:
                return "Origin scheme mismatch"
            if origin_effective_port != host_effective_port:
                return "Origin port mismatch"
            if origin_host != host:
                return "Origin host mismatch"
        return None

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serves index.html with Content-Type text/html."""
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index_path):
            return web.Response(text="<h1>Kaia Cyber-Sentinel: Static assets initializing...</h1>", content_type="text/html")
        with open(index_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "service": "Kaia Cyber-Sentinel Operations Center",
            "version": "1.0.0",
            "bind": f"{self.host}:{self.port}",
            "loopback_isolated": True,
            "connected_ws_clients": len(self.ws_clients),
        })

    async def handle_snapshot(self, request: web.Request) -> web.Response:
        """Returns the current serialized snapshot."""
        loop = asyncio.get_running_loop()
        snap = await loop.run_in_executor(None, self.collector_mgr.take_snapshot)
        data = serialize_snapshot(
            snap,
            self.telemetry_collector,
            self.service_collector,
            self.ebpf_engine,
            self.proc_sampler,
            self.state_tracker,
            self.introspector,
        )
        return web.json_response(data)

    async def handle_command(self, request: web.Request) -> web.Response:
        """Dispatches an interactive operator command."""
        # Reject invalid Host/Origin before parsing an attacker-controlled body or
        # dispatching an action through the privileged Policy Gate.
        request_error = self._local_request_error(request)
        if request_error:
            return web.json_response({"success": False, "error": request_error}, status=403)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON body"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"success": False, "error": "JSON body must be an object"}, status=400)
        cmd_line = body.get("command", "")
        if not isinstance(cmd_line, str):
            return web.json_response({"success": False, "error": "Command must be string"}, status=400)
        if len(cmd_line) > 2048:
            return web.json_response({"success": False, "error": "Command exceeds 2048 characters"}, status=413)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, dispatch_operator_command, cmd_line)
        return web.json_response(result)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Read-only outbound telemetry WebSocket broadcast."""
        # WebSockets are not protected by the browser's same-origin policy.
        # Without this check, any visited site can open a socket to loopback
        # and read process, listener, and host telemetry from the local daemon.
        request_error = self._local_request_error(request)
        if request_error:
            raise web.HTTPForbidden(text=request_error)
        ws = web.WebSocketResponse(heartbeat=15.0)
        await ws.prepare(request)
        self.ws_clients.add(ws)
        logger.info("HUD client connected from %s (Total: %d)", request.remote, len(self.ws_clients))

        try:
            # Send initial snapshot immediately
            loop = asyncio.get_running_loop()
            snap = await loop.run_in_executor(None, self.collector_mgr.take_snapshot)
            init_data = serialize_snapshot(
                snap,
                self.telemetry_collector,
                self.service_collector,
                self.ebpf_engine,
                self.proc_sampler,
                self.state_tracker,
                self.introspector,
            )
            await ws.send_json(init_data)

            # Inbound loop: discard arbitrary commands from WS (commands go through /api/command)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Echo ping/pong or ignore
                    if msg.data == "ping":
                        await ws.send_str("pong")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            self.ws_clients.discard(ws)
            logger.info("HUD client disconnected (Total: %d)", len(self.ws_clients))

        return ws

    async def _broadcast_loop(self) -> None:
        """Periodic broadcast loop pushing snapshots to all active WebSockets."""
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            if self.ws_clients:
                try:
                    snap = await loop.run_in_executor(None, self.collector_mgr.take_snapshot)
                    data = serialize_snapshot(
                        snap,
                        self.telemetry_collector,
                        self.service_collector,
                        self.ebpf_engine,
                        self.proc_sampler,
                        self.state_tracker,
                        self.introspector,
                    )
                    payload_str = json.dumps(data)

                    # Broadcast to all clients
                    dead_clients = set()
                    for ws in list(self.ws_clients):
                        try:
                            if not ws.closed:
                                await ws.send_str(payload_str)
                            else:
                                dead_clients.add(ws)
                        except Exception:
                            dead_clients.add(ws)

                    for ws in dead_clients:
                        self.ws_clients.discard(ws)
                except Exception as e:
                    logger.debug("Error during telemetry broadcast: %s", e)

            await asyncio.sleep(TELEMETRY_BROADCAST_INTERVAL_SEC)

    def start_collectors(self) -> None:
        """Launches collectors if owned."""
        if self.owns_collectors:
            self.collector_mgr.start_all()
        try:
            self.telemetry_collector.start()
        except Exception:
            pass
        try:
            self.service_collector.start()
        except Exception:
            pass
        try:
            self.ebpf_engine.start()
        except Exception:
            pass

    def stop_collectors(self) -> None:
        """Signals stop event and cleans up collectors."""
        self._stop_event.set()
        if self.owns_collectors:
            self.collector_mgr.stop_all()

    async def start_server_async(self) -> None:
        """Asynchronously boots the aiohttp server and background task."""
        self.start_collectors()
        self.app = self.create_app()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("Kaia Cyber-Sentinel Web Bridge listening at http://%s:%d", self.host, self.port)

    async def stop_server_async(self) -> None:
        """Gracefully halts the server and cleans up."""
        self.stop_collectors()
        if self._broadcast_task:
            self._broadcast_task.cancel()
        for ws in list(self.ws_clients):
            try:
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"Server shutdown")
            except Exception:
                pass
        self.ws_clients.clear()
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    def run_sync(self) -> None:
        """Runs the server in a blocking asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.start_server_async())
            print(f"\n=======================================================")
            print(f" KAIA CYBER-SENTINEL OPERATIONS CENTER ACTIVE")
            print(f" HUD URL: http://{self.host}:{self.port}")
            print(f" Zero-Trust Isolation: Loopback {self.host} Enforced")
            print(f" Press Ctrl+C to terminate Sentinel Operations")
            print(f"=======================================================\n")
            loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down Kaia Cyber-Sentinel Web Bridge...")
        finally:
            loop.run_until_complete(self.stop_server_async())
            loop.close()


# ==================== CLI RUNNER ====================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Kaia Cyber-Sentinel Web Bridge")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (Must be 127.0.0.1 or localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8765)")
    args = parser.parse_args()

    bridge = KaiaWebBridge(host=args.host, port=args.port)
    bridge.run_sync()


if __name__ == "__main__":
    main()
