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

import hmac
import os
import re
import secrets
import sqlite3
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
    ledger_token,
    query_gate_diagnostics,
    read_ledger_tail,
    token_expiry,
)
from security.policy_gate import generate_capability_token


# ==================== CONSTANTS & INVARIANTS ====================

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TELEMETRY_BROADCAST_INTERVAL_SEC = 0.20  # 5 updates per second
WS_SEND_TIMEOUT_SEC = 2.0  # a client that cannot take a frame in this time is dropped
MAX_POLICY_GATE_RESPONSE_BYTES = 4 * 1024 * 1024
ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_WS_CLIENTS = 16  # the HUD is one window; this only bounds a local flood

# /api/command mints capability tokens for whatever it is asked (block, restart,
# lockdown). Host and Origin checks stop a web page, but not another local
# process: any program of any user on this machine could POST to the loopback
# port. Commands therefore need the per-start session key, which only the
# launcher (through the environment) and the HUD page it opens ever hold.
SESSION_KEY_ENV = "KAIA_HUD_SESSION_KEY"
SESSION_HEADER = "X-Kaia-Session"
SESSION_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,128}")


def resolve_session_key(environ=os.environ) -> str:
    """The launcher's key when it is well formed, else a fresh random one."""
    supplied = environ.get(SESSION_KEY_ENV, "")
    if SESSION_KEY_PATTERN.fullmatch(supplied):
        return supplied
    return secrets.token_urlsafe(32)

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
    ebpf_engine: Optional[Any] = None,  # the engine API: recent execs, opens, deletes
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
            # Structured security-event fields (empty for plain journal lines).
            "iso": getattr(log, "iso", ""),
            "event": getattr(log, "event", ""),
            "status": getattr(log, "status", ""),
            "alert_level": getattr(log, "alert_level", ""),
            "reason": getattr(log, "reason", ""),
            "detail": getattr(log, "detail", ""),
        }
        for log in snap.logs
    ]

    # 2. Token cards from the newest ledger records (a bounded tail, not the file).
    tokens_detail: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    audit_path = config.AUDIT_LOG_PATH if HAS_CONFIG else os.path.join(BASE_DIR, "storage", "security", "audit_ledger.json")
    now_utc = datetime.now(UTC)
    for rec in reversed(read_ledger_tail(audit_path, max_records=50)):
        token = ledger_token(rec)
        if token is None or token["id"] in seen_ids:
            continue
        seen_ids.add(token["id"])
        exp_dt = token_expiry(token)
        remaining = (exp_dt - now_utc).total_seconds() if exp_dt else 0
        tokens_detail.append({
            "id": token["id"][:8],
            "fingerprint": token["id"],
            "capability": token.get("capability", "UNKNOWN"),
            "target": token.get("target", "*"),
            "issued_at": token.get("issued_at", ""),
            "expires": token.get("expires", ""),
            "ttl_seconds": int(remaining) if remaining > 0 else 0,
            "is_expired": remaining <= 0,
            "disposition": rec.get("result", "observed"),
        })
        if len(tokens_detail) >= 8:
            break

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

    # A control that is meant to run and does not (or a gate that stopped
    # publishing) is an anomaly. "degraded" and "off" are standing conditions of
    # this host and do not move the posture; the HUD's control strip shows them.
    health = snap.system_security.get("control_health") or {}
    controls_failed = health.get("overall") == "failed"

    if lockdown_active:
        sentinel_state = "LockdownMode"
        threat_level = "CRITICAL"
    elif recent["alerting"]:
        sentinel_state = "ContainmentAlert"
        threat_level = "ELEVATED"
    elif not gate_running or controls_failed:
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
    """Send a bounded, length-framed request to the trusted Policy Gate socket."""
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
            # Do not overwrite a specific primary-socket protocol failure with
            # the expected fact that the fallback socket was unavailable.
            if last_error == "Policy Gate socket unavailable":
                last_error = "Untrusted or unavailable fallback socket path"
            continue
        if not os.path.exists(sock_path):
            continue
        s = None
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.5)
            s.connect(sock_path)

            payload_bytes = json.dumps(nested).encode("utf-8")
            header = len(payload_bytes).to_bytes(4, byteorder="big")
            s.sendall(header + payload_bytes)

            header_resp = bytearray()
            while len(header_resp) < 4:
                packet = s.recv(4 - len(header_resp))
                if not packet:
                    raise RuntimeError("IPC connection closed before response header")
                header_resp.extend(packet)
            length = int.from_bytes(header_resp, byteorder="big")
            if length <= 0 or length > MAX_POLICY_GATE_RESPONSE_BYTES:
                raise ValueError(
                    f"Invalid IPC response frame length {length}; maximum is "
                    f"{MAX_POLICY_GATE_RESPONSE_BYTES} bytes"
                )

            payload_resp = bytearray()
            while len(payload_resp) < length:
                packet = s.recv(min(65536, length - len(payload_resp)))
                if not packet:
                    raise RuntimeError("IPC connection closed before response payload")
                payload_resp.extend(packet)

            response = json.loads(payload_resp.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("IPC response must be a JSON object")
            approved = response.get("approved", False)
            exec_res = response.get("executor_response", {})
            return approved, exec_res
        except Exception as exc:
            last_error = str(exc)
            continue
        finally:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    return False, {"message": f"Policy Gate offline or unreachable: {last_error}"}

def show_audit_lines(hours: int, limit: int = 20) -> List[str]:
    """Newest security events from the last ``hours``, with level and reason. Read-only."""
    from security import alert_levels
    db_path = config.SECURITY_DB_PATH if HAS_CONFIG else os.path.join(BASE_DIR, "storage", "security", "security_events.db")
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = conn.execute(
                "SELECT timestamp, type, actor, disposition FROM security_events "
                "WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?", (since, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        return [f"[ERROR] Cannot read the security event database: {e}"]
    if not rows:
        return [f"No security events in the last {hours} h."]
    lines = [f"Security events, last {hours} h (newest {len(rows)}):"]
    for ts, event_type, actor, disposition in rows:
        level, reason = alert_levels.level_of(event_type)
        lines.append(f"  {str(ts)[:19]}  {level.upper():<8} {event_type}  {str(actor)[:60]}  [{disposition}]  {reason}")
    return lines


def active_block_lines(limit: int = 10) -> List[str]:
    """Kaia's active blocks, folded from the ledger exactly as boot replay does."""
    import importlib.util
    path = os.path.join(BASE_DIR, "scripts", "kaia_restore_blocks.py")
    try:
        spec = importlib.util.spec_from_file_location("kaia_restore_blocks", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        blocks = module.collect_blocks(config.AUDIT_LOG_PATH)
    except Exception as e:
        return [f"[ERROR] Cannot read the audit ledger: {e}"]
    if not blocks:
        return ["No active Kaia blocks recorded in the audit ledger."]
    lines = [f"Active Kaia blocks (newest {min(limit, len(blocks))} of {len(blocks)}):"]
    for b in list(reversed(blocks))[:limit]:
        scope = "all traffic" if b["protocol"] == "all" else f"{b['protocol']}/{b['port']}"
        lines.append(f"  {b['target_ip']:<40} {scope}")
    return lines


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

        elif cmd == "show" and len(args) >= 1 and args[0] == "audit":
            hours = 24
            if len(args) == 3 and args[1] == "--since" and re.fullmatch(r"[0-9]{1,3}h", args[2]):
                hours = int(args[2][:-1])
            elif len(args) != 1:
                return {"success": False, "command": raw_cmd, "lines": ["Usage: show audit [--since <N>h]  (N up to 168)"]}
            hours = max(1, min(hours, 168))
            return {"success": True, "command": raw_cmd, "lines": show_audit_lines(hours)}

        elif cmd == "list" and args == ["recent", "blocks"]:
            return {"success": True, "command": raw_cmd, "lines": active_block_lines()}

        elif cmd == "add" and len(args) >= 3 and args[0] == "rule":
            name, ioc = args[1], args[2]
            mitre = args[3] if len(args) > 3 else None
            if len(args) > 4:
                return {"success": False, "command": raw_cmd, "lines": ["Usage: add rule <name> <ioc> [mitre]"]}
            # The gate verifies this exact scope (write_file on rules/<name>.yar)
            # and validates every field itself; the console only proposes.
            token = generate_capability_token("write_file", f"rules/{name}.yar", duration_seconds=15)
            payload = {
                "rule_name": name,
                "target_ioc_indicator": ioc,
                "threat_description": f"Operator-added IOC rule {name}",
                "author": "operator (HUD console)",
                "justification": "operator requested YARA rule",
                "session_id": "sentinel_ui",
            }
            if mitre:
                payload["mitre_framework_id"] = mitre
            approved, resp = send_policy_gate_ipc("add_rule", payload, token)
            msg = resp.get("message") if isinstance(resp, dict) else str(resp)
            if approved:
                return {"success": True, "command": raw_cmd, "lines": [f"[APPROVED] {msg}"]}
            return {"success": False, "command": raw_cmd, "lines": [f"[DENIED] add rule rejected: {msg}"]}

        elif cmd in ("add", "list"):
            return {"success": False, "command": raw_cmd,
                    "lines": ["Usage: add rule <name> <ioc> [mitre]  |  list recent blocks"]}

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
            import ipaddress
            try:
                ip = ipaddress.ip_address(args[1]).compressed
            except ValueError:
                return {"success": False, "command": raw_cmd, "lines": ["Usage: check threat <ip>  (not an IP address)"]}
            from security import threat_intel
            rep = threat_intel.get_ip_reputation(ip)
            shodan = threat_intel.lookup_internetdb(ip)
            output_lines.append(f"Threat Intelligence Report for {ip}:")
            # An address nobody has rated is not a clean one: the default record
            # (score 100, "unrated") used to print as "100/100".
            if rep.get("cached"):
                output_lines.append(f"  • IP Reputation Score: {rep.get('score', 0)}/100 "
                                    f"({', '.join(str(t) for t in rep.get('tags', [])) or 'no tags'})")
            else:
                output_lines.append("  • IP Reputation: not rated (no local reputation record)")
            if shodan.get("source") in ("non-global", "invalid"):
                output_lines.append("  • InternetDB: not queried (private or non-global address)")
            elif not shodan.get("found"):
                output_lines.append("  • InternetDB: no record")
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
        session_key: Optional[str] = None,
    ) -> None:
        # Enforce Zero-Trust Invariant 1: Loopback only
        if host not in ALLOWED_LOOPBACK_HOSTS:
            raise ValueError(
                f"Security Invariant Violation (INV-Loopback): Kaia Web Bridge must only bind to "
                f"loopback ({', '.join(ALLOWED_LOOPBACK_HOSTS)}). Attempted to bind to '{host}'."
            )

        self.host = host
        self.port = port
        self.session_key = session_key if session_key is not None else resolve_session_key()
        if not SESSION_KEY_PATTERN.fullmatch(self.session_key):
            raise ValueError("HUD session key must be 32-128 URL-safe characters")
        self.owns_collectors = False

        if collector_mgr is not None:
            self.collector_mgr = collector_mgr
        else:
            self.collector_mgr = CollectorManager()
            self.owns_collectors = True

        self._stop_event = threading.Event()
        self.telemetry_collector = telemetry_collector or TelemetryCollector(self._stop_event)
        self.service_collector = service_collector or ServiceCollector(self._stop_event)
        # The gate is the only process that can load eBPF; read what it publishes
        # instead of compiling (and failing) an unprivileged engine here.
        from security.telemetry_publish import PublishedTelemetry, FILE_NAME as TELEMETRY_FILE
        state_dir = str(config.STATE_DIR) if HAS_CONFIG else os.path.join(BASE_DIR, "storage", "security")
        self.ebpf_engine = PublishedTelemetry(os.path.join(state_dir, TELEMETRY_FILE))
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

        # Before the body is parsed: an unauthenticated local process gets nothing.
        supplied = request.headers.get(SESSION_HEADER, "")
        if not hmac.compare_digest(supplied.encode("utf-8", "replace"), self.session_key.encode("utf-8")):
            return web.json_response(
                {"success": False, "error": "Missing or wrong HUD session key",
                 "lines": ["[DENIED] This console is not authorised: open the HUD with ./kaia-hud, "
                           "or use the URL the bridge printed at start."]},
                status=403,
            )

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
        if len(self.ws_clients) >= MAX_WS_CLIENTS:
            raise web.HTTPServiceUnavailable(text="Too many HUD clients")
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

    async def _broadcast(self, payload_str: str) -> None:
        """Send to every client concurrently; drop one that cannot keep up.

        Sends were sequential, and send_str waits for the socket to drain, so
        one client that stopped reading (a frozen tab, a stalled connection)
        slowed the stream for every other client to ~0.5 updates/s (measured).
        A client that does not accept a frame within WS_SEND_TIMEOUT_SEC is
        closed and removed; the browser reconnects.
        """
        clients = list(self.ws_clients)

        async def send(ws):
            if ws.closed:
                raise ConnectionError("closed")
            await asyncio.wait_for(ws.send_str(payload_str), WS_SEND_TIMEOUT_SEC)

        results = await asyncio.gather(*(send(ws) for ws in clients), return_exceptions=True)
        for ws, result in zip(clients, results):
            if isinstance(result, BaseException):
                self.ws_clients.discard(ws)
                if isinstance(result, asyncio.TimeoutError):
                    logger.warning("Dropping a HUD client that stopped reading (send timed out)")
                # Closing a stalled socket can block too; never wait on it here.
                asyncio.ensure_future(self._close_quietly(ws))

    @staticmethod
    async def _close_quietly(ws) -> None:
        try:
            await asyncio.wait_for(ws.close(), WS_SEND_TIMEOUT_SEC)
        except Exception:
            pass

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

                    await self._broadcast(payload_str)
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
            # The fragment never leaves the browser; the page needs it for commands.
            print(f" HUD URL: http://{self.host}:{self.port}/#session={self.session_key}")
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
