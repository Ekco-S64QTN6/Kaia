#!/usr/bin/env python3
"""
Kaia Dashboard: System & Security Monitoring TUI
=================================================

curses dashboard for the Kaia hardened AI admin agent.
snapshot-based rendering architecture with strict thread-ownership boundaries.
"""

import curses
import time
import sys
import os
import threading
import subprocess
import re
import signal
import atexit
import select
import socket
import glob
import sqlite3
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
from datetime import datetime, timedelta
import queue
import logging

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from core import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Optional GPU library
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)



# ==================== CONSTANTS ====================

# Timing intervals (seconds)
FRAME_INTERVAL_MS: int = 100          # ~10 FPS via stdscr.timeout()
PING_INTERVAL: float = 2.0
SERVICE_POLL_INTERVAL: float = 2.0
TELEMETRY_INTERVAL: float = 1.0
COLLECTOR_SHUTDOWN_TIMEOUT: float = 2.0

# Buffer sizes
LOG_BUFFER_SIZE: int = 500
SPARKLINE_LENGTH: int = 10

# LPS (logs per second) parameters
LPS_WINDOW_SECONDS: float = 5.0
LPS_PEAK_WINDOW_SECONDS: float = 60.0
LPS_MIN_SCALE: float = 5.0
LPS_BAR_WIDTH: int = 10
LPS_BAR_FILLED: str = "■"
LPS_BAR_EMPTY: str = "□"

# Terminal minimum dimensions
MIN_TERMINAL_WIDTH: int = 60
MIN_TERMINAL_HEIGHT: int = 15

# Sparkline block characters (8 levels: empty → full)
SPARKLINE_BLOCKS: Tuple[str, ...] = (" ", "▂", "▃", "▄", "▅", "▆", "▇", "█")

# Box drawing characters
BOX: Dict[str, str] = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│",
}

# Ping targets: name → (method, target)
PING_TARGETS: Dict[str, Tuple[str, str]] = {
    "Ollama": ("http", "http://localhost:11434"),
    "DNS 1.1.1.1": ("socket", "1.1.1.1"),
    "PolicyGate": ("socket_file", "/run/kaiacord/policy_gate.sock"),
}

# Monitored systemd services and processes
MONITORED_SERVICES: Tuple[str, ...] = (
    "ollama.service",
    "postgresql.service",
    "PolicyGate",
)

# Fallback process-name search terms per service
SERVICE_SEARCH_TERMS: Dict[str, str] = {
    "ollama.service": "ollama",
    "postgresql.service": "postgres",
    "PolicyGate": "policy_gate.py",
}

# Log filter keyword sets (frozen for thread safety)
NET_FILTER_KEYWORDS: frozenset = frozenset({
    "ping", "dns", "connection", "socket", "http", "network",
    "block_ip", "nftables", "firewall", "mitigation",
})
HW_FILTER_KEYWORDS: frozenset = frozenset({
    "temp", "thermal", "throttle", "split lock", "nvidia", "cpu",
    "gpu", "fan", "sensor", "limit", "power", "bus_lock",
})
SEC_FILTER_KEYWORDS: frozenset = frozenset({
    "denied", "blocked", "violation", "unauthorized", "policy_gate",
    "lattice", "sandbox", "capability", "token", "sentinel",
})

# Log severity classification keywords
ERROR_KEYWORDS: frozenset = frozenset({
    "ERROR", "FAIL", "CRITICAL", "SEGFAULT", "BUS_LOCK", "ALARM",
})
WARN_KEYWORDS: frozenset = frozenset({"WARN", "WARNING"})
SUCCESS_KEYWORDS: frozenset = frozenset({"SUCCESS", "STARTING", "STARTED"})

# Split-lock detection patterns
SPLIT_LOCK_PATTERN = re.compile(r"split\s+lock", re.IGNORECASE)
SPLIT_LOCK_APP_PATTERN = re.compile(
    r"split\s+lock.*?[\s:/](\S+?)(?:\[|\s|$)", re.IGNORECASE,
)

# HTTP user-agent for ping requests (prevents 403 blocks)
HTTP_USER_AGENT: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


# ==================== DATACLASSES ====================

@dataclass
class ServiceState:
    """Mutable service state owned by the service collector."""
    name: str
    active: bool = False
    substate: str = "inactive"
    pid: int = 0
    restarts: int = 0
    cpu: float = 0.0
    memory_mb: float = 0.0


@dataclass
class LogEntry:
    """A single parsed journal log entry."""
    timestamp: str
    level: str      # "INFO" | "WARN" | "ERROR" | "SUCCESS"
    message: str


@dataclass(frozen=True)
class KaiamonSnapshot:
    """
    Immutable snapshot of all application state for rendering.

    Created once per frame by CollectorManager.take_snapshot().
    The UI thread renders exclusively from this object and never
    inspects collector-owned structures directly.
    """
    logs: Tuple[LogEntry, ...] = field(default_factory=tuple)
    lps: float = 0.0
    lps_scale: float = 5.0
    filter_mode: str = "All"
    paused: bool = False
    threat_intel: dict = field(default_factory=dict)
    containment: dict = field(default_factory=dict)
    system_security: dict = field(default_factory=dict)
    snapshot_time: float = field(default_factory=time.time)


# ==================== LAYOUT MANAGER ====================

@dataclass
class Pane:
    """Represents a rectangular pane in the terminal layout."""
    y: int
    x: int
    height: int
    width: int
    title: str = ""
    footer: str = ""
    color_pair: int = 1


class LayoutManager:
    """Calculates pane positions for the five-panel layout."""

    def __init__(self) -> None:
        self.height: int = 24
        self.width: int = 80
        self.panes: Dict[str, Pane] = {}

    def calculate_layout(self, height: int, width: int) -> bool:
        """Calculate pane dimensions.  Returns False if terminal is too small."""
        self.height = height
        self.width = width
        if height < MIN_TERMINAL_HEIGHT or width < MIN_TERMINAL_WIDTH:
            return False

        # Command interface takes bottom 5 lines
        cmd_height = 5
        top_height = height - cmd_height

        half_width = width // 2
        half_height = top_height // 2

        self.panes["logs"] = Pane(
            y=0, x=0, height=half_height, width=half_width,
            title="SECURITY AUDIT LOG", color_pair=1,
        )
        self.panes["threat_intel"] = Pane(
            y=0, x=half_width, height=half_height, width=width - half_width,
            title="THREAT INTELLIGENCE", color_pair=2,
        )
        self.panes["containment"] = Pane(
            y=half_height, x=0, height=top_height - half_height, width=half_width,
            title="CONTAINMENT & SENTINEL", color_pair=1,
        )
        self.panes["system_security"] = Pane(
            y=half_height, x=half_width, height=top_height - half_height, width=width - half_width,
            title="SYSTEM SECURITY", color_pair=2,
        )

        menu_footer = "[Q]uit [C]lear [P]ause [F]ilter:All/Sec/Net/Hw"
        self.panes["command"] = Pane(
            y=top_height, x=0, height=cmd_height, width=width,
            title="COMMAND INTERFACE", footer=menu_footer, color_pair=1,
        )
        return True


class ThreatIntelCollector(threading.Thread):
    """Pane 2: Threat Intelligence data collector."""
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="threat-intel-collector")
        self._stop = stop_event
        self._lock = threading.Lock()
        self._data = {
            "active_blocks": 0,
            "recent_blocks": [],
            "gate_offline": False,
            "rules_counts": {"input": 0, "forward": 0, "output": 0},
            "recent_assets": [],
            "honeypot_status": {"last_trigger_ip": "None", "last_trigger_time": "None", "total_triggers": 0}
        }
        
    def get_data(self) -> dict:
        with self._lock:
            return dict(self._data)
            
    def _get_nft_counts(self):
        counts = {"input": 0, "forward": 0, "output": 0}
        gate_offline = False
        try:
            res = subprocess.run(["sudo", "nft", "-nn", "list", "ruleset"], capture_output=True, text=True, timeout=3.0)
            if res.returncode != 0:
                gate_offline = True
            else:
                lines = res.stdout.splitlines()
                current_chain = None
                for line in lines:
                    line_strip = line.strip()
                    if "chain " in line_strip:
                        if "input" in line_strip:
                            current_chain = "input"
                        elif "forward" in line_strip:
                            current_chain = "forward"
                        elif "output" in line_strip:
                            current_chain = "output"
                        else:
                            current_chain = None
                    elif line_strip == "}":
                        current_chain = None
                    elif current_chain and ("drop" in line_strip.lower() or "reject" in line_strip.lower()):
                        counts[current_chain] += 1
        except Exception:
            gate_offline = True
        return counts, gate_offline

    def run(self) -> None:
        from security import threat_intel
        while not self._stop.is_set():
            counts, gate_offline = self._get_nft_counts()
            
            if not os.path.exists(config.POLICY_GATE_SOCKET):
                gate_offline = True
            else:
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    s.connect(config.POLICY_GATE_SOCKET)
                    s.close()
                except Exception:
                    gate_offline = True
                
            active_blocks = counts["input"]
            
            recent_blocks = []
            if os.path.exists(config.SECURITY_DB_PATH):
                try:
                    conn = sqlite3.connect(config.SECURITY_DB_PATH, timeout=1.0)
                    cursor = conn.cursor()
                    threshold = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
                    cursor.execute("""
                        SELECT actor, timestamp FROM security_events
                        WHERE type = 'block_ip' AND disposition = 'approved' AND timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (threshold,))
                    rows = cursor.fetchall()
                    unique_ips = []
                    for row in rows:
                        ip = row[0]
                        if ip not in unique_ips:
                            unique_ips.append(ip)
                        if len(unique_ips) >= 5:
                            break
                    conn.close()
                    
                    for ip in unique_ips:
                        rep = threat_intel.get_ip_reputation(ip)
                        shodan = threat_intel.lookup_internetdb(ip)
                        geo = threat_intel.lookup_geoip(ip)
                        recent_blocks.append({
                            "ip": ip,
                            "geo": geo.get("country", "Unknown"),
                            "tags": shodan.get("tags", []),
                            "ports": shodan.get("ports", [])
                        })
                except Exception:
                    pass
            
            recent_assets = []
            try:
                from security.network_discovery import PassiveDiscoveryEngine
                discovery = PassiveDiscoveryEngine.get_instance()
                recent_assets = discovery.get_recent_assets(5)
            except Exception as e:
                logger.error(f"Failed to fetch LAN assets: {e}")

            # Honeypot triggers
            total_triggers = 0
            last_ip = "None"
            last_time = "None"
            if os.path.exists(config.SECURITY_DB_PATH):
                try:
                    conn = sqlite3.connect(config.SECURITY_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT actor, timestamp FROM security_events
                        WHERE type IN ('honeypot_port_trigger', 'honeypot_file_access')
                        ORDER BY timestamp DESC
                    """)
                    rows = cursor.fetchall()
                    total_triggers = len(rows)
                    if rows:
                        last_ip = rows[0][0]
                        last_time = rows[0][1]
                    conn.close()
                except Exception:
                    pass
            honeypot_status = {
                "last_trigger_ip": last_ip,
                "last_trigger_time": last_time,
                "total_triggers": total_triggers
            }

            with self._lock:
                self._data = {
                    "active_blocks": active_blocks,
                    "recent_blocks": recent_blocks,
                    "gate_offline": gate_offline,
                    "rules_counts": counts,
                    "recent_assets": recent_assets,
                    "honeypot_status": honeypot_status
                }
            self._stop.wait(10.0)


class ContainmentCollector(threading.Thread):
    """Pane 3: Containment & Sentinel data collector."""
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="containment-collector")
        self._stop = stop_event
        self._lock = threading.Lock()
        self._data = {
            "lattice_str": "",
            "sandbox_tier": "",
            "cgroup_cpu": "",
            "cgroup_mem": "",
            "cgroup_pids": 0,
            "sentinel_alerts": 0,
            "last_alert_path": "None",
            "fim_alerts": [],
            "privilege_alerts": []
        }
        
    def get_data(self) -> dict:
        with self._lock:
            return dict(self._data)
            
    def run(self) -> None:
        while not self._stop.is_set():
            g_lvl = config.GLOBAL_LATTICE_LEVEL
            w_lvl = config.WORKSPACE_LATTICE_LEVEL
            g_idx = config.LATTICE_LEVELS.index(g_lvl) if g_lvl in config.LATTICE_LEVELS else 0
            w_idx = config.LATTICE_LEVELS.index(w_lvl) if w_lvl in config.LATTICE_LEVELS else 0
            eff_idx = max(g_idx, w_idx)
            eff_lvl = config.LATTICE_LEVELS[eff_idx]
            
            lattice_str = f"GLOBAL({g_idx}) ∩ WORKSPACE({w_idx}) → EFFECTIVE({eff_idx})"
            sandbox_tier = eff_lvl
            
            cgroup_cpu = config.CGROUP_CPU_QUOTA
            cgroup_mem = config.CGROUP_MEMORY_MAX
            cgroup_pids = config.CGROUP_TASKS_MAX
            
            sentinel_alerts = 0
            last_alert_path = "None"
            
            if os.path.exists(config.SECURITY_DB_PATH):
                try:
                    conn = sqlite3.connect(config.SECURITY_DB_PATH, timeout=1.0)
                    cursor = conn.cursor()
                    threshold = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
                    cursor.execute("""
                        SELECT COUNT(*), MAX(timestamp) FROM security_events
                        WHERE type = 'telemetry_script_sentinel_alert' AND timestamp >= ?
                    """, (threshold,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        sentinel_alerts = row[0]
                        cursor.execute("""
                            SELECT actor FROM security_events
                            WHERE type = 'telemetry_script_sentinel_alert' AND timestamp >= ?
                            ORDER BY timestamp DESC LIMIT 1
                        """, (threshold,))
                        alert_row = cursor.fetchone()
                        if alert_row:
                            last_alert_path = alert_row[0]
                    conn.close()
                except Exception:
                    pass
            
            fim_alerts = []
            fim_db_path = "/var/lib/secdaemon/fim_audit.db"
            if os.path.exists(fim_db_path):
                try:
                    conn = sqlite3.connect(fim_db_path, timeout=1.0)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT timestamp, event_type, pid, comm, path, yara_matches, sha256
                        FROM fim_events ORDER BY rowid DESC LIMIT 5
                    """)
                    rows = cursor.fetchall()
                    for row in rows:
                        fim_alerts.append({
                            "timestamp": row[0], "event_type": row[1], "pid": row[2],
                            "comm": row[3], "path": row[4], "yara_matches": row[5], "sha256": row[6]
                        })
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to query FIM audit DB: {e}")

            privilege_alerts = []
            if os.path.exists(config.SECURITY_DB_PATH):
                try:
                    conn = sqlite3.connect(config.SECURITY_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT actor, timestamp, payload_hash FROM security_events
                        WHERE type = 'privilege_escalation'
                        ORDER BY timestamp DESC LIMIT 3
                    """)
                    rows = cursor.fetchall()
                    for r in rows:
                        privilege_alerts.append({
                            "comm_pid": r[0],
                            "timestamp": r[1],
                            "details": r[2]
                        })
                    conn.close()
                except Exception:
                    pass

            with self._lock:
                self._data = {
                    "lattice_str": lattice_str,
                    "sandbox_tier": sandbox_tier,
                    "cgroup_cpu": cgroup_cpu,
                    "cgroup_mem": cgroup_mem,
                    "cgroup_pids": cgroup_pids,
                    "sentinel_alerts": sentinel_alerts,
                    "last_alert_path": last_alert_path,
                    "fim_alerts": fim_alerts,
                    "privilege_alerts": privilege_alerts
                }
            self._stop.wait(2.0)


class SystemSecurityCollector(threading.Thread):
    """Pane 4: System Security data collector."""
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="system-security-collector")
        self._stop = stop_event
        self._lock = threading.Lock()
        self._data = {
            "tokens_active": 0,
            "tokens_expired": 0,
            "tokens_rejected": 0,
            "rules_counts": {"input": 0, "forward": 0, "output": 0},
            "gate_running": False,
            "gate_pid": 0,
            "gate_uptime": "N/A",
            "yara_rules_count": 0,
            "lockdown_active": False
        }
        
    def get_data(self) -> dict:
        with self._lock:
            return dict(self._data)
            
    def _parse_tokens(self):
        active = 0
        expired = 0
        rejected = 0
        seen_signatures = set()
        
        if os.path.exists(config.AUDIT_LOG_PATH):
            try:
                with open(config.AUDIT_LOG_PATH, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            result = record.get("result", "")
                            cap_token_str = record.get("capability_token")
                            
                            if not cap_token_str:
                                continue
                                
                            token_data = json.loads(cap_token_str)
                            sig = token_data.get("signature")
                            if not sig or sig in seen_signatures:
                                continue
                            seen_signatures.add(sig)
                            
                            if result == "denied":
                                rejected += 1
                                continue
                                
                            expires_str = token_data.get("expires", "")
                            if expires_str:
                                expires_dt = datetime.fromisoformat(expires_str.replace("Z", ""))
                                if datetime.utcnow() > expires_dt:
                                    expired += 1
                                else:
                                    active += 1
                        except Exception:
                            pass
            except Exception:
                pass
        return active, expired, rejected

    def run(self) -> None:
        while not self._stop.is_set():
            active, expired, rejected = self._parse_tokens()
            
            counts = {"input": 0, "forward": 0, "output": 0}
            try:
                res = subprocess.run(["sudo", "nft", "-nn", "list", "ruleset"], capture_output=True, text=True, timeout=2.0)
                if res.returncode == 0:
                    lines = res.stdout.splitlines()
                    current_chain = None
                    for line in lines:
                        line_strip = line.strip()
                        if "chain " in line_strip:
                            if "input" in line_strip:
                                current_chain = "input"
                            elif "forward" in line_strip:
                                current_chain = "forward"
                            elif "output" in line_strip:
                                current_chain = "output"
                            else:
                                current_chain = None
                        elif line_strip == "}":
                            current_chain = None
                        elif current_chain and ("drop" in line_strip.lower() or "reject" in line_strip.lower()):
                            counts[current_chain] += 1
            except Exception:
                pass
                
            gate_running = False
            gate_pid = 0
            gate_uptime = "N/A"
            
            if os.path.exists(config.POLICY_GATE_SOCKET):
                try:
                    res = subprocess.run(["pgrep", "-f", "policy_gate.py"], capture_output=True, text=True)
                    if res.returncode == 0 and res.stdout.strip():
                        gate_pid = int(res.stdout.split()[0])
                        gate_running = True
                        uptime_res = subprocess.run(["ps", "-o", "etime=", "-p", str(gate_pid)], capture_output=True, text=True)
                        if uptime_res.returncode == 0:
                            gate_uptime = uptime_res.stdout.strip()
                except Exception:
                    pass
            
            # Yara rules count
            yara_rules_count = 0
            try:
                if os.path.exists(config.YARA_RULES_DIR):
                    yara_rules_count = len([f for f in os.listdir(config.YARA_RULES_DIR) if f.endswith(".yar")])
            except Exception:
                pass

            # Lockdown active
            lockdown_active = False
            try:
                res = subprocess.run(["/usr/bin/systemctl", "is-active", "kaia-lockdown.service"], capture_output=True, text=True, timeout=2.0)
                if res.stdout.strip() == "active":
                    lockdown_active = True
            except Exception:
                pass

            with self._lock:
                self._data = {
                    "tokens_active": active,
                    "tokens_expired": expired,
                    "tokens_rejected": rejected,
                    "rules_counts": counts,
                    "gate_running": gate_running,
                    "gate_pid": gate_pid,
                    "gate_uptime": gate_uptime,
                    "yara_rules_count": yara_rules_count,
                    "lockdown_active": lockdown_active
                }
            self._stop.wait(5.0)


class ServiceCollector:
    """Monitors systemd service health via systemctl + psutil fallback."""

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop = stop_event
        self._lock = threading.Lock()
        self._services: Dict[str, ServiceState] = {
            name: ServiceState(name=name) for name in MONITORED_SERVICES
        }
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the collector thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="service-collector",
        )
        self._thread.start()

    def get_data(self) -> Tuple[ServiceState, ...]:
        """Return defensive copies of current service states."""
        with self._lock:
            return tuple(
                ServiceState(
                    name=s.name, active=s.active, substate=s.substate,
                    pid=s.pid, restarts=s.restarts, cpu=s.cpu,
                    memory_mb=s.memory_mb,
                )
                for s in self._services.values()
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            for sname in MONITORED_SERVICES:
                if self._stop.is_set():
                    return
                self._poll_service(sname)
            self._stop.wait(SERVICE_POLL_INTERVAL)

    def _poll_service(self, sname: str) -> None:
        """Poll a single service for status and resource usage."""
        active, substate, pid, restarts = False, "inactive", 0, 0
        cpu, mem_mb = 0.0, 0.0

        # Primary: systemctl query
        try:
            result = subprocess.run(
                ["systemctl", "show", sname,
                 "-p", "ActiveState,SubState,MainPID,NRestarts"],
                capture_output=True, text=True, timeout=1.0,
            )
            if result.returncode == 0:
                props: Dict[str, str] = {}
                for line in result.stdout.strip().split("\n"):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        props[k.strip()] = v.strip()
                active = props.get("ActiveState") == "active"
                substate = props.get("SubState", "dead")
                pid = int(props.get("MainPID", "0"))
                restarts = int(props.get("NRestarts", "0"))
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        except Exception:
            pass

        # Fallback: search running processes if systemd reports inactive
        if not active or pid == 0:
            search = SERVICE_SEARCH_TERMS.get(sname, sname.split(".")[0])
            try:
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        cmdline = proc.info.get("cmdline") or []
                        if any(search in arg.lower() for arg in cmdline):
                            active, substate, pid = True, "running", proc.info["pid"]
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied,
                            psutil.ZombieProcess):
                        continue
            except Exception:
                pass

        # Get process resource usage
        if active and pid > 0:
            try:
                p = psutil.Process(pid)
                cpu = p.cpu_percent(interval=None)
                mem_mb = p.memory_info().rss / (1024.0 * 1024.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                pass
            except Exception:
                pass

        with self._lock:
            self._services[sname] = ServiceState(
                name=sname, active=active, substate=substate,
                pid=pid, restarts=restarts, cpu=cpu, memory_mb=mem_mb,
            )



class TelemetryCollector:
    """Collects CPU/GPU thermal, power, and throttle data from sysfs and NVML."""

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop = stop_event
        self._lock = threading.Lock()
        # CPU readings
        self._cpu_temp: float = 0.0
        self._cpu_hotspot: float = 0.0
        self._cpu_power: float = 0.0
        self._cpu_throttled: bool = False
        # GPU readings
        self._gpu_temp: float = 0.0
        self._gpu_power: float = 0.0
        self._gpu_fan: float = 0.0
        self._gpu_mem_used: float = 0.0
        # Split-lock (updated by LogCollector via update_split_lock)
        self._split_lock_events: int = 0
        self._split_lock_last_app: str = "None"
        # Hardware paths (discovered once at startup)
        self._k10temp_path: Optional[str] = None
        self._rapl_path: Optional[str] = None
        self._rapl_is_energy: bool = False  # True=energy_uj, False=power1_input
        self._throttle_paths: List[str] = []
        self._prev_throttle_count: int = 0
        # GPU handles
        self._nvml_handle = None
        self._has_nvml: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Discover hardware paths and launch the collector thread."""
        self._discover_hardware()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="telemetry-collector",
        )
        self._thread.start()

    def get_data(self) -> dict:
        """Return a plain dict copy of current telemetry readings."""
        with self._lock:
            return {
                "cpu_temp": self._cpu_temp,
                "cpu_hotspot": self._cpu_hotspot,
                "cpu_power": self._cpu_power,
                "cpu_throttled": self._cpu_throttled,
                "gpu_temp": self._gpu_temp,
                "gpu_power": self._gpu_power,
                "gpu_fan": self._gpu_fan,
                "gpu_mem_used": self._gpu_mem_used,
                "split_lock_events": self._split_lock_events,
                "split_lock_last_app": self._split_lock_last_app,
            }

    def update_split_lock(self, events: int, last_app: str) -> None:
        """Thread-safe update from LogCollector's split-lock parser."""
        with self._lock:
            self._split_lock_events = events
            self._split_lock_last_app = last_app

    def shutdown_nvml(self) -> None:
        """Release NVML resources."""
        if self._has_nvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # --- hardware discovery (runs once at startup) ---

    def _discover_hardware(self) -> None:
        """Probe sysfs for sensor paths and initialize GPU libraries."""
        # k10temp hwmon path (AMD Ryzen)
        for path in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                with open(os.path.join(path, "name"), "r") as f:
                    if f.read().strip() == "k10temp":
                        self._k10temp_path = path
                        break
            except OSError:
                continue

        # RAPL energy path — try Intel then AMD powercap
        for candidate in (
            "/sys/class/powercap/intel-rapl:0/energy_uj",
            "/sys/class/powercap/amd-rapl:0/energy_uj",
        ):
            if os.path.exists(candidate):
                self._rapl_path = candidate
                self._rapl_is_energy = True
                break

        # Fallback: hwmon power1_input (AMD Ryzen 7000/9000 via k10temp)
        if not self._rapl_path:
            for path in glob.glob("/sys/class/hwmon/hwmon*"):
                power_path = os.path.join(path, "power1_input")
                if os.path.exists(power_path):
                    try:
                        with open(os.path.join(path, "name"), "r") as f:
                            name = f.read().strip()
                        if name in ("k10temp", "amdgpu"):
                            self._rapl_path = power_path
                            self._rapl_is_energy = False
                            break
                    except OSError:
                        continue

        # Kernel thermal throttle sysfs paths
        for cpu_dir in sorted(
            glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle"),
        ):
            for counter in ("package_throttle_count", "core_throttle_count"):
                p = os.path.join(cpu_dir, counter)
                if os.path.exists(p):
                    self._throttle_paths.append(p)
        self._prev_throttle_count = self._read_total_throttle_count()

        # Initialize NVML (pynvml)
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._has_nvml = True
            except Exception:
                self._has_nvml = False

    # --- CPU readings ---

    def _read_total_throttle_count(self) -> int:
        """Sum all kernel throttle counters."""
        total = 0
        for path in self._throttle_paths:
            try:
                with open(path, "r") as f:
                    total += int(f.read().strip())
            except (OSError, ValueError):
                continue
        return total

    def _read_cpu_temps(self) -> Tuple[float, float]:
        """Read CPU temperatures.  Returns (tctl, tccd1)."""
        tctl, tccd1 = 0.0, 0.0
        if self._k10temp_path:
            try:
                with open(os.path.join(self._k10temp_path, "temp1_input")) as f:
                    tctl = float(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                pass
            try:
                with open(os.path.join(self._k10temp_path, "temp3_input")) as f:
                    tccd1 = float(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                tccd1 = tctl
        # Fallback: psutil sensors
        if tctl == 0.0:
            try:
                temps = psutil.sensors_temperatures()
                for source in ("k10temp", "coretemp", "cpu_thermal"):
                    if source in temps and temps[source]:
                        tctl = temps[source][0].current
                        tccd1 = tctl
                        break
            except Exception:
                pass
        return tctl, tccd1

    def _read_cpu_power(self) -> float:
        """Read CPU power via RAPL or estimate from utilization."""
        # RAPL energy_uj path (requires two samples)
        if self._rapl_path and self._rapl_is_energy:
            try:
                with open(self._rapl_path, "r") as f:
                    e1 = float(f.read().strip())
                t1 = time.perf_counter()
                time.sleep(0.1)
                with open(self._rapl_path, "r") as f:
                    e2 = float(f.read().strip())
                dt = time.perf_counter() - t1
                if dt > 0:
                    power = (e2 - e1) / (1e6 * dt)
                    if power > 0:
                        return power
            except (OSError, ValueError):
                pass
        # Direct power1_input path (microwatts)
        elif self._rapl_path and not self._rapl_is_energy:
            try:
                with open(self._rapl_path, "r") as f:
                    return float(f.read().strip()) / 1e6
            except (OSError, ValueError):
                pass
        # Estimation fallback
        try:
            cpu_util = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq()
            curr_f = freq.current if freq else 4000.0
            max_f = (freq.max if freq and freq.max else 5486.0)
            return 15.0 + (88.0 - 15.0) * (cpu_util / 100.0) * (curr_f / max_f)
        except Exception:
            return 0.0

    def _check_throttling(self, tctl: float) -> bool:
        """Detect CPU throttling via kernel counters, with heuristic fallback."""
        if self._throttle_paths:
            current = self._read_total_throttle_count()
            if current > self._prev_throttle_count:
                self._prev_throttle_count = current
                return True
        # Heuristic fallback
        try:
            cpu_util = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq()
            if freq:
                max_f = freq.max or 5486.0
                if tctl > 90.0 or (cpu_util > 90.0 and freq.current < max_f * 0.75):
                    return True
        except Exception:
            pass
        return False

    # --- GPU readings ---

    def _read_gpu(self) -> Tuple[float, float, float, float]:
        """Read GPU telemetry.  Returns (temp, power_w, fan_pct, mem_used_mb)."""
        # Priority 1: pynvml (subprocess-free)
        if self._has_nvml and self._nvml_handle:
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU,
                ))
                power = float(pynvml.nvmlDeviceGetPowerUsage(
                    self._nvml_handle,
                )) / 1000.0
                try:
                    fan = float(pynvml.nvmlDeviceGetFanSpeed(self._nvml_handle))
                except pynvml.NVMLError:
                    fan = 0.0  # Not supported on some GPU models
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return temp, power, fan, float(mem.used) / (1024.0 * 1024.0)
            except Exception:
                pass

        # Priority 2: nvidia-smi subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu,power.draw,fan.speed,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.0,
            )
            if result.returncode == 0:
                parts = [v.strip() for v in result.stdout.strip().split(",")]
                if len(parts) >= 4:

                    def _safe_float(s: str) -> float:
                        try:
                            return float(s)
                        except (ValueError, TypeError):
                            return 0.0

                    return (
                        _safe_float(parts[0]),
                        _safe_float(parts[1]),
                        _safe_float(parts[2]),
                        _safe_float(parts[3]),
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception:
            pass
        # Priority 3: degraded mode (no GPU data)
        return 0.0, 0.0, 0.0, 0.0

    # --- thread body ---

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tctl, tccd1 = self._read_cpu_temps()
                cpu_power = self._read_cpu_power()
                throttled = self._check_throttling(tctl)
                gpu_temp, gpu_power, gpu_fan, gpu_mem = self._read_gpu()

                with self._lock:
                    self._cpu_temp = tctl
                    self._cpu_hotspot = tccd1
                    self._cpu_power = cpu_power
                    self._cpu_throttled = throttled
                    self._gpu_temp = gpu_temp
                    self._gpu_power = gpu_power
                    self._gpu_fan = gpu_fan
                    self._gpu_mem_used = gpu_mem
            except Exception:
                pass  # Never crash the collector
            self._stop.wait(TELEMETRY_INTERVAL)


class AuditLogCollector:
    """
    Polls security_events.db and audit_ledger.json for security audit events.

    Replaces the journalctl-based LogCollector from kaiamon.py with a
    SQLite-polling approach. Uses rowid tracking for efficient incremental
    reads — only fetches events newer than the last seen row.

    Owns all log state including the LPS (events-per-second) tracking deque
    and the adaptive LPS scaling logic.  The split-lock parser forwards
    events to TelemetryCollector for display in the thermals pane.
    """

    AUDIT_POLL_INTERVAL: float = 0.25  # 250ms polling cycle

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop = stop_event
        self._lock = threading.Lock()
        self._logs: deque = deque(maxlen=LOG_BUFFER_SIZE)
        self._lps_timestamps: deque = deque()       # float timestamps only
        self._lps_peak_history: deque = deque()      # (timestamp, lps_value)
        self._split_lock_events: int = 0
        self._split_lock_last_app: str = "None"
        self._split_lock_seen: Set[str] = set()      # dedup keys
        self._paused: bool = False
        self._filter_mode: str = "All"
        self._thread: Optional[threading.Thread] = None
        # State tracking for incremental reads
        self._last_seen_rowid: int = 0
        self._last_ledger_size: int = 0
        self._last_ledger_entries: int = 0
        # Resolve DB paths
        self._security_db_path: str = ""
        self._audit_ledger_path: str = ""
        if HAS_CONFIG:
            self._security_db_path = config.SECURITY_DB_PATH
            self._audit_ledger_path = config.AUDIT_LOG_PATH
        else:
            # Fallback: derive from project structure
            base = os.path.dirname(os.path.abspath(__file__))
            self._security_db_path = os.path.join(base, "storage", "security_events.db")
            self._audit_ledger_path = os.path.join(base, "storage", "audit_ledger.json")

    def start(self) -> None:
        """Launch the collector thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="audit-log-collector",
        )
        self._thread.start()

    def stop(self) -> None:
        """No subprocess to terminate, but kept for interface compatibility."""
        pass

    # --- public properties ---

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value

    @property
    def filter_mode(self) -> str:
        return self._filter_mode

    @filter_mode.setter
    def filter_mode(self, value: str) -> None:
        self._filter_mode = value

    def clear(self) -> None:
        """Clear all log data AND LPS state together."""
        with self._lock:
            self._logs.clear()
            self._lps_timestamps.clear()
            self._lps_peak_history.clear()

    def get_data(self) -> Tuple[Tuple[LogEntry, ...], float, float, str, bool]:
        """Returns (logs, lps, lps_scale, filter_mode, paused)."""
        with self._lock:
            logs = tuple(self._logs)
            lps = self._calculate_logs_per_second()
            scale = self._calculate_lps_scale(lps)
            return logs, lps, scale, self._filter_mode, self._paused

    def get_split_lock_data(self) -> Tuple[int, str]:
        """Return current split-lock counters."""
        with self._lock:
            return self._split_lock_events, self._split_lock_last_app

    # --- LPS computation (preserved from original) ---

    def _calculate_logs_per_second(self) -> float:
        """Calculate current events per second from the timestamps-only deque."""
        now = time.time()
        cutoff = now - LPS_WINDOW_SECONDS
        while self._lps_timestamps and self._lps_timestamps[0] < cutoff:
            self._lps_timestamps.popleft()
        return len(self._lps_timestamps) / LPS_WINDOW_SECONDS

    def _calculate_lps_scale(self, current_lps: float) -> float:
        """Adaptive LPS bar scale using a rolling 1-minute peak with headroom."""
        now = time.time()
        self._lps_peak_history.append((now, current_lps))
        cutoff = now - LPS_PEAK_WINDOW_SECONDS
        while self._lps_peak_history and self._lps_peak_history[0][0] < cutoff:
            self._lps_peak_history.popleft()
        if not self._lps_peak_history:
            return LPS_MIN_SCALE
        peak = max(v for _, v in self._lps_peak_history)
        return max(LPS_MIN_SCALE, peak * 1.2)  # 20% headroom

    # --- event classification ---

    def _classify_severity(self, text: str) -> str:
        """Classify event severity from event text."""
        upper = text.upper()
        if any(k in upper for k in ERROR_KEYWORDS):
            return "ERROR"
        if any(k in upper for k in ("DENIED", "BLOCKED", "VIOLATION", "UNAUTHORIZED")):
            return "ERROR"
        if any(k in upper for k in WARN_KEYWORDS):
            return "WARN"
        if any(k in upper for k in SUCCESS_KEYWORDS):
            return "SUCCESS"
        if any(k in upper for k in ("APPROVED", "ALLOWED")):
            return "SUCCESS"
        return "INFO"

    # --- data source: security_events.db ---

    def _poll_security_db(self) -> List[LogEntry]:
        """Read new rows from security_events.db since last poll."""
        entries = []
        if not os.path.exists(self._security_db_path):
            return entries
        try:
            conn = sqlite3.connect(self._security_db_path, timeout=1.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT rowid, timestamp, type, source, actor, payload_hash, disposition, session_id
                FROM security_events
                WHERE rowid > ?
                ORDER BY rowid ASC
                LIMIT 50
            """, (self._last_seen_rowid,))
            rows = cursor.fetchall()
            for row in rows:
                rowid, ts, event_type, source, actor, payload_hash, disposition, session_id = row
                self._last_seen_rowid = rowid
                disp_ts = ts[:19].split("T")[-1] if "T" in str(ts) else str(ts)[:8]
                
                if disposition == "approved":
                    icon = "✓"
                    status = "approved"
                    level = "SUCCESS"
                elif disposition in ("blocked", "denied"):
                    icon = "✗"
                    status = "denied"
                    level = "ERROR"
                else:
                    icon = "⚠"
                    status = disposition
                    level = "WARN"
                    
                detail = f"hash:{payload_hash[:8]}" if payload_hash else ""
                if source:
                    detail += f" ({source})"
                
                msg = f"{icon} {event_type.upper():<12} │ {status:<8} │ {detail}"
                entries.append(LogEntry(timestamp=disp_ts, level=level, message=msg))
            conn.close()
        except Exception:
            pass  # Never crash the collector
        return entries

    # --- data source: audit_ledger.json ---

    def _poll_audit_ledger(self) -> List[LogEntry]:
        """Read new entries from audit_ledger.json since last poll."""
        entries = []
        if not os.path.exists(self._audit_ledger_path):
            return entries
        try:
            file_size = os.path.getsize(self._audit_ledger_path)
            if file_size == self._last_ledger_size:
                return entries  # No changes
            self._last_ledger_size = file_size

            with open(self._audit_ledger_path, "r") as f:
                content = f.read().strip()
            if not content:
                return entries

            if content.startswith("["):
                records = json.loads(content)
            else:
                records = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            new_records = records[self._last_ledger_entries:]
            self._last_ledger_entries = len(records)

            for record in new_records:
                ts = str(record.get("timestamp", ""))
                disp_ts = ts[:19].split("T")[-1] if "T" in ts else ts[:8]
                action = record.get("action", "unknown")
                result = record.get("result", "")
                reason = record.get("reason", "")
                
                if result == "approved":
                    icon = "✓"
                    status = "approved"
                    level = "SUCCESS"
                elif result in ("denied", "blocked"):
                    icon = "✗"
                    status = "denied"
                    level = "ERROR"
                else:
                    icon = "⚠"
                    status = result
                    level = "WARN"
                    
                msg = f"{icon} {action.upper():<12} │ {status:<8} │ {reason[:60]}"
                entries.append(LogEntry(timestamp=disp_ts, level=level, message=msg))
        except Exception:
            pass  # Never crash the collector
        return entries

    # --- split-lock detection (preserved from original) ---

    def _check_split_lock(self, msg: str, ts: str) -> None:
        """Detect split-lock events with deduplication."""
        if SPLIT_LOCK_PATTERN.search(msg):
            app_match = SPLIT_LOCK_APP_PATTERN.search(msg)
            app_name = app_match.group(1) if app_match else "Unknown"
            dedup_key = f"{ts}:{app_name}"
            if dedup_key not in self._split_lock_seen:
                self._split_lock_seen.add(dedup_key)
                self._split_lock_events += 1
                self._split_lock_last_app = app_name
                if len(self._split_lock_seen) > 1000:
                    self._split_lock_seen.clear()

    # --- thread body (polling loop) ---

    def _run(self) -> None:
        """Poll security_events.db and audit_ledger.json on a 250ms cycle."""
        while not self._stop.is_set():
            try:
                new_entries = []
                new_entries.extend(self._poll_security_db())
                new_entries.extend(self._poll_audit_ledger())

                if new_entries:
                    now = time.time()
                    with self._lock:
                        if not self._paused:
                            for entry in new_entries:
                                self._logs.append(entry)
                                self._lps_timestamps.append(now)
                                self._check_split_lock(entry.message, entry.timestamp)
            except Exception:
                pass  # Never crash the collector

            self._stop.wait(self.AUDIT_POLL_INTERVAL)



# ==================== COLLECTOR MANAGER ====================

class CollectorManager:
    """
    Centralized lifecycle manager for all data collectors.

    Responsibilities:
      - Start / stop all collector threads
      - Produce immutable KaiamonSnapshot objects for the UI
      - Manage subprocess cleanup
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._threat_intel = ThreatIntelCollector(self._stop_event)
        self._containment = ContainmentCollector(self._stop_event)
        self._system_security = SystemSecurityCollector(self._stop_event)
        self._log = AuditLogCollector(self._stop_event)

    @property
    def log_collector(self) -> AuditLogCollector:
        """Expose log collector for pause/filter/clear commands."""
        return self._log

    def start_all(self) -> None:
        """Start every collector thread."""
        self._threat_intel.start()
        self._containment.start()
        self._system_security.start()
        self._log.start()

    def stop_all(self) -> None:
        """Signal shutdown and release all resources."""
        self._stop_event.set()
        self._log.stop()

    def take_snapshot(self) -> KaiamonSnapshot:
        """
        Create an immutable snapshot of all collector data.

        This is the ONLY place where collector-owned state is read.
        The returned object is safe to use from the UI thread without locks.
        """
        threat_intel = self._threat_intel.get_data()
        containment = self._containment.get_data()
        system_security = self._system_security.get_data()
        logs, lps, lps_scale, filter_mode, paused = self._log.get_data()

        return KaiamonSnapshot(
            logs=logs,
            lps=lps,
            lps_scale=lps_scale,
            filter_mode=filter_mode,
            paused=paused,
            threat_intel=threat_intel,
            containment=containment,
            system_security=system_security
        )


# ==================== UI ====================

class KaiamonUI:
    """
    Production-quality curses UI for Kaiamon.

    Architecture rules (matching btop_dashboard_v2.py):
      1. curses runs ONLY in this class, ONLY in the main thread
      2. Rendering is driven exclusively by immutable KaiamonSnapshot objects
      3. Terminal is ALWAYS restored on exit (Q, Ctrl-C, SIGTERM, exception)
      4. No collector thread may access curses, layout, or trigger redraws
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._stdscr = None
        self._layout = LayoutManager()
        self._collector_mgr = CollectorManager()
        self._cmd_queue = queue.Queue()
        self._resp_queue = queue.Queue()
        self._input_buffer = ""
        self._cmd_history = []
        self._history_idx = -1
        self._responses = []
        self._lockdown_pending = False
        self._cmd_stop = threading.Event()
        threading.Thread(target=self._command_worker, daemon=True, name="cmd-worker").start()

    def _add_response(self, text: str) -> None:
        """Add a command response line safely."""
        self._resp_queue.put(text)

    def _command_worker(self) -> None:
        """Worker thread to process commands asynchronously."""
        from security.policy_gate import generate_capability_token
        
        def send_pg(action, payload, cap_token):
            import socket
            import json
            import uuid
            
            req_id = str(uuid.uuid4())
            nested = {
                "request_id": req_id,
                "action": action,
                "payload": payload,
                "capability_token": cap_token
            }
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(config.POLICY_GATE_SOCKET)
                payload_bytes = json.dumps(nested).encode('utf-8')
                header = len(payload_bytes).to_bytes(4, byteorder='big')
                s.sendall(header + payload_bytes)
                
                header_resp = bytearray()
                while len(header_resp) < 4:
                    p = s.recv(4 - len(header_resp))
                    if not p:
                        return False, "Connection closed"
                    header_resp.extend(p)
                length = int.from_bytes(header_resp, byteorder='big')
                
                payload_resp = bytearray()
                while len(payload_resp) < length:
                    p = s.recv(length - len(payload_resp))
                    if not p:
                        return False, "Connection closed"
                    payload_resp.extend(p)
                s.close()
                resp = json.loads(payload_resp.decode('utf-8'))
                return resp.get("approved", False), resp.get("executor_response", {})
            except Exception as e:
                return False, f"Socket error: {e}"

        while not self._cmd_stop.is_set():
            try:
                cmd_line = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            cmd_line = cmd_line.strip()
            if not cmd_line:
                continue
                
            self._add_response(f"> {cmd_line}")
            
            try:
                if self._lockdown_pending:
                    self._lockdown_pending = False
                    if cmd_line.lower() in ("y", "yes"):
                        self._add_response("Initiating Emergency Network Lockdown...")
                        approved, resp = send_pg("lockdown", {}, None)
                        if approved:
                            self._add_response("[SUCCESS] Emergency Lockdown activated.")
                        else:
                            msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                            self._add_response(f"[ERROR] Lockdown failed to trigger: {msg}")
                    else:
                        self._add_response("Lockdown aborted.")
                    continue

                parts = cmd_line.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                if False:
                    pass
                if cmd == "block" and len(args) >= 1:
                    ip = args[0]
                    self._add_response(f"Mitigating threat IP: {ip}...")
                    token = generate_capability_token("block_ip", ip, duration_seconds=15)
                    payload = {"target_ip": ip, "protocol": "all", "session_id": "dashboard"}
                    approved, resp = send_pg("block_ip", payload, token)
                    if approved:
                        self._add_response(f"[APPROVED] Firewall updated for IP {ip}.")
                    else:
                        msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                        self._add_response(f"[DENIED] block_ip rejected: {msg}")
                        
                elif cmd == "show" and len(args) >= 1 and args[0] == "rules":
                    self._add_response("Querying host firewall rules...")
                    payload = {"query_type": "nft_list", "args": [], "justification": "operator query", "session_id": "dashboard"}
                    approved, resp = send_pg("diagnostics", payload, None)
                    if approved:
                        stdout = resp.get("stdout", "")
                        for line in stdout.splitlines()[:30]:
                            self._add_response(line)
                        if len(stdout.splitlines()) > 30:
                            self._add_response("... (output truncated)")
                    else:
                        msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                        self._add_response(f"[DENIED] show rules rejected: {msg}")
                        
                elif cmd == "restart" and len(args) >= 1:
                    srv = args[0]
                    self._add_response(f"Restarting service {srv}...")
                    token = generate_capability_token("restart_service", srv, duration_seconds=15)
                    payload = {"service_name": srv, "justification": "operator request", "session_id": "dashboard"}
                    approved, resp = send_pg("restart_service", payload, token)
                    if approved:
                        self._add_response(f"[APPROVED] Service {srv} restarted successfully.")
                    else:
                        msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                        self._add_response(f"[DENIED] restart {srv} rejected: {msg}")
                        
                elif cmd == "list" and len(args) >= 2 and args[0] == "recent" and args[1] == "blocks":
                    self._add_response("Querying recent blocked IPs...")
                    if os.path.exists(config.SECURITY_DB_PATH):
                        conn = sqlite3.connect(config.SECURITY_DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT actor, timestamp FROM security_events
                            WHERE type = 'block_ip' AND disposition = 'approved'
                            ORDER BY timestamp DESC LIMIT 10
                        """)
                        rows = cursor.fetchall()
                        if rows:
                            for row in rows:
                                self._add_response(f" - {row[0]} at {row[1]}")
                        else:
                            self._add_response("No recent blocks found.")
                        conn.close()
                    else:
                        self._add_response("Security events database not found.")
                        
                elif cmd == "check" and len(args) >= 2 and args[0] == "threat":
                    ip = args[1]
                    self._add_response(f"Enriching threat intel for IP: {ip}...")
                    from security import threat_intel
                    rep = threat_intel.get_ip_reputation(ip)
                    shodan = threat_intel.lookup_internetdb(ip)
                    self._add_response(f"Reputation: {rep.get('score', 'unknown')}/100")
                    if shodan.get("ports"):
                        self._add_response(f"Open Ports: {shodan.get('ports')}")
                    if shodan.get("tags"):
                        self._add_response(f"Shodan Tags: {shodan.get('tags')}")
                    if shodan.get("vulns"):
                        self._add_response(f"Vulnerabilities: {shodan.get('vulns')}")
                        
                elif cmd == "show" and len(args) >= 3 and args[0] == "audit" and args[1] == "--since":
                    since_str = args[2]
                    try:
                        hours = int(since_str.replace("h", ""))
                        self._add_response(f"Querying audit events in the last {hours} hours...")
                        if os.path.exists(config.SECURITY_DB_PATH):
                            conn = sqlite3.connect(config.SECURITY_DB_PATH)
                            cursor = conn.cursor()
                            threshold = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
                            cursor.execute("""
                                SELECT timestamp, type, actor, disposition FROM security_events
                                WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 20
                            """, (threshold,))
                            rows = cursor.fetchall()
                            if rows:
                                for row in rows:
                                    self._add_response(f"[{row[0][:19]}] {row[1]} │ {row[2]} │ {row[3]}")
                            else:
                                self._add_response("No audit events found in this window.")
                            conn.close()
                        else:
                            self._add_response("Security events database not found.")
                    except ValueError:
                        self._add_response("Invalid usage. E.g.: show audit --since 2h")
                elif cmd == "lockdown":
                    self._add_response("Are you sure? [y/N]")
                    self._lockdown_pending = True
                    
                elif cmd == "add" and len(args) >= 3 and args[0] == "rule":
                    name = args[1]
                    indicator = args[2]
                    mitre = None
                    if len(args) >= 4 and args[3].startswith("mitre:"):
                        mitre = args[3].split(":", 1)[1]
                        
                    self._add_response(f"Compiling and validating YARA rule: {name}...")
                    
                    token = generate_capability_token("write_file", f"rules/{name}.yar", duration_seconds=30)
                    payload = {
                        "rule_name": name,
                        "author": "Kaia Automated Rule Engine",
                        "threat_description": f"Operator rule targeting indicator {indicator}",
                        "target_ioc_indicator": indicator,
                        "mitre_framework_id": mitre,
                        "session_id": "dashboard"
                    }
                    approved, resp = send_pg("add_rule", payload, token)
                    if approved:
                        self._add_response(f"[APPROVED] YARA rule {name} validated and active.")
                    else:
                        msg = resp.get("message") if isinstance(resp, dict) else str(resp)
                        self._add_response(f"[DENIED] add rule failed: {msg}")

                elif cmd == "show" and len(args) >= 2 and args[0] == "assets":
                    self._add_response("Querying passive LAN assets...")
                    try:
                        from security.network_discovery import PassiveDiscoveryEngine
                        discovery = PassiveDiscoveryEngine.get_instance()
                        assets = discovery.get_recent_assets(20)
                        if assets:
                            for a in assets:
                                self._add_response(f" • {a['ip']:<15} {a['mac']} │ {a['vendor']} │ {a['detection_vector']} │ Last: {a['last_seen']}")
                        else:
                            self._add_response("No LAN assets discovered yet.")
                    except Exception as e:
                        self._add_response(f"Error querying assets: {e}")

                elif cmd == "show" and len(args) >= 3 and args[0] == "fim" and args[1] == "alerts":
                    self._add_response("Querying recent FIM alerts...")
                    fim_db_path = "/var/lib/secdaemon/fim_audit.db"
                    if os.path.exists(fim_db_path):
                        try:
                            conn = sqlite3.connect(fim_db_path, timeout=1.0)
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT timestamp, event_type, pid, comm, path, yara_matches, sha256
                                FROM fim_events ORDER BY rowid DESC LIMIT 10
                            """)
                            rows = cursor.fetchall()
                            if rows:
                                for row in rows:
                                    lbl = f" [{row[0][:19]}] {row[1].upper()} │ {os.path.basename(row[4])} │ {row[3]} (PID {row[2]})"
                                    if row[5]:
                                        lbl += f" │ YARA:{row[5]}"
                                    self._add_response(lbl)
                            else:
                                self._add_response("No FIM alerts recorded.")
                            conn.close()
                        except Exception as e:
                            self._add_response(f"Error querying FIM DB: {e}")
                    else:
                        self._add_response("FIM audit database not found.")
                else:
                    self._add_response(f"Unknown command or arguments: {cmd_line}")
            except Exception as e:
                self._add_response(f"Error executing command: {e}")

    # ==================== Terminal Safety ====================

    def run(self) -> None:
        """Entry point — must be called from the main thread."""
        atexit.register(self._restore_terminal)

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            curses.wrapper(self._main_loop)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        finally:
            self._running = False
            self._cmd_stop.set()
            self._collector_mgr.stop_all()
            self._restore_terminal()
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGINT / SIGTERM by requesting clean shutdown."""
        self._running = False

    def _restore_terminal(self) -> None:
        """Ensure the terminal is fully restored regardless of exit path."""
        try:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        except Exception:
            pass
        try:
            sys.stdout.write("\033[0m\033[?25h\033[?1049l\033[H\033[2J")
            sys.stdout.flush()
        except Exception:
            pass

    # ==================== Initialization ====================

    def _init_curses(self, stdscr) -> None:
        """Initialize curses settings and color pairs."""
        self._stdscr = stdscr
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(FRAME_INTERVAL_MS)
        stdscr.keypad(True)

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)      # Borders & titles
        curses.init_pair(2, curses.COLOR_MAGENTA, -1)    # Secondary labels
        curses.init_pair(3, curses.COLOR_GREEN, -1)      # Good / success
        curses.init_pair(4, curses.COLOR_YELLOW, -1)     # Warning
        curses.init_pair(5, curses.COLOR_RED, -1)        # Error / critical
        curses.init_pair(6, -1, -1)                      # Default text
        curses.init_pair(7, curses.COLOR_BLUE, -1)       # Info

    # ==================== Main Loop ====================

    def _main_loop(self, stdscr) -> None:
        """Main UI loop — runs in main thread only."""
        self._init_curses(stdscr)
        self._running = True
        self._collector_mgr.start_all()

        while self._running:
            if not self._handle_input():
                self._running = False
                break

            try:
                snapshot = self._collector_mgr.take_snapshot()
                self._draw_frame(snapshot)
                stdscr.refresh()
            except Exception:
                pass  # Never crash the UI loop

    # ==================== Input Handling ====================

    def _handle_input(self) -> bool:
        """Process keyboard input.  Returns False to exit."""
        if not self._stdscr:
            return True
        try:
            ch = self._stdscr.getch()
            if ch == curses.ERR:
                return True
            if ch in (ord("q"), ord("Q")):
                return False
            elif ch in (ord("c"), ord("C")):
                self._collector_mgr.log_collector.clear()
            elif ch in (ord("p"), ord("P")):
                lc = self._collector_mgr.log_collector
                lc.paused = not lc.paused
            elif ch in (ord("f"), ord("F")):
                lc = self._collector_mgr.log_collector
                modes = ("All", "Sec", "Net", "Hw")
                idx = (modes.index(lc.filter_mode) + 1) % len(modes)
                lc.filter_mode = modes[idx]
            elif ch in (ord("r"), ord("R")):
                self._stdscr.clear()
            elif ch == 27:
                self._input_buffer = ""
            elif ch in (10, 13):
                if self._input_buffer.strip():
                    cmd = self._input_buffer
                    self._cmd_history.append(cmd)
                    if len(self._cmd_history) > 50:
                        self._cmd_history.pop(0)
                    self._history_idx = -1
                    self._cmd_queue.put(cmd)
                    self._input_buffer = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if len(self._input_buffer) > 0:
                    self._input_buffer = self._input_buffer[:-1]
            elif ch in (curses.KEY_UP, 259):
                if self._cmd_history:
                    if self._history_idx == -1:
                        self._history_idx = len(self._cmd_history) - 1
                    elif self._history_idx > 0:
                        self._history_idx -= 1
                    self._input_buffer = self._cmd_history[self._history_idx]
            elif ch in (curses.KEY_DOWN, 258):
                if self._cmd_history and self._history_idx != -1:
                    if self._history_idx < len(self._cmd_history) - 1:
                        self._history_idx += 1
                        self._input_buffer = self._cmd_history[self._history_idx]
                    else:
                        self._history_idx = -1
                        self._input_buffer = ""
            elif 32 <= ch <= 126:
                self._input_buffer += chr(ch)
        except Exception:
            pass
        return True

    # ==================== Drawing Primitives ====================

    def _safe_addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        """Safely write a string with screen-bounds checking."""
        if not self._stdscr:
            return
        max_y, max_x = self._stdscr.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        text = text[:max_x - x]
        try:
            self._stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def _draw_box(self, pane: Pane) -> None:
        """Draw a Unicode box around a pane with optional title and footer."""
        if not self._stdscr:
            return
        max_y, max_x = self._stdscr.getmaxyx()
        y, x, h, w = pane.y, pane.x, pane.height, pane.width
        if y >= max_y or x >= max_x:
            return
        h = min(h, max_y - y)
        w = min(w, max_x - x)
        if h < 3 or w < 3:
            return

        color = curses.color_pair(pane.color_pair)

        # --- top border with title ---
        title_str = f" {pane.title} " if pane.title else ""
        if len(title_str) > w - 4:
            title_str = title_str[:w - 4]
        left_h = (w - 2 - len(title_str)) // 2
        right_h = w - 2 - len(title_str) - left_h
        top = BOX["tl"] + BOX["h"] * left_h + title_str + BOX["h"] * right_h + BOX["tr"]
        self._safe_addstr(y, x, top, color)

        # --- vertical borders + clear interior ---
        for i in range(1, h - 1):
            self._safe_addstr(y + i, x, BOX["v"], color)
            self._safe_addstr(y + i, x + 1, " " * (w - 2))
            self._safe_addstr(y + i, x + w - 1, BOX["v"], color)

        # --- bottom border with footer ---
        footer_str = f" {pane.footer} " if pane.footer else ""
        if len(footer_str) > w - 4:
            footer_str = footer_str[:w - 4]
        left_f = (w - 2 - len(footer_str)) // 2
        right_f = w - 2 - len(footer_str) - left_f
        bot = BOX["bl"] + BOX["h"] * left_f + footer_str + BOX["h"] * right_f + BOX["br"]
        self._safe_addstr(y + h - 1, x, bot, color)

    def _make_sparkline(self, data: Tuple[float, ...]) -> str:
        """Generate a Unicode sparkline from a tuple of float values."""
        if not data:
            return " " * SPARKLINE_LENGTH
        vals = list(data)
        limit = max((v for v in vals if v >= 0), default=0.0)
        if limit == 0:
            return " " * len(vals)
        spark = ""
        for val in vals:
            clamped = max(0.0, val)
            ratio = clamped / limit
            idx = min(len(SPARKLINE_BLOCKS) - 1, int(ratio * (len(SPARKLINE_BLOCKS) - 1)))
            spark += SPARKLINE_BLOCKS[idx]
        return spark

    # ==================== Pane Drawing ====================

    def _draw_logs_pane(self, snap: KaiamonSnapshot) -> None:
        """Draw the system logs panel with severity coloring and EPS bar."""
        pane = self._layout.panes.get("logs")
        if not pane:
            return

        lps_ratio = (min(1.0, snap.lps / snap.lps_scale) if snap.lps_scale > 0 else 0.0)
        filled = int(lps_ratio * LPS_BAR_WIDTH)
        bar = LPS_BAR_FILLED * filled + LPS_BAR_EMPTY * (LPS_BAR_WIDTH - filled)
        pane.title = (f"SECURITY AUD LOG ─ EPS: {snap.lps:4.1f}/s [{bar}] "
                      f"─ Filter: {snap.filter_mode}")
        if snap.paused:
            pane.title += " ─ PAUSED"

        self._draw_box(pane)
        ly = pane.y + 1
        lx = pane.x + 2
        l_h = pane.height - 2
        l_w = pane.width - 4

        filtered: List[LogEntry] = []
        for log in snap.logs:
            if snap.filter_mode == "Net":
                if not any(k in log.message.lower() for k in NET_FILTER_KEYWORDS):
                    continue
            elif snap.filter_mode == "Hw":
                if not any(k in log.message.lower() for k in HW_FILTER_KEYWORDS):
                    continue
            elif snap.filter_mode == "Sec":
                if not any(k in log.message.lower() for k in SEC_FILTER_KEYWORDS):
                    continue
            filtered.append(log)

        for log in filtered[-l_h:]:
            if ly >= pane.y + pane.height - 1:
                break
            if log.level == "ERROR":
                l_color = curses.color_pair(5) | curses.A_BOLD
            elif log.level == "WARN":
                l_color = curses.color_pair(4) | curses.A_BOLD
            elif log.level == "SUCCESS":
                l_color = curses.color_pair(3) | curses.A_BOLD
            else:
                l_color = curses.color_pair(6)

            self._safe_addstr(ly, lx, f"{log.timestamp} ", curses.color_pair(1))
            self._safe_addstr(ly, lx + 9, log.message[:l_w - 9], l_color)
            ly += 1

    def _draw_threat_intel_pane(self, snap: KaiamonSnapshot) -> None:
        """Draw Pane 2: Threat Intelligence."""
        pane = self._layout.panes.get("threat_intel")
        if not pane:
            return
        self._draw_box(pane)
        ty = pane.y + 1
        tx = pane.x + 2
        tw = pane.width - 4
        
        info = snap.threat_intel
        if not info:
            return
            
        if info.get("gate_offline"):
            self._safe_addstr(ty, tx, "⚠ POLICY GATE OFFLINE".center(tw), curses.color_pair(5) | curses.A_BOLD)
            ty += 2
            
        self._safe_addstr(ty, tx, f"Active Firewall Blocks: {info.get('active_blocks', 0)}", curses.color_pair(2) | curses.A_BOLD)
        ty += 1
        
        self._safe_addstr(ty, tx, "Recent Blocks:", curses.color_pair(1) | curses.A_BOLD)
        ty += 1
        
        for item in info.get("recent_blocks", []):
            if ty >= pane.y + pane.height - 1:
                break
            ip = item.get("ip")
            geo = item.get("geo", "Unknown")
            tags = ",".join(item.get("tags", []))
            ports = ",".join(str(p) for p in item.get("ports", []))
            
            line = f" • {ip:<15} ({geo})"
            if tags:
                line += f" tags:{tags}"
            if ports:
                line += f" ports:{ports}"
            self._safe_addstr(ty, tx, line[:tw], curses.color_pair(6))
            ty += 1

        ty += 1
        # Add LAN Assets
        self._safe_addstr(ty, tx, "LAN Assets (L2/L3 Passive):", curses.color_pair(1) | curses.A_BOLD)
        ty += 1
        for asset in info.get("recent_assets", []):
            if ty >= pane.y + pane.height - 1:
                break
            ip = asset.get("ip")
            vendor = asset.get("vendor", "unknown")
            vector = asset.get("detection_vector", "ARP")
            line = f" • {ip:<15} {vendor} ({vector})"
            self._safe_addstr(ty, tx, line[:tw], curses.color_pair(6))
            ty += 1

        # Add Honeypot status
        hp = info.get("honeypot_status", {})
        if ty < pane.y + pane.height - 1:
            ty += 1
            self._safe_addstr(ty, tx, "Honeypots / Decoys:", curses.color_pair(1) | curses.A_BOLD)
            ty += 1
            triggers = hp.get("total_triggers", 0)
            hp_color = curses.color_pair(5) | curses.A_BOLD if triggers > 0 else curses.color_pair(3)
            self._safe_addstr(ty, tx, f" • Triggers: {triggers}", hp_color)
            ty += 1
            if triggers > 0 and ty < pane.y + pane.height - 1:
                last_ip = hp.get("last_trigger_ip", "None")
                self._safe_addstr(ty, tx, f" • Last IP:  {last_ip}", curses.color_pair(4))
                ty += 1

    def _draw_containment_pane(self, snap: KaiamonSnapshot) -> None:
        """Draw Pane 3: Containment & Sentinel."""
        pane = self._layout.panes.get("containment")
        if not pane:
            return
        self._draw_box(pane)
        cy = pane.y + 1
        cx = pane.x + 2
        cw = pane.width - 4
        
        info = snap.containment
        if not info:
            return
            
        self._safe_addstr(cy, cx, "Lattice Calculation:", curses.color_pair(1) | curses.A_BOLD)
        cy += 1
        self._safe_addstr(cy, cx + 2, info.get("lattice_str", ""), curses.color_pair(3) | curses.A_BOLD)
        cy += 2
        
        self._safe_addstr(cy, cx, f"Sandbox Tier: {info.get('sandbox_tier')}", curses.color_pair(6))
        cy += 1
        
        cgroup_str = f"cgroup: CPU {info.get('cgroup_cpu')}  MEM {info.get('cgroup_mem')}  PID {info.get('cgroup_pids')}"
        self._safe_addstr(cy, cx, cgroup_str, curses.color_pair(6))
        cy += 2
        
        self._safe_addstr(cy, cx, "Script Sentinel:", curses.color_pair(1) | curses.A_BOLD)
        cy += 1
        alerts = info.get("sentinel_alerts", 0)
        color = curses.color_pair(5) if alerts > 0 else curses.color_pair(3)
        self._safe_addstr(cy, cx + 2, f"ALERTS: {alerts}", color | curses.A_BOLD)
        cy += 1
        if alerts > 0:
            last_path = info.get("last_alert_path", "None")
            self._safe_addstr(cy, cx + 2, f"Last: {last_path}"[-cw:], curses.color_pair(4))
            cy += 1

        cy += 1
        # Add FIM Alerts
        self._safe_addstr(cy, cx, "FIM Alerts (fanotify):", curses.color_pair(1) | curses.A_BOLD)
        cy += 1
        fim_alerts = info.get("fim_alerts", [])
        if not fim_alerts:
            self._safe_addstr(cy, cx + 2, "No FIM alerts", curses.color_pair(3))
            cy += 1
        else:
            for alert in fim_alerts:
                if cy >= pane.y + pane.height - 1:
                    break
                path = alert.get("path")
                yara = alert.get("yara_matches")
                lbl = f" • {os.path.basename(path)} ({alert.get('event_type')})"
                if yara:
                    lbl += f" YARA:{yara}"
                self._safe_addstr(cy, cx + 2, lbl[:cw], curses.color_pair(4))
                cy += 1
        cy += 1

        # Add Privilege Escalation Alerts
        self._safe_addstr(cy, cx, "Privilege Escalations (eBPF):", curses.color_pair(1) | curses.A_BOLD)
        cy += 1
        privs = info.get("privilege_alerts", [])
        if not privs:
            self._safe_addstr(cy, cx + 2, "No privilege alerts", curses.color_pair(3))
            cy += 1
        else:
            for p in privs:
                if cy >= pane.y + pane.height - 1:
                    break
                self._safe_addstr(cy, cx + 2, f" • {p.get('comm_pid')} {p.get('details')}"[:cw], curses.color_pair(5) | curses.A_BOLD)
                cy += 1

    def _draw_system_security_pane(self, snap: KaiamonSnapshot) -> None:
        """Draw Pane 4: System Security."""
        pane = self._layout.panes.get("system_security")
        if not pane:
            return
        self._draw_box(pane)
        sy = pane.y + 1
        sx = pane.x + 2
        sw = pane.width - 4
        
        info = snap.system_security
        if not info:
            return
            
        self._safe_addstr(sy, sx, "Capability Tokens", curses.color_pair(1) | curses.A_BOLD)
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Active:   {info.get('tokens_active')}", curses.color_pair(3))
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Expired:  {info.get('tokens_expired')}", curses.color_pair(4))
        sy += 1
        rejected = info.get("tokens_rejected", 0)
        r_color = curses.color_pair(5) | curses.A_BOLD if rejected > 0 else curses.color_pair(6)
        self._safe_addstr(sy, sx + 2, f"Rejected: {rejected}", r_color)
        sy += 2
        
        self._safe_addstr(sy, sx, "Firewall Statistics", curses.color_pair(1) | curses.A_BOLD)
        sy += 1
        counts = info.get("rules_counts", {})
        self._safe_addstr(sy, sx + 2, f"Input DROP:   {counts.get('input', 0)}", curses.color_pair(6))
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Forward DROP: {counts.get('forward', 0)}", curses.color_pair(6))
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Output DROP:  {counts.get('output', 0)}", curses.color_pair(6))
        sy += 2
        
        self._safe_addstr(sy, sx, "Policy Gate Health", curses.color_pair(1) | curses.A_BOLD)
        sy += 1
        status_str = f"RUNNING (PID {info.get('gate_pid')})" if info.get("gate_running") else "OFFLINE"
        status_color = curses.color_pair(3) if info.get("gate_running") else curses.color_pair(5) | curses.A_BOLD
        self._safe_addstr(sy, sx + 2, status_str, status_color)
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Uptime: {info.get('gate_uptime')}", curses.color_pair(6))
        sy += 2

        # Add YARA rules count
        self._safe_addstr(sy, sx, "YARA Rule Pipeline", curses.color_pair(1) | curses.A_BOLD)
        sy += 1
        self._safe_addstr(sy, sx + 2, f"Active Rules: {info.get('yara_rules_count', 0)}", curses.color_pair(6))
        sy += 2

        # Add Lockdown Status
        lockdown_active = info.get("lockdown_active", False)
        status_str = "⚠ LOCKDOWN ACTIVE" if lockdown_active else "INACTIVE"
        status_color = curses.color_pair(5) | curses.A_BOLD if lockdown_active else curses.color_pair(3)
        self._safe_addstr(sy, sx, f"Lockdown: {status_str}", status_color)

    def _draw_command_pane(self, snap: KaiamonSnapshot) -> None:
        """Draw Pane 5: Command Interface."""
        pane = self._layout.panes.get("command")
        if not pane:
            return
        self._draw_box(pane)
        cy = pane.y + 1
        cx = pane.x + 2
        ch = pane.height - 2
        cw = pane.width - 4
        
        fit_responses = self._responses[-ch:]
        for line in fit_responses:
            if cy >= pane.y + pane.height - 1:
                break
            self._safe_addstr(cy, cx, line[:cw], curses.color_pair(6))
            cy += 1
            
        self._safe_addstr(pane.y + pane.height - 2, cx, " " * cw, curses.color_pair(6))
        input_str = f"> {self._input_buffer}"
        self._safe_addstr(pane.y + pane.height - 2, cx, input_str[:cw], curses.color_pair(6) | curses.A_BOLD)

    # ==================== Frame Composition ====================

    def _draw_frame(self, snap: KaiamonSnapshot) -> None:
        """Draw a complete frame from an immutable snapshot."""
        if not self._stdscr:
            return
        try:
            self._stdscr.erase()
            max_y, max_x = self._stdscr.getmaxyx()

            if not self._layout.calculate_layout(max_y, max_x):
                self._safe_addstr(0, 0, "Terminal too small!",
                                  curses.color_pair(5) | curses.A_BOLD)
                return

            while not self._resp_queue.empty():
                try:
                    line = self._resp_queue.get_nowait()
                    self._responses.append(line)
                    if len(self._responses) > 100:
                        self._responses.pop(0)
                except Exception:
                    break

            self._draw_logs_pane(snap)
            self._draw_threat_intel_pane(snap)
            self._draw_containment_pane(snap)
            self._draw_system_security_pane(snap)
            self._draw_command_pane(snap)
        except curses.error:
            pass


# ==================== ENTRY POINT ====================

def main() -> None:
    """Launch the Kaiamon system companion."""
    ui = KaiamonUI()
    ui.run()


if __name__ == "__main__":
    main()

