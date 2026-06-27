import os
import psutil
import logging
import subprocess
import hashlib
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import config
from security.db import log_security_event

logger = logging.getLogger(__name__)

def get_network_connections() -> list:
    """Returns active network connections from eBPF, falling back to psutil if disabled."""
    try:
        from security.ebpf_telemetry import EBPFTelemetryEngine, HAS_BCC
        if HAS_BCC:
            engine = EBPFTelemetryEngine.get_instance()
            conns = engine.get_recent_connections()
            if conns:
                formatted = []
                for c in conns:
                    formatted.append({
                        "ip": c["daddr"],
                        "port": str(c["dport"]),
                        "pid": str(c["pid"]),
                        "comm": c["comm"],
                        "state": "ESTABLISHED",
                        "timestamp": datetime.fromtimestamp(c["timestamp"]).isoformat() + "Z"
                    })
                return formatted
    except Exception as e:
        logger.warning(f"Failed to fetch network connections via eBPF: {e}")

    # Fallback to psutil
    connections = []
    try:
        for conn in psutil.net_connections(kind='inet'):
            laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else ""
            
            # Map pid to process name if possible
            comm = ""
            if conn.pid:
                try:
                    comm = psutil.Process(conn.pid).name()
                except Exception:
                    pass

            connections.append({
                "ip": conn.raddr.ip if conn.raddr else "",
                "port": str(conn.raddr.port) if conn.raddr else "",
                "pid": str(conn.pid) if conn.pid else "",
                "comm": comm,
                "state": conn.status,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
    except Exception as e:
        logger.error(f"Failed to fetch network connections telemetry: {e}")
    return connections

def get_process_lifecycle_events() -> list:
    """Returns process execution events from eBPF, falling back to psutil if disabled."""
    try:
        from security.ebpf_telemetry import EBPFTelemetryEngine, HAS_BCC
        if HAS_BCC:
            engine = EBPFTelemetryEngine.get_instance()
            execs = engine.get_recent_execs()
            if execs:
                formatted = []
                for e in execs:
                    formatted.append({
                        "pid": str(e["pid"]),
                        "comm": e["comm"],
                        "path": e["filename"],
                        "timestamp": datetime.fromtimestamp(e["timestamp"]).isoformat() + "Z"
                    })
                return formatted
    except Exception as e:
        logger.warning(f"Failed to fetch process telemetry via eBPF: {e}")

    # Fallback to psutil
    processes = []
    try:
        for p in psutil.process_iter(['pid', 'name', 'exe', 'username']):
            try:
                processes.append({
                    "pid": str(p.info['pid']),
                    "comm": p.info['name'],
                    "path": p.info['exe'] or "",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Failed to fetch process telemetry: {e}")
    return processes

def get_systemd_unit_status(unit_name: str) -> dict:
    """Gets unit status. Uses D-Bus via gi.repository.Gio with fallback to systemctl show."""
    status = {"unit": unit_name, "state": "unknown", "load_state": "unknown", "active_state": "unknown", "sub_state": "unknown"}
    
    # 1. Try D-Bus via Gio
    try:
        from gi.repository import Gio, GLib
        bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        proxy = Gio.DBusProxy.new_sync(
            bus, Gio.DBusProxyFlags.NONE, None,
            "org.freedesktop.systemd1",
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager", None
        )
        res = proxy.call_sync(
            "GetUnit",
            GLib.Variant("(s)", (unit_name,)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        unit_path = res.unpack()[0]
        
        unit_proxy = Gio.DBusProxy.new_sync(
            bus, Gio.DBusProxyFlags.NONE, None,
            "org.freedesktop.systemd1",
            unit_path,
            "org.freedesktop.DBus.Properties", None
        )
        
        def get_prop(interface, prop_name):
            val = unit_proxy.call_sync(
                "Get",
                GLib.Variant("(ss)", (interface, prop_name)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return val.unpack()[0]
            
        status["load_state"] = get_prop("org.freedesktop.systemd1.Unit", "LoadState")
        status["active_state"] = get_prop("org.freedesktop.systemd1.Unit", "ActiveState")
        status["sub_state"] = get_prop("org.freedesktop.systemd1.Unit", "SubState")
        status["state"] = status["active_state"]
        return status
    except Exception as e:
        logger.debug(f"D-Bus query failed for {unit_name}, falling back to systemctl: {e}")

    # 2. Fallback to systemctl subprocess
    try:
        result = subprocess.run(
            ["systemctl", "show", unit_name, "--property=LoadState,ActiveState,SubState"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "LoadState":
                        status["load_state"] = v.strip()
                    elif k == "ActiveState":
                        status["active_state"] = v.strip()
                    elif k == "SubState":
                        status["sub_state"] = v.strip()
            status["state"] = status["active_state"]
    except Exception as e:
        logger.error(f"Failed to query systemd status for {unit_name}: {e}")
    return status

# --- Script Sentinel Implementation ---

class ScriptSentinelHandler(FileSystemEventHandler):
    def on_created(self, event):
        self._check_file(event.src_path)

    def on_modified(self, event):
        self._check_file(event.src_path)

    def _check_file(self, filepath):
        if os.path.isdir(filepath):
            return
        
        filename = os.path.basename(filepath)
        if filename.startswith(".") or "__pycache__" in filepath:
            return

        is_script = False
        if filepath.endswith(".sh"):
            is_script = True
        else:
            try:
                with open(filepath, "rb") as f:
                    head = f.read(128)
                    if head.startswith(b"#!"):
                        is_script = True
            except Exception:
                pass

        if is_script:
            logger.warning(f"Script Sentinel: script file change detected: {filepath}")
            try:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                file_hash = "unknown"
            
            log_security_event(
                event_type="telemetry_script_sentinel_alert",
                source="script_sentinel",
                actor=filepath,
                payload_hash=file_hash,
                disposition="blocked",
                session_id="system_sentinel"
            )

_sentinel_observer = None

def start_script_sentinel():
    """Starts the Script Sentinel filesystem watchdog in a background observer thread."""
    global _sentinel_observer
    if _sentinel_observer is not None:
        return
    
    workspace_dir = os.path.abspath(config.WORKSPACE_DIR)
    os.makedirs(workspace_dir, exist_ok=True)
    
    handler = ScriptSentinelHandler()
    _sentinel_observer = Observer()
    _sentinel_observer.schedule(handler, path=workspace_dir, recursive=True)
    _sentinel_observer.daemon = True
    _sentinel_observer.start()
    logger.info(f"Script Sentinel observer started on: {workspace_dir}")

def stop_script_sentinel():
    """Stops the Script Sentinel filesystem watchdog."""
    global _sentinel_observer
    if _sentinel_observer:
        _sentinel_observer.stop()
        _sentinel_observer.join(timeout=2.0)
        _sentinel_observer = None
        logger.info("Script Sentinel observer stopped.")
