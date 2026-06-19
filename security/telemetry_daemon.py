import psutil
import logging
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

def get_network_connections() -> list:
    """Returns a list of active network connections in a structured, clean format."""
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
    """Gets currently running processes."""
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
    """Gets unit status. Uses systemctl directly since dbus library might be missing."""
    status = {"unit": unit_name, "state": "unknown", "load_state": "unknown", "active_state": "unknown", "sub_state": "unknown"}
    try:
        # Check unit state via systemctl show
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
