# Linux Service Failures Troubleshooting Guide

## Initial Service Status Check
* **Check Overall Status:**
    * `sudo systemctl status <service_name>`: Provides detailed status, including active state, last exit code, and recent log entries.
    * `sudo systemctl is-active <service_name>`: Returns `active` or `inactive`/`failed`.
* **Check All Failed Services:** `systemctl --failed`

## Examining Service Logs
* **Journalctl for Service:**
    * `sudo journalctl -u <service_name>`: Show all logs for a specific service.
    * `sudo journalctl -u <service_name> -e`: Jump to the end of the logs.
    * `sudo journalctl -u <service_name> --since "1 hour ago"`: Filter logs by time.
    * `sudo journalctl -u <service_name> -f`: Follow new log entries in real-time.
* **Application-Specific Logs:** Check `/var/log/<application_name>/` or application configuration for custom log paths.

## Common Systemd Commands
* **Start Service:** `sudo systemctl start <service_name>`
* **Stop Service:** `sudo systemctl stop <service_name>`
* **Restart Service:** `sudo systemctl restart <service_name>` (stops and then starts)
* **Reload Service:** `sudo systemctl reload <service_name>` (if the service supports it for config changes without full restart)
* **Enable Service (start on boot):** `sudo systemctl enable <service_name>`
* **Disable Service (don't start on boot):** `sudo systemctl disable <service_name>`
* **Reload Systemd Daemon:** `sudo systemctl daemon-reload` (after modifying unit files)

## Common Causes & Solutions
* **Configuration Errors:**
    * Check service unit file: `cat /etc/systemd/system/<service_name>.service` or `systemctl cat <service_name>`.
    * Validate config syntax if applicable (e.g., `nginx -t`, `apachectl configtest`).
* **Dependencies:** Service might fail if a required dependency (another service or resource) isn't running. Check `Requires=` and `After=` in unit file.
* **Permissions Issues:** Ensure the service user has necessary read/write permissions for its files/directories. Look for "Permission denied" in logs.
* **Resource Limits:** Service might exceed CPU, memory, or file descriptor limits. Check `Limit*` directives in unit file.
* **Out of Memory (OOM) Killer:** `dmesg -T | grep -i oom`: Check if the OOM killer terminated the service due to low memory.
* **Network Connectivity:** If service relies on network, ensure network is up and reachable.
* **Port Conflicts:** `sudo ss -tulpn | grep <port_number>`: Check if another process is already using the required port.

## Advanced Debugging
* **Dry Run (if supported):** Some services have a test/dry-run mode for configs.
* **Strace:** `sudo strace -p <pid>` or `sudo strace <command>`: Trace system calls. (For advanced users).
* **GDB:** Use a debugger for complex application failures (requires debugging symbols).
