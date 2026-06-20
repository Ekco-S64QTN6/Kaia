# systemd Quick Reference (systemctl)

`systemd` is the init system and service manager for Linux. `systemctl` is the primary command-line tool for interacting with it.

## Service Management
* **Check Status of a Service:**
    `systemctl status <service_name>`
    * Example: `systemctl status nginx`
    * Shows whether it's active, running, disabled, failed, and recent log entries.
* **Start a Service:**
    `sudo systemctl start <service_name>`
* **Stop a Service:**
    `sudo systemctl stop <service_name>`
* **Restart a Service:**
    `sudo systemctl restart <service_name>`
* **Reload a Service (apply config changes without full restart):**
    `sudo systemctl reload <service_name>` (Only works if the service supports it)
* **Enable a Service (start on boot):**
    `sudo systemctl enable <service_name>`
* **Disable a Service (do not start on boot):**
    `sudo systemctl disable <service_name>`
* **Mask a Service (prevent manual or automatic starting):**
    `sudo systemctl mask <service_name>`
* **Unmask a Service:**
    `sudo systemctl unmask <service_name>`

## Unit File Management
* **Reload systemd daemon (after modifying unit files):**
    `sudo systemctl daemon-reload`
* **Edit a Service Unit File:**
    `sudo systemctl edit --full <service_name>` (Edits the main unit file)
    `sudo systemctl edit <service_name>` (Creates/edits an override file for minor changes)
* **View a Service Unit File:**
    `systemctl cat <service_name>`

## System State & Logs
* **List All Running Services:**
    `systemctl list-units --type=service --state=running`
* **List All Failed Services:**
    `systemctl --failed`
* **List All Enabled Services (start on boot):**
    `systemctl list-unit-files --type=service --state=enabled`
* **Reboot System:**
    `sudo systemctl reboot`
* **Shutdown System:**
    `sudo systemctl poweroff`
* **View System Journal (Logs):**
    `journalctl`
    * `journalctl -f`: Follow new log entries in real-time.
    * `journalctl -u <service_name>`: View logs for a specific service.
    * `journalctl --since "today"` or `--since "2025-07-16 09:00:00"`: Filter by time.
    * `journalctl -p err`: View only error messages.

## Targets (Runlevels)
* **List All Targets:**
    `systemctl list-units --type=target`
* **Get Current Default Target:**
    `systemctl get-default`
* **Set Default Target:**
    `sudo systemctl set-default multi-user.target` (for console-only boot)
    `sudo systemctl set-default graphical.target` (for graphical desktop boot)
