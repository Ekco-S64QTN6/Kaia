import subprocess
import os
import shutil
import logging
import config

logger = logging.getLogger(__name__)

class HostExecutor:
    @staticmethod
    def execute_diagnostics(query_type: str, args: list) -> tuple:
        """Runs diagnostics commands without shell."""
        if query_type == "ss":
            cmd = ["ss"] + args
        elif query_type == "ip_route":
            cmd = ["ip", "route"] + args
        elif query_type == "nft_list":
            cmd = ["nft", "list", "ruleset"]
        else:
            return False, "", "Invalid diagnostics query type"

        if query_type == "nft_list" and os.geteuid() != 0:
            cmd = ["sudo"] + cmd

        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_mitigation(target_ip: str, protocol: str, port: int = None) -> tuple:
        """Adds an nftables drop rule for the IP."""
        import socket
        try:
            socket.inet_aton(target_ip)
        except socket.error:
            try:
                socket.inet_pton(socket.AF_INET6, target_ip)
            except socket.error:
                return False, "", f"Invalid IP address format: {target_ip}"

        # Detect address family so the rule uses the right selector.
        is_v6 = ":" in target_ip
        saddr_sel = "ip6" if is_v6 else "ip"

        table = config.NFT_BLOCK_TABLE
        chain = config.NFT_BLOCK_CHAIN
        prio = config.NFT_BLOCK_PRIORITY
        sudo = [] if os.geteuid() == 0 else ["sudo"]

        # Write into Kaia's OWN table, not ufw's.
        #
        # The previous implementation did `nft add rule ip filter input ...`.
        # On this host "ip filter" is ufw's table (managed via iptables-nft), so
        # the block landed inside ufw's ruleset and the next `ufw reload` --
        # or the next lockdown/unlock cycle -- silently discarded it. A block
        # that can vanish with no signal is worse than no block, because the
        # audit ledger still says it was applied.
        #
        # An "inet" table covers v4 and v6, hook priority -10 puts it ahead of
        # ufw's filter hook (priority 0), and `policy accept` means it only ever
        # drops what we explicitly add. Flush/inspect with:
        #     nft list table inet kaia_block
        ensure_table = sudo + ["nft", "add", "table", "inet", table]
        ok, _, err = HostExecutor._run_cmd(ensure_table)
        if not ok:
            return False, "", f"Failed to create nftables table '{table}': {err}"

        ensure_chain = sudo + [
            "nft", "add", "chain", "inet", table, chain,
            f"{{ type filter hook input priority {prio}; policy accept; }}",
        ]
        ok, _, err = HostExecutor._run_cmd(ensure_chain)
        if not ok:
            return False, "", f"Failed to create nftables chain '{chain}': {err}"

        cmd = sudo + ["nft", "add", "rule", "inet", table, chain, saddr_sel, "saddr", target_ip]
        if protocol in ["tcp", "udp"] and port:
            cmd += [protocol, "dport", str(port)]
        cmd += ["drop"]

        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_service_control(service_name: str) -> tuple:
        """Restarts a systemd unit."""
        # Shared with the Policy Gate via config so the two cannot drift.
        if service_name not in config.SERVICE_RESTART_ALLOWLIST:
            return False, "", f"Service {service_name} is not in the allowlist for restarts."

        cmd = ["systemctl", "restart", service_name]
        if os.geteuid() != 0:
            cmd = ["sudo"] + cmd
        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_state_modification(filepath: str, content: str) -> tuple:
        """Writes to a file, verifying it is within the workspace and not modifying protected files/directories."""
        abs_path = os.path.abspath(os.path.expanduser(filepath))
        workspace_abs = os.path.abspath(config.WORKSPACE_DIR)

        if not abs_path.startswith(workspace_abs):
            return False, "", f"Path modification violation: target path {abs_path} is outside workspace {workspace_abs}."

        # Prevent overwriting scripts, python source files, or hidden configuration files
        filename = os.path.basename(abs_path)
        if filename.endswith(".py") or filename.endswith(".sh"):
            return False, "", "Path modification violation: writing to script or code files (.py, .sh) is blocked."

        BLOCKED_HIDDEN = {".env", ".git"}
        if filename.startswith("."):
            if filename in BLOCKED_HIDDEN:
                return False, "", f"Path modification violation: writing to protected file '{filename}' is blocked."
            return False, "", f"Path modification violation: writing to hidden files ({filename}) is blocked."

        # Prevent modifying folders inside protected directory tree
        rel_path = os.path.relpath(abs_path, workspace_abs)
        parts = rel_path.split(os.sep)

        # Block any path containing .git as a directory component
        if ".git" in parts:
            return False, "", "Path modification violation: writing into .git directory tree is blocked."

        # Block modifying files in system directories
        BLOCKED_DIRS = {"core", "security", "tests", "scripts", "toolbox", "storage"}
        if parts[0] in BLOCKED_DIRS or parts[0].startswith("."):
            return False, "", f"Path modification violation: writing to protected directory '{parts[0]}' is blocked."

        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w") as f:
                f.write(content)
            return True, f"File {abs_path} written successfully.", ""
        except Exception as e:
            return False, "", f"Failed to write file: {e}"

    @staticmethod
    def execute_script(script_name: str, effective_level: str = None) -> tuple:
        """Runs an allowlisted script inside an isolated sandbox matching the effective lattice level."""
        # Defence in depth. The Policy Gate already checks SCRIPT_ALLOWLIST, but
        # tests/test_negative_security.py::test_executor_allowlist_enforced_directly
        # establishes that the executor must also enforce its own allowlist when
        # called without the gate. execute_service_control did; this did not.
        if script_name not in config.SCRIPT_ALLOWLIST:
            return False, "", f"Script '{script_name}' is not in the allowlist for execution."

        # os.path.join("~", name) silently discards the "~" when name is absolute,
        # and "../" components escape the home directory, so a bare join is not a
        # containment boundary. Require a plain filename.
        if os.sep in script_name or (os.altsep and os.altsep in script_name) \
                or script_name in (".", "..") or script_name.startswith("~"):
            return False, "", f"Script name '{script_name}' must be a bare filename, not a path."

        home = os.path.realpath(os.path.expanduser("~"))
        script_path = os.path.realpath(os.path.join(home, script_name))
        if os.path.dirname(script_path) != home:
            return False, "", f"Script path escapes the home directory: {script_path}"
        if not os.path.exists(script_path):
            return False, "", f"Script not found at: {script_path}"
        
        workspace_abs = os.path.abspath(config.WORKSPACE_DIR)

        # Resolve level if not explicitly provided
        if not effective_level:
            try:
                g_idx = config.LATTICE_LEVELS.index(config.GLOBAL_LATTICE_LEVEL)
                w_idx = config.LATTICE_LEVELS.index(config.WORKSPACE_LATTICE_LEVEL)
                effective_idx = max(g_idx, w_idx)
                effective_level = config.LATTICE_LEVELS[effective_idx]
            except Exception:
                effective_level = "bwrap"

        # Construct basic command according to the selected tier
        if effective_level == "none":
            cmd = [script_path]
            
        elif effective_level == "namespace":
            cmd = ["unshare", "--user", "--map-root-user", "--fork", "--pid", "--mount-proc", "--uts", "--ipc", "--net", script_path]
            
        elif effective_level == "sandbox-exec":
            cmd = [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64",
                "--symlink", "usr/sbin", "/sbin",
                "--dir", "/tmp",
                "--proc", "/proc",
                "--dev", "/dev",
                "--unshare-all",
                "--bind", workspace_abs, workspace_abs,
                "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),
                "--tmpfs", os.path.join(workspace_abs, "storage"),
                "--ro-bind", script_path, "/tmp/run_script.sh",
                "--",
                "/tmp/run_script.sh"
            ]
            
        elif effective_level in ["bwrap", "auto"]:
            cmd = [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64",
                "--symlink", "usr/sbin", "/sbin",
                "--dir", "/tmp",
                "--dir", "/run",
                "--proc", "/proc",
                "--dev", "/dev",
                "--unshare-all",
                "--new-session",
                "--die-with-parent",
                "--bind", workspace_abs, workspace_abs,
                "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),
                "--bind", "/dev/null", os.path.join(workspace_abs, "logs", "kaia.log"),
                "--tmpfs", os.path.join(workspace_abs, "storage"),
                "--ro-bind", script_path, "/tmp/run_script.sh",
                "--",
                "/tmp/run_script.sh"
            ]
            
        elif effective_level == "systemd-nspawn":
            machine_dir = "/var/lib/machines/kaia"
            if os.path.exists(machine_dir):
                cmd = ["sudo", "systemd-nspawn", "-D", machine_dir, "--private-users=pick", "/bin/bash", "-c", script_path]
            else:
                logger.warning("systemd-nspawn machine directory not found at /var/lib/machines/kaia. Falling back to bwrap.")
                cmd = [
                    "bwrap",
                    "--ro-bind", "/usr", "/usr",
                    "--symlink", "usr/bin", "/bin",
                    "--symlink", "usr/lib", "/lib",
                    "--symlink", "usr/lib64", "/lib64",
                    "--symlink", "usr/sbin", "/sbin",
                    "--dir", "/tmp",
                    "--dir", "/run",
                    "--proc", "/proc",
                    "--dev", "/dev",
                    "--unshare-all",
                    "--new-session",
                    "--die-with-parent",
                    "--bind", workspace_abs, workspace_abs,
                    "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),
                    "--bind", "/dev/null", os.path.join(workspace_abs, "logs", "kaia.log"),
                    "--tmpfs", os.path.join(workspace_abs, "storage"),
                    "--ro-bind", script_path, "/tmp/run_script.sh",
                    "--",
                    "/tmp/run_script.sh"
                ]
                
        else: # gvisor, firecracker fallbacks
            logger.warning(f"Containment primitive '{effective_level}' not fully installed on host. Falling back to bwrap.")
            cmd = [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64",
                "--symlink", "usr/sbin", "/sbin",
                "--dir", "/tmp",
                "--dir", "/run",
                "--proc", "/proc",
                "--dev", "/dev",
                "--unshare-all",
                "--new-session",
                "--die-with-parent",
                "--bind", workspace_abs, workspace_abs,
                "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),
                "--bind", "/dev/null", os.path.join(workspace_abs, "logs", "kaia.log"),
                "--tmpfs", os.path.join(workspace_abs, "storage"),
                "--ro-bind", script_path, "/tmp/run_script.sh",
                "--",
                "/tmp/run_script.sh"
            ]

        # Apply cgroup resource ceilings via systemd-run scope wrapper.
        # master_plan.md §6.3 makes these mandatory for all script executions,
        # so a wrapper we cannot build is a fail-closed condition (INV-001).
        ok, cgroup_wrapper, err = HostExecutor._build_cgroup_wrapper()
        if not ok:
            return False, "", err
        cmd = cgroup_wrapper + cmd

        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def _build_cgroup_wrapper() -> tuple:
        """Build a systemd-run scope wrapper appropriate to the current context.

        Returns (ok, wrapper_argv, error).

        The Policy Gate runs as a root *system* service, which has no user
        session bus. `systemd-run --user` fails there with "Failed to connect to
        user scope bus: $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not
        defined", which previously broke every sandboxed script execution before
        bwrap was ever reached. Root uses a system scope; an interactive user
        with a session bus uses a user scope.

        §6.3 requires these ceilings on every script execution, so if neither
        form is available we fail closed rather than running uncontained.
        """
        if shutil.which("systemd-run") is None:
            return False, [], (
                "Containment failure: systemd-run not found, so the cgroup "
                "resource ceilings required by the security policy cannot be "
                "applied. Refusing to execute uncontained."
            )

        props = [
            "-p", f"CPUQuota={config.CGROUP_CPU_QUOTA}",
            "-p", f"MemoryMax={config.CGROUP_MEMORY_MAX}",
            "-p", f"TasksMax={config.CGROUP_TASKS_MAX}",
            "-p", f"IOWeight={config.CGROUP_IO_WEIGHT}",
        ]
        base = ["systemd-run", "--scope", "--collect", "--quiet"]

        if os.geteuid() == 0:
            return True, base + props, ""

        if os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            return True, ["systemd-run", "--user", "--scope", "--collect", "--quiet"] + props, ""

        return False, [], (
            "Containment failure: no user session bus available and not running "
            "as root, so cgroup resource ceilings cannot be applied. Refusing to "
            "execute uncontained."
        )



    @staticmethod
    def _run_cmd(cmd: list) -> tuple:
        try:
            logger.info(f"HostExecutor running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=config.TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                return True, result.stdout.strip(), result.stderr.strip()
            else:
                return False, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError as e:
            return False, "", f"Command executable not found: {e}"
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {config.TIMEOUT_SECONDS}s."
        except Exception as e:
            return False, "", str(e)
