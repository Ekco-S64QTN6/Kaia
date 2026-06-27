import subprocess
import os
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

        if query_type == "nft_list":
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

        # Construct nft command
        cmd = ["sudo", "nft", "add", "rule", "ip", "filter", "input", "ip", "saddr", target_ip]
        if protocol in ["tcp", "udp"] and port:
            cmd += [protocol, "dport", str(port)]
        cmd += ["drop"]
        
        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_service_control(service_name: str) -> tuple:
        """Restarts a systemd unit."""
        ALLOWED_SERVICES = ["nginx", "postgresql", "ollama"]
        if service_name not in ALLOWED_SERVICES:
            return False, "", f"Service {service_name} is not in the allowlist for restarts."

        cmd = ["sudo", "systemctl", "restart", service_name]
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
        script_path = os.path.abspath(os.path.expanduser(os.path.join("~", script_name)))
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

        # Apply cgroup resource ceilings via systemd-run scope wrapper
        cgroup_wrapper = [
            "systemd-run",
            "--user",
            "--scope",
            "-p", f"CPUQuota={config.CGROUP_CPU_QUOTA}",
            "-p", f"MemoryMax={config.CGROUP_MEMORY_MAX}",
            "-p", f"TasksMax={config.CGROUP_TASKS_MAX}",
            "-p", f"IOWeight={config.CGROUP_IO_WEIGHT}"
        ]
        cmd = cgroup_wrapper + cmd

        return HostExecutor._run_cmd(cmd)



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
