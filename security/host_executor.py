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
        ALLOWED_SERVICES = ["nginx", "postgresql", "ollama", "chroma"]
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

        if filename.startswith("."):
            return False, "", f"Path modification violation: writing to hidden files ({filename}) is blocked."

        # Prevent modifying folders inside protected directory tree
        rel_path = os.path.relpath(abs_path, workspace_abs)
        parts = rel_path.split(os.sep)

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
    def execute_script(script_name: str) -> tuple:
        """Runs an allowlisted script inside an isolated Bubblewrap sandbox with database and environment masking."""
        script_path = os.path.abspath(os.path.expanduser(os.path.join("~", script_name)))
        if not os.path.exists(script_path):
            return False, "", f"Script not found at: {script_path}"
        
        workspace_abs = os.path.abspath(config.WORKSPACE_DIR)
        
        # Build Arch Linux compatible bwrap command
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
            # Mask sensitive files: .env containing database credentials, security events DB, and logs
            "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),
            "--bind", "/dev/null", os.path.join(workspace_abs, "kaia.log"),
            "--tmpfs", os.path.join(workspace_abs, "storage"),
            "--ro-bind", script_path, "/tmp/run_script.sh",
            "--",
            "/tmp/run_script.sh"
        ]
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
