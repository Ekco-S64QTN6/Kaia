import subprocess
import os
import pwd
import ipaddress
import re
import stat
import uuid
import errno

SYSTEM_BINARIES = {
    "ss": "/usr/bin/ss",
    "ip": "/usr/bin/ip",
    "nft": "/usr/bin/nft",
    "sudo": "/usr/bin/sudo",
    "systemctl": "/usr/bin/systemctl",
    "unshare": "/usr/bin/unshare",
    "bwrap": "/usr/bin/bwrap",
    "systemd-run": "/usr/bin/systemd-run",
    "systemd-nspawn": "/usr/bin/systemd-nspawn",
    "bash": "/usr/bin/bash",
}
import logging
import config

logger = logging.getLogger(__name__)

class HostExecutor:
    @staticmethod
    def execute_diagnostics(query_type: str, args: list) -> tuple:
        """Runs strictly read-only diagnostics commands without a shell."""
        import re

        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            return False, "", "Diagnostics arguments must be a list of strings"

        if query_type == "ss":
            # ss has mutating modes (notably -K/--kill). Accept only compact
            # read-only display flags, with no operands or option-value flags.
            if any(not re.fullmatch(r"-[Hntulaprome46]+", arg) for arg in args):
                return False, "", "Unsupported ss diagnostics option; only read-only display flags are allowed"
            cmd = [SYSTEM_BINARIES["ss"]] + args
        elif query_type == "ip_route":
            # Passing arbitrary arguments here would expose route mutation
            # operations such as add, del, change, replace, and flush. Keep
            # this endpoint to default/show/list reads, optionally family-scoped.
            allowed_args = {
                (), ("show",), ("list",), ("-4",), ("-6",),
                ("-4", "show"), ("-6", "show"),
                ("-4", "list"), ("-6", "list"),
            }
            if tuple(args) not in allowed_args:
                return False, "", "Unsupported ip route diagnostics arguments; route mutations are forbidden"
            cmd = [SYSTEM_BINARIES["ip"], "route"] + args
        elif query_type == "nft_list":
            if args:
                return False, "", "nft_list does not accept arguments"
            cmd = [SYSTEM_BINARIES["nft"], "list", "ruleset"]
        else:
            return False, "", "Invalid diagnostics query type"

        if query_type == "nft_list" and os.geteuid() != 0:
            cmd = [SYSTEM_BINARIES["sudo"]] + cmd

        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_mitigation(target_ip: str, protocol: str, port: int = None) -> tuple:
        """Adds an nftables drop rule for the IP."""
        try:
            address = ipaddress.ip_address(target_ip)
            target_ip = address.compressed
        except (ValueError, TypeError):
            return False, "", f"Invalid IP address format: {target_ip}"

        # Never reinterpret an invalid protocol/port combination as an
        # IP-wide block. That used to happen for protocol='icmp', or tcp/udp
        # without a port, because the executor simply omitted the extra match.
        if protocol not in ("all", "tcp", "udp"):
            return False, "", f"Invalid protocol: {protocol}"
        if isinstance(port, bool) or (port is not None and (
            not isinstance(port, int) or not 1 <= port <= 65535
        )):
            return False, "", f"Invalid port: {port}"
        if protocol == "all" and port is not None:
            return False, "", "A port may only be specified with tcp or udp"
        if protocol in ("tcp", "udp") and port is None:
            return False, "", "A port is required when protocol is tcp or udp"

        # Detect address family so the rule uses the right selector.
        is_v6 = address.version == 6
        saddr_sel = "ip6" if is_v6 else "ip"

        table = config.NFT_BLOCK_TABLE
        chain = config.NFT_BLOCK_CHAIN
        prio = config.NFT_BLOCK_PRIORITY
        sudo = [] if os.geteuid() == 0 else [SYSTEM_BINARIES["sudo"]]

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
        # These commands must be idempotent: a table/chain already created
        # by a prior block is normal, not a failure. Without nft's -exist,
        # every block after the first fails and boot restore stops after one.
        ensure_table = sudo + [SYSTEM_BINARIES["nft"], "-exist", "add", "table", "inet", table]
        ok, _, err = HostExecutor._run_cmd(ensure_table)
        if not ok:
            return False, "", f"Failed to ensure nftables table '{table}': {err}"

        ensure_chain = sudo + [
            SYSTEM_BINARIES["nft"], "-exist", "add", "chain", "inet", table, chain,
            f"{{ type filter hook input priority {prio}; policy accept; }}",
        ]
        ok, _, err = HostExecutor._run_cmd(ensure_chain)
        if not ok:
            return False, "", f"Failed to create nftables chain '{chain}': {err}"

        cmd = sudo + [SYSTEM_BINARIES["nft"], "add", "rule", "inet", table, chain, saddr_sel, "saddr", target_ip]
        if protocol in ["tcp", "udp"] and port:
            cmd += [protocol, "dport", str(port)]
        cmd += ["drop"]

        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_unmitigation(target_ip: str, protocol: str = "all", port: int = None) -> tuple:
        """Remove exact matching Kaia-managed nftables drop rule(s), idempotently.

        Rules are removed by their nftables handles, never by flushing a chain or
        deleting rules that merely share an IP. An absent rule is a successful
        desired state; command/permission failures are reported as errors.
        """
        try:
            address = ipaddress.ip_address(target_ip)
            target_ip = address.compressed
        except (ValueError, TypeError):
            return False, "", f"Invalid IP address format: {target_ip}"

        if protocol not in ("all", "tcp", "udp"):
            return False, "", f"Invalid protocol: {protocol}"
        if isinstance(port, bool) or (port is not None and (
            not isinstance(port, int) or not 1 <= port <= 65535
        )):
            return False, "", f"Invalid port: {port}"
        if protocol == "all" and port is not None:
            return False, "", "A port may only be specified with tcp or udp"
        if protocol in ("tcp", "udp") and port is None:
            return False, "", "A port is required when protocol is tcp or udp"

        family = "ip6" if address.version == 6 else "ip"
        expected = [family, "saddr", target_ip]
        if protocol in ("tcp", "udp"):
            expected.extend([protocol, "dport", str(port)])
        expected.append("drop")

        table = config.NFT_BLOCK_TABLE
        chain = config.NFT_BLOCK_CHAIN
        sudo = [] if os.geteuid() == 0 else [SYSTEM_BINARIES["sudo"]]
        nft = SYSTEM_BINARIES["nft"]
        removed = 0

        # Re-read after each deletion so stale handles or concurrent changes do
        # not cause a different rule to be deleted accidentally.
        for _ in range(128):
            ok, stdout, stderr = HostExecutor._run_cmd(
                sudo + [nft, "-a", "list", "chain", "inet", table, chain]
            )
            if not ok:
                error_text = (stderr or stdout).lower()
                # _run_cmd reports a missing executable explicitly. Do not
                # confuse that with nft successfully inspecting an absent
                # Kaia table/chain; only the latter is idempotent success.
                if "command executable not found:" in error_text:
                    return False, "", f"Required nftables executable could not run: {stderr or stdout}"
                if any(marker in error_text for marker in (
                    "no such file or directory", "does not exist", "no such table", "no such chain"
                )):
                    return True, f"No matching Kaia firewall rule (removed {removed}).", ""
                return False, "", f"Failed to inspect Kaia firewall rules: {stderr or stdout}"

            match_handle = None
            for line in stdout.splitlines():
                expression, marker, comment = line.partition("#")
                if not marker:
                    continue
                if expression.split() != expected:
                    continue
                handle_match = re.search(r"\bhandle\s+(\d+)\s*$", comment.strip())
                if handle_match:
                    match_handle = handle_match.group(1)
                    break

            if match_handle is None:
                message = (
                    f"Removed {removed} matching Kaia firewall rule(s)."
                    if removed else "No matching Kaia firewall rule existed."
                )
                return True, message, ""

            ok, _, stderr = HostExecutor._run_cmd(
                sudo + [nft, "delete", "rule", "inet", table, chain, "handle", match_handle]
            )
            if not ok:
                return False, "", f"Failed to delete matching firewall handle {match_handle}: {stderr}"

            removed += 1

        return False, "", "Refusing to continue: more than 128 duplicate matching firewall rules found."

    @staticmethod
    def execute_service_control(service_name: str) -> tuple:
        """Restarts a systemd unit."""
        # Shared with the Policy Gate via config so the two cannot drift.
        if service_name not in config.SERVICE_RESTART_ALLOWLIST:
            return False, "", f"Service {service_name} is not in the allowlist for restarts."

        cmd = [SYSTEM_BINARIES["systemctl"], "restart", service_name]
        if os.geteuid() != 0:
            cmd = [SYSTEM_BINARIES["sudo"]] + cmd
        return HostExecutor._run_cmd(cmd)

    @staticmethod
    def execute_state_modification(filepath: str, content: str) -> tuple:
        """Atomically write an allowed workspace file without following symlink races."""
        if not isinstance(filepath, str) or not filepath:
            return False, "", "Path modification violation: filepath must be a non-empty string."
        if not isinstance(content, str):
            return False, "", "Path modification violation: file content must be a string."

        workspace_abs = os.path.realpath(os.path.abspath(config.WORKSPACE_DIR))
        requested_abs = os.path.abspath(os.path.expanduser(filepath))

        try:
            contained = os.path.commonpath((workspace_abs, requested_abs)) == workspace_abs
        except (ValueError, TypeError):
            contained = False
        if not contained or requested_abs == workspace_abs:
            return False, "", (
                f"Path modification violation: target path {requested_abs} is outside "
                f"workspace {workspace_abs} or is not a file path beneath it."
            )

        # Preserve the policy allowlist on the lexical path. The actual filesystem
        # walk below uses directory descriptors and O_NOFOLLOW instead of resolving
        # symlinks and reopening a mutable path string later.
        rel_path = os.path.relpath(requested_abs, workspace_abs)
        parts = rel_path.split(os.sep)
        if not parts or any(part in ("", ".", "..") for part in parts):
            return False, "", "Path modification violation: invalid relative target path."

        filename = parts[-1]
        if filename.endswith((".py", ".sh")):
            return False, "", "Path modification violation: writing to script or code files (.py, .sh) is blocked."

        if filename.startswith("."):
            if filename in {".env", ".git"}:
                return False, "", f"Path modification violation: writing to protected file '{filename}' is blocked."
            return False, "", f"Path modification violation: writing to hidden files ({filename}) is blocked."

        if ".git" in parts:
            return False, "", "Path modification violation: writing into .git directory tree is blocked."

        blocked_dirs = {"core", "security", "tests", "scripts", "toolbox", "storage"}
        if parts[0] in blocked_dirs or parts[0].startswith("."):
            return False, "", f"Path modification violation: writing to protected directory '{parts[0]}' is blocked."

        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        root_fd = None
        parent_fd = None
        temp_name = None
        try:
            # Open the workspace root and walk every parent relative to an already
            # opened directory. Symlinked parent components are rejected by
            # O_NOFOLLOW; renames cannot redirect the subsequent open into a
            # different path through string re-resolution.
            root_fd = os.open(workspace_abs, directory_flags)
            parent_fd = os.dup(root_fd)
            root_metadata = os.fstat(root_fd)
            enforce_root_tree = os.geteuid() == 0
            if enforce_root_tree and (
                root_metadata.st_uid != 0 or root_metadata.st_mode & 0o022
            ):
                raise PermissionError(
                    "root executor requires a root-owned workspace not writable by group/others"
                )

            for component in parts[:-1]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                except FileNotFoundError:
                    # Preserve the prior behavior of creating parent directories,
                    # but create them relative to the pinned parent descriptor.
                    try:
                        os.mkdir(component, mode=0o755, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                child_stat = os.fstat(child_fd)
                if not stat.S_ISDIR(child_stat.st_mode):
                    os.close(child_fd)
                    raise OSError(f"target parent is not a directory: {component}")
                if enforce_root_tree and (
                    child_stat.st_uid != 0 or child_stat.st_mode & 0o022
                ):
                    os.close(child_fd)
                    raise PermissionError(
                        f"root executor refuses non-root-controlled parent directory: {component}"
                    )
                os.close(parent_fd)
                parent_fd = child_fd

            # Refuse a symlink, directory, device or FIFO as the final target.
            # A missing target is fine; replacement below is atomic and never
            # follows a final symlink even if the name changes after this check.
            try:
                old_stat = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                old_stat = None
            if old_stat is not None and not stat.S_ISREG(old_stat.st_mode):
                raise OSError("target exists but is not a regular file")

            # Write a private temporary file in the exact target directory, fsync
            # its contents, then rename by dirfd. This avoids truncating the target
            # before the new content is complete and never writes through hard links.
            temp_name = f".kaia-write-{uuid.uuid4().hex}.tmp"
            temp_fd = os.open(temp_name, file_flags, 0o600, dir_fd=parent_fd)
            try:
                payload = content.encode("utf-8")
                offset = 0
                while offset < len(payload):
                    written = os.write(temp_fd, payload[offset:])
                    if written <= 0:
                        raise OSError("short write while preparing replacement")
                    offset += written
                os.fsync(temp_fd)
            finally:
                os.close(temp_fd)

            # If the destination changed to a symlink or non-regular entry while
            # content was being written, abort rather than silently replacing it.
            try:
                latest_stat = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                latest_stat = None
            if latest_stat is not None and not stat.S_ISREG(latest_stat.st_mode):
                raise OSError("target changed to a non-regular file during write")

            os.replace(
                temp_name, filename,
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            )
            temp_name = None
            try:
                os.fsync(parent_fd)
            except OSError as e:
                # Rename already succeeded. Report the completed write accurately,
                # while making uncertain directory-entry durability visible.
                logger.warning("Workspace write committed but directory fsync failed for %s: %s", requested_abs, e)
            return True, f"File {requested_abs} written successfully.", ""
        except (OSError, ValueError, TypeError) as e:
            if isinstance(e, OSError) and e.errno in (errno.ELOOP, errno.ENOTDIR):
                return False, "", f"Path modification violation: symlinked or non-directory path component rejected: {e}"
            return False, "", f"Failed to write file safely: {e}"
        finally:
            if temp_name is not None and parent_fd is not None:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if parent_fd is not None:
                os.close(parent_fd)
            if root_fd is not None:
                os.close(root_fd)

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
            cmd = [SYSTEM_BINARIES["unshare"], "--user", "--map-root-user", "--fork", "--pid", "--mount-proc", "--uts", "--ipc", "--net", script_path]
            
        elif effective_level == "sandbox-exec":
            cmd = [
                SYSTEM_BINARIES["bwrap"],
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
                SYSTEM_BINARIES["bwrap"],
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
                sudo_prefix = [] if os.geteuid() == 0 else [SYSTEM_BINARIES["sudo"]]
                cmd = sudo_prefix + [
                    SYSTEM_BINARIES["systemd-nspawn"], "-D", machine_dir,
                    "--private-users=pick", f"--bind-ro={script_path}:/tmp/run_script.sh",
                    SYSTEM_BINARIES["bash"], "/tmp/run_script.sh",
                ]
            else:
                logger.warning("systemd-nspawn machine directory not found at /var/lib/machines/kaia. Falling back to bwrap.")
                cmd = [
                    SYSTEM_BINARIES["bwrap"],
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
                SYSTEM_BINARIES["bwrap"],
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
        systemd_run = SYSTEM_BINARIES["systemd-run"]
        if not os.path.isfile(systemd_run) or not os.access(systemd_run, os.X_OK):
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
        base = [systemd_run, "--scope", "--collect", "--quiet"]

        if os.geteuid() == 0:
            return True, base + props, ""

        if os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            return True, [systemd_run, "--user", "--scope", "--collect", "--quiet"] + props, ""

        return False, [], (
            "Containment failure: no user session bus available and not running "
            "as root, so cgroup resource ceilings cannot be applied. Refusing to "
            "execute uncontained."
        )



    @staticmethod
    def _build_child_environment() -> dict:
        """Pass only non-secret environment needed by host utilities and user scopes."""
        try:
            account = pwd.getpwuid(os.geteuid())
            home, username = account.pw_dir, account.pw_name
        except Exception:
            home, username = "/root", "root"

        child_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": home,
            "USER": username,
            "LOGNAME": username,
            "LANG": "C.UTF-8",
            "LC_ALL": "C",
        }
        # A non-root interactive invocation may need these to reach its own
        # systemd user manager. Never forward arbitrary variables (especially
        # KAIA_CAPABILITY_TOKEN_SECRET) from the Policy Gate's environment.
        if os.geteuid() != 0:
            for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
                value = os.environ.get(name)
                if value:
                    child_env[name] = value
        return child_env

    @staticmethod
    def _run_cmd(cmd: list) -> tuple:
        try:
            if not isinstance(cmd, list) or not cmd or not isinstance(cmd[0], str):
                return False, "", "HostExecutor requires a non-empty argv list."
            if not os.path.isabs(cmd[0]):
                return False, "", (
                    f"HostExecutor requires an absolute executable path; rejected: {cmd[0]!r}"
                )
            logger.info(f"HostExecutor running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env=HostExecutor._build_child_environment(),
                timeout=config.TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return True, result.stdout.strip(), result.stderr.strip()
            return False, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError as e:
            return False, "", f"Command executable not found: {e}"
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {config.TIMEOUT_SECONDS}s."
        except Exception as e:
            return False, "", str(e)
