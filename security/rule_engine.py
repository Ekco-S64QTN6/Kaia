import os
import re
import sys
import tempfile
import stat
import subprocess
import logging
import threading
import config

logger = logging.getLogger(__name__)

SYSTEMD_RUN_BINARY = "/usr/bin/systemd-run"
YARA_BINARY = "/usr/bin/yara"

HAS_YARA = False
try:
    import yara
    HAS_YARA = True
except ImportError:
    logger.warning("yara-python not found. YARA scanning features will be disabled.")

class RuleEngine:
    _instance = None
    _lock = threading.Lock() if 'threading' in sys.modules else None

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.rules_dir = config.YARA_RULES_DIR
        self.scanner = None
        # A predictable shared /tmp path lets an unprivileged user race root's
        # corpus writes through a symlink. tempfile creates a fresh directory
        # atomically under sticky /tmp; keep it alive for this engine instance.
        self._benign_corpus_tmpdir = tempfile.TemporaryDirectory(
            prefix="kaia_benign_corpus-", dir="/tmp"
        )
        self.benign_corpus_dir = self._benign_corpus_tmpdir.name
        os.chmod(self.benign_corpus_dir, 0o755)
        self.reload_callbacks = []
        
        # Initialize rules directory
        os.makedirs(self.rules_dir, exist_ok=True)
        self._setup_benign_corpus()
        self.reload_rules()

    def register_reload_callback(self, callback):
        self.reload_callbacks.append(callback)

    def _setup_benign_corpus(self):
        """Create clean samples in this engine's exclusively-created temp directory."""
        try:
            directory_metadata = os.lstat(self.benign_corpus_dir)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid != os.geteuid()
                or directory_metadata.st_mode & 0o022
            ):
                raise PermissionError("YARA corpus directory is not trusted")

            lorem_ipsum_variants = [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
                "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.",
                "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui."
            ]
            flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            for i, sample in enumerate(lorem_ipsum_variants):
                filepath = os.path.join(self.benign_corpus_dir, f"benign_{i}.txt")
                # O_EXCL rejects existing entries (including symlinks), while
                # O_NOFOLLOW makes the no-follow intent explicit on supported OSes.
                fd = os.open(filepath, flags, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as corpus_file:
                    corpus_file.write(sample)
                    corpus_file.flush()
                    os.fsync(corpus_file.fileno())
        except Exception as exc:
            logger.error("Failed to initialize the safe YARA corpus: %s", exc, exc_info=True)
            raise RuntimeError(
                "Could not initialize safe YARA validation corpus; refusing rule validation."
            ) from exc

    @staticmethod
    def _escape_yara_literal(value: str) -> str:
        """Encode arbitrary UTF-8 text as a YARA quoted string without source injection."""
        encoded = value.encode("utf-8")
        escaped = []
        for byte in encoded:
            if byte == 0x5C:
                escaped.append("\\\\")
            elif byte == 0x22:
                escaped.append("\\\"")
            elif byte == 0x0A:
                escaped.append("\\n")
            elif byte == 0x0D:
                escaped.append("\\r")
            elif byte == 0x09:
                escaped.append("\\t")
            elif 0x20 <= byte <= 0x7E:
                escaped.append(chr(byte))
            else:
                escaped.append(f"\\x{byte:02X}")
        return "".join(escaped)

    def compile_rule_text(
        self,
        rule_name: str,
        author: str,
        threat_description: str,
        target_ioc_indicator: str,
        mitre_framework_id: str = None,
    ) -> str:
        """Build a YARA rule using escaped literal strings, not YARA source fragments."""
        author_literal = self._escape_yara_literal(author)
        description_literal = self._escape_yara_literal(threat_description)
        ioc_literal = self._escape_yara_literal(target_ioc_indicator)
        mitre_literal = self._escape_yara_literal(mitre_framework_id or "")

        return f"""rule {rule_name} {{
    meta:
        author = "{author_literal}"
        description = "{description_literal}"
        mitre_id = "{mitre_literal}"
    strings:
        $ioc = "{ioc_literal}"
    condition:
        any of them
}}
"""
    def validate_rule(self, rule_text: str) -> tuple:
        """Validate a YARA rule only in the required isolated systemd service.

        There is deliberately no in-process fallback: a missing executable or
        unavailable service manager means we cannot prove the rule is safe to
        activate, so rule ingestion fails closed.
        """
        fd = -1
        rule_path = None
        try:
            fd, rule_path = tempfile.mkstemp(suffix=".yar", prefix="kaia-yara-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1
                f.write(rule_text)
                f.flush()
                os.fsync(f.fileno())

            # DynamicUser cannot read a root-only 0600 temporary file. The path
            # is random, the file is immutable during the check, and the
            # transient service bind-mounts it read-only.
            os.chmod(rule_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

            cmd = [
                SYSTEMD_RUN_BINARY,
                "--wait",
                "--pipe",
                "--collect",
                "--quiet",
                "-p", "PrivateTmp=yes",
                "-p", "ProtectSystem=strict",
                "-p", "PrivateNetwork=yes",
                "-p", "DynamicUser=yes",
                "-p", "NoNewPrivileges=yes",
                "-p", "CapabilityBoundingSet=",
                "-p", "SystemCallFilter=@system-service",
                "-p", "RestrictAddressFamilies=none",
                "-p", f"BindReadOnlyPaths={rule_path}:{rule_path}",
                "-p", f"BindReadOnlyPaths={self.benign_corpus_dir}:{self.benign_corpus_dir}",
                "--",
                YARA_BINARY, rule_path, self.benign_corpus_dir,
            ]
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": "/root",
                "LANG": "C.UTF-8",
                "LC_ALL": "C",
            }
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env=env,
                timeout=10.0,
            )

            if result.returncode != 0:
                err = (result.stderr or result.stdout or "Sandboxed YARA validation failed").strip()
                lower = err.lower()
                if "failed to find executable" in lower or "no such file or directory" in lower or "not found" in lower:
                    return False, f"Required sandbox/YARA executable is unavailable; refusing to activate rule: {err}"
                return False, f"Sandboxed YARA validation failed: {err}"

            if result.stdout.strip():
                return False, "False positive detected: rule matched benign corpus files."
            return True, ""

        except FileNotFoundError as exc:
            logger.error("Required YARA validation sandbox is unavailable: %s", exc)
            return False, "Required sandbox/YARA executable is unavailable; refusing to activate rule."
        except subprocess.TimeoutExpired:
            return False, "Sandboxed YARA validation timed out; refusing to activate rule."
        except Exception as exc:
            logger.error("YARA validation failed closed: %s", exc, exc_info=True)
            return False, f"Sandboxed YARA validation error: {exc}"
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if rule_path:
                try:
                    os.unlink(rule_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.error("Could not remove temporary YARA file %s: %s", rule_path, exc)

    def add_rule(
        self,
        rule_name: str,
        author: str,
        threat_description: str,
        target_ioc_indicator: str,
        mitre_framework_id: str = None,
    ) -> tuple:
        """Validate and activate a generated IOC rule, refusing partial activation."""
        if not isinstance(rule_name, str) or not re.fullmatch(r"[a-zA-Z0-9_]{3,64}", rule_name):
            return False, "Invalid rule name: expected 3-64 alphanumeric/underscore characters."
        if not isinstance(author, str) or not 1 <= len(author) <= 128:
            return False, "Author must contain between 1 and 128 characters."
        if not isinstance(threat_description, str) or not 10 <= len(threat_description) <= 256:
            return False, "Threat description must contain between 10 and 256 characters."
        if not isinstance(target_ioc_indicator, str) or not 4 <= len(target_ioc_indicator) <= 128:
            return False, "IOC indicator must contain between 4 and 128 characters."
        if mitre_framework_id is not None and not re.fullmatch(r"T[0-9]{4}", str(mitre_framework_id)):
            return False, "Invalid MITRE framework ID."
        if not HAS_YARA:
            return False, "yara-python is required to activate rules; refusing to save an inactive rule."

        os.makedirs(self.rules_dir, exist_ok=True)
        filepath = os.path.join(self.rules_dir, f"{rule_name}.yar")
        if os.path.exists(filepath):
            return False, f"Rule '{rule_name}' already exists; refusing to overwrite it."

        rule_text = self.compile_rule_text(
            rule_name, author, threat_description, target_ioc_indicator, mitre_framework_id
        )
        ok, err = self.validate_rule(rule_text)
        if not ok:
            return False, err

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=".kaia-rule-", suffix=".tmp", dir=self.rules_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(rule_text)
                f.flush()
                os.fsync(f.fileno())

            # Hard-link creation is atomic and refuses a concurrent duplicate
            # rather than replacing an existing active rule.
            os.link(temp_path, filepath)
            os.unlink(temp_path)
            temp_path = None
            self.reload_rules()
            if self.scanner is None:
                # Do not report successful activation when the runtime scanner
                # cannot compile the rules directory.
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
                self.reload_rules()
                return False, "Validated rule could not be activated by the runtime scanner."
            logger.info("Saved and activated new YARA rule: %s", filepath)
            return True, ""
        except FileExistsError:
            return False, f"Rule '{rule_name}' already exists; refusing to overwrite it."
        except Exception as exc:
            logger.error("Failed to persist/activate YARA rule %s: %s", rule_name, exc)
            if os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
            self.reload_rules()
            return False, f"Failed to save or activate rule: {exc}"
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def reload_rules(self):
        if not HAS_YARA:
            return
        
        filepaths = {}
        for entry in os.listdir(self.rules_dir):
            if entry.endswith(".yar"):
                filepaths[entry.replace(".yar", "")] = os.path.join(self.rules_dir, entry)
                
        if not filepaths:
            self.scanner = None
            return
            
        try:
            self.scanner = yara.compile(filepaths=filepaths)
            logger.info(f"Compiled {len(filepaths)} stored YARA rules successfully.")
            for cb in self.reload_callbacks:
                try:
                    cb(self.scanner)
                except Exception as e:
                    logger.error(f"Error executing YARA reload callback: {e}")
        except Exception as e:
            logger.error(f"Failed to compile stored YARA rules: {e}")
            self.scanner = None
