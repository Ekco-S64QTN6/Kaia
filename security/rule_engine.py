import os
import re
import sys
import tempfile
import subprocess
import logging
import threading
import config

logger = logging.getLogger(__name__)

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
        self.benign_corpus_dir = "/tmp/kaia_benign_corpus"
        self.reload_callbacks = []
        
        # Initialize rules directory
        os.makedirs(self.rules_dir, exist_ok=True)
        self._setup_benign_corpus()
        self.reload_rules()

    def register_reload_callback(self, callback):
        self.reload_callbacks.append(callback)

    def _setup_benign_corpus(self):
        """Creates 5 known-clean text files in /tmp/kaia_benign_corpus."""
        os.makedirs(self.benign_corpus_dir, exist_ok=True)
        lorem_ipsum_variants = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.",
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui."
        ]
        for i, text in enumerate(lorem_ipsum_variants):
            filepath = os.path.join(self.benign_corpus_dir, f"benign_{i}.txt")
            try:
                if not os.path.exists(filepath):
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(text)
            except Exception as e:
                logger.error(f"Failed to create benign corpus file {filepath}: {e}")

    def compile_rule_text(self, rule_name: str, author: str, threat_description: str, target_ioc_indicator: str, mitre_framework_id: str = None) -> str:
        # Escape indicator and strip yara metacharacters
        escaped_ioc = re.escape(target_ioc_indicator)
        for char in ['{', '}', '[', ']', '/', '\\']:
            escaped_ioc = escaped_ioc.replace(char, '')
            
        mitre_id = mitre_framework_id or ""
        
        rule_text = f"""rule {rule_name} {{
    meta:
        author = "{author}"
        description = "{threat_description}"
        mitre_id = "{mitre_id}"
    strings:
        $ioc = "{escaped_ioc}"
    condition:
        any of them
}}
"""
        return rule_text

    def validate_rule(self, rule_text: str) -> tuple:
        """
        Validates the YARA rule by running it in an isolated systemd sandbox.
        Returns (is_valid, error_message).
        """
        # 1. Ephemeral tempfile write
        fd, rule_path = tempfile.mkstemp(suffix=".yar", prefix="yara_rule_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(rule_text)
        except Exception as e:
            return False, f"Failed to write temp rule file: {e}"

        # 2. Run systemd-run validation sandbox
        cmd = [
            "systemd-run",
            "--scope",
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
            "yara", rule_path, self.benign_corpus_dir
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            # Cleanup temp rule file
            try:
                os.remove(rule_path)
            except Exception:
                pass
                
            if res.returncode != 0:
                # Compile or runtime error
                err = res.stderr.strip() or res.stdout.strip() or "YARA validation failed"
                if "Failed to find executable" in err or "No such file or directory" in err or "not found" in err:
                    raise FileNotFoundError("yara binary not found in path")
                return False, f"YARA validation syntax error: {err}"
                
            if res.stdout.strip():
                # Matched benign files
                return False, "False positive detected: rule matched benign corpus files."
                
            return True, ""
        except FileNotFoundError:
            # systemd-run or yara not installed
            logger.warning("systemd-run or yara binary not found. Falling back to local validation.")
            # Local validation fallback
            if not HAS_YARA:
                try:
                    os.remove(rule_path)
                except Exception:
                    pass
                return True, "" # Fail-open if yara is missing
            try:
                rules = yara.compile(source=rule_text)
                for entry in os.listdir(self.benign_corpus_dir):
                    filepath = os.path.join(self.benign_corpus_dir, entry)
                    if os.path.isfile(filepath):
                        matches = rules.match(filepath=filepath)
                        if matches:
                            return False, "False positive detected locally on benign corpus."
                try:
                    os.remove(rule_path)
                except Exception:
                    pass
                return True, ""
            except Exception as e:
                try:
                    os.remove(rule_path)
                except Exception:
                    pass
                return False, f"Local YARA compile error: {e}"
        except Exception as e:
            try:
                os.remove(rule_path)
            except Exception:
                pass
            return False, f"Validation system error: {e}"

    def add_rule(self, rule_name: str, author: str, threat_description: str, target_ioc_indicator: str, mitre_framework_id: str = None) -> tuple:
        """Compiles, validates, and stores a new IOC rule."""
        rule_text = self.compile_rule_text(rule_name, author, threat_description, target_ioc_indicator, mitre_framework_id)
        
        ok, err = self.validate_rule(rule_text)
        if not ok:
            return False, err
            
        # Write to stored rules directory
        filename = f"{rule_name}.yar"
        filepath = os.path.join(self.rules_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(rule_text)
            logger.info(f"Saved new YARA rule: {filepath}")
            self.reload_rules()
            return True, ""
        except Exception as e:
            return False, f"Failed to save rule file: {e}"

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
