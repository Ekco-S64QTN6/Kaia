import os
import sys
import time
import hashlib
import logging
import threading
import config
from security.db import log_security_event

logger = logging.getLogger(__name__)

class TamperDetector:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._baselines = {}  # filepath -> (hash_val, size_or_type)
        
        # Files whose modification has already been alerted on, so one incident
        # produces one lockdown instead of one every check interval.
        # filepath -> the observed state that fired (hash, or "missing").
        self._fired = {}
        # Timestamps of recent lockdown triggers, for the circuit breaker.
        self._lockdown_times = []

        # 1. Core configs (immutable)
        self.immutable_files = [
            os.path.join(config.WORKSPACE_DIR, ".env"),
            os.path.join(config.WORKSPACE_DIR, "core", "config.py"),
            os.path.join(config.WORKSPACE_DIR, "kaia_dashboard.py"),
            os.path.join(config.WORKSPACE_DIR, "main.py"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh"),
        ]

        # Every module in the security package, discovered rather than listed.
        #
        # An explicit list goes stale the moment a module is added: before this,
        # 8 of the 14 files in security/ were unprotected -- including
        # telemetry_sanitizer.py, which INV-004 depends on, and rule_engine.py.
        # Self-protection matters here too: neutralise tamper_detection.py and
        # nothing else is ever checked again; neutralise db.py and the alert is
        # never recorded.
        security_pkg = os.path.join(config.WORKSPACE_DIR, "security")
        try:
            for name in sorted(os.listdir(security_pkg)):
                if name.endswith(".py"):
                    path = os.path.join(security_pkg, name)
                    if path not in self.immutable_files:
                        self.immutable_files.append(path)
        except OSError as e:
            logger.error(f"Could not enumerate {security_pkg} for integrity watch: {e}")
        
        # Systemd service unit (if exists)
        sysd_service = "/etc/systemd/system/kaia-policy-gate.service"
        if os.path.exists(sysd_service):
            self.immutable_files.append(sysd_service)
            
        # 2. Append-only security files
        self.append_only_files = [
            config.AUDIT_LOG_PATH,
        ]
        
        # 3. SQLite append-only files
        self.sqlite_append_only_files = [
            config.SECURITY_DB_PATH,
        ]

    def _load_trusted_baseline(self) -> dict:
        """Load operator-recorded hashes from a root-owned file, if present.

        Returns {abspath: sha256}. Empty dict when no baseline exists, in which
        case the detector falls back to trusting whatever is on disk at startup
        (and says so loudly).

        Format is `sha256sum`-compatible:  <hex>  <absolute path>
        """
        path = getattr(config, "TRUSTED_BASELINE_PATH", None)
        if not path or not os.path.exists(path):
            return {}
        trusted = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        trusted[os.path.abspath(parts[1].strip())] = parts[0].strip()
        except Exception as e:
            logger.error(f"Could not read trusted baseline {path}: {e}")
            return {}
        logger.info(f"Loaded {len(trusted)} trusted hashes from {path}")
        return trusted

    def _compute_sha256(self, filepath: str, limit: int = None) -> str:
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                if limit is not None:
                    h.update(f.read(limit))
                else:
                    h.update(f.read())
            return h.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {filepath}: {e}")
            return ""

    def _compute_sqlite_prefix_hash(self, db_path: str, row_count: int) -> str:
        """
        Hashes the logical content (not raw bytes) of the first `row_count` rows
        by rowid. Stable across WAL checkpoints, header counter changes, and
        VACUUM, since it only reflects committed row content.
        """
        import sqlite3
        h = hashlib.sha256()
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, timestamp, type, source, actor, payload_hash, disposition, session_id
                FROM security_events
                ORDER BY rowid ASC
                LIMIT ?
            """, (row_count,))
            for row in cursor.fetchall():
                h.update("|".join(str(c) for c in row).encode("utf-8"))
        except Exception as e:
            logger.error(f"Error computing SQLite content hash for {db_path}: {e}")
            return ""
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        return h.hexdigest()

    def _get_sqlite_row_count(self, db_path: str) -> int:
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting rows in {db_path}: {e}")
            return 0
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def start(self):
        """Establish baselines and start audit thread."""
        logger.info("Initializing tamper detection baseline hashes...")
        
        # Startup sanity check: verify critical files exist
        for filepath in self.immutable_files:
            # .env might be missing in some test setups, check others
            if filepath.endswith(".env") and not os.path.exists(filepath):
                continue
            if not os.path.exists(filepath):
                print(f"FATAL: Critical system file missing: {filepath}", file=sys.stderr)
                sys.exit(1)

        if not getattr(config, "TAMPER_ENFORCE", True):
            logger.critical(
                "TAMPER ENFORCEMENT IS DISABLED (KAIA_TAMPER_ENFORCE=0). Modifications "
                "will be detected and logged but will NOT trigger emergency lockdown. "
                "This is a security control running in monitor-only mode."
            )

        # Baseline immutable files against the TRUSTED record where one exists.
        #
        # Hashing whatever is on disk means a file edited while the daemon was
        # stopped silently becomes the new baseline -- and anyone who can write a
        # file can also restart a service, so that was a complete bypass. Where an
        # operator-recorded hash exists we adopt it as the baseline and report the
        # discrepancy immediately.
        trusted = self._load_trusted_baseline()
        if not trusted:
            logger.warning(
                "No trusted integrity baseline at %s -- falling back to hashing the "
                "current on-disk state. Modifications made while this daemon was "
                "stopped CANNOT be detected. Create one with: sudo ./scripts/kaia-baseline.sh",
                getattr(config, "TRUSTED_BASELINE_PATH", "<unset>"),
            )

        for filepath in self.immutable_files:
            if os.path.exists(filepath):
                h = self._compute_sha256(filepath)
                expected = trusted.get(os.path.abspath(filepath))
                if expected and expected != h:
                    self._trigger_tamper_alert(
                        filepath,
                        "Startup hash does not match the trusted baseline "
                        "(file was modified while the daemon was stopped)",
                        h,
                    )
                # Adopt the trusted hash so ongoing checks measure against the
                # recorded state, not the possibly-tampered current one.
                self._baselines[filepath] = (expected or h, "immutable")

        # Baseline append-only files (using prefix-hash technique to avoid false alarms on normal appends)
        for filepath in self.append_only_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                h = self._compute_sha256(filepath, limit=size)
                self._baselines[filepath] = (h, size)

        # Baseline SQLite append-only files (using logical content hashing to avoid false alarms on WAL/header updates)
        for filepath in self.sqlite_append_only_files:
            if os.path.exists(filepath):
                row_count = self._get_sqlite_row_count(filepath)
                h = self._compute_sqlite_prefix_hash(filepath, row_count)
                self._baselines[filepath] = (h, ("sqlite", row_count))

        self._thread = threading.Thread(target=self._run, daemon=True, name="tamper-detector")
        self._thread.start()
        logger.info("Tamper detection thread started successfully.")

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.wait(30.0):
            self.check_integrity()

    def check_integrity(self):
        for filepath, (baseline_hash, file_type) in list(self._baselines.items()):
            if not os.path.exists(filepath):
                self._trigger_tamper_alert(filepath, "File deleted / missing", "missing")
                continue

            if file_type == "immutable":
                current_hash = self._compute_sha256(filepath)
                reason = "Baseline hash mismatch"
            elif isinstance(file_type, tuple) and len(file_type) == 2 and file_type[0] == "sqlite":
                current_hash = self._compute_sqlite_prefix_hash(filepath, file_type[1])
                reason = "SQLite historical row content mismatch (rows altered or deleted)"
            else:
                # Append-only: read up to the original size
                current_hash = self._compute_sha256(filepath, limit=file_type)
                reason = "Historical prefix hash mismatch (file altered or truncated)"

            if current_hash != baseline_hash:
                self._trigger_tamper_alert(filepath, reason, current_hash)
            elif filepath in self._fired:
                # Back to baseline: re-arm so a later, distinct modification
                # alerts again instead of being swallowed by the latch.
                self._fired.pop(filepath, None)
                logger.warning(
                    f"Tamper state cleared, file matches baseline again: {filepath}. "
                    "Detector re-armed for this path."
                )

    def _trigger_tamper_alert(self, filepath: str, reason: str, observed_state: str = ""):
        # Latch. The detector re-checks every 30s and the hash does not return to
        # baseline on its own, so without this a single modification triggers a
        # fresh lockdown forever -- re-flushing nftables every interval while an
        # operator is trying to investigate. Alert loudly once per distinct file
        # state; keep observing quietly after that.
        if self._fired.get(filepath) == observed_state:
            logger.warning(
                f"Tamper still present (already alerted, lockdown not re-triggered): "
                f"{filepath} | Reason: {reason}"
            )
            return
        self._fired[filepath] = observed_state

        msg = f"CRITICAL TAMPER DETECTED: {filepath} | Reason: {reason}"
        print(msg, file=sys.stderr)
        logger.critical(msg)
        
        # Log critical event to security DB
        try:
            log_security_event(
                event_type="tamper_detected",
                source="tamper_detector",
                actor=filepath,
                payload_hash=self._compute_sha256(filepath)[:32] if os.path.exists(filepath) else "deleted",
                disposition="blocked",
                session_id="system_protection"
            )
        except Exception as e:
            logger.error(f"Failed to log tamper event: {e}")

        # Monitor-only mode: detect and record, but do not respond.
        if not getattr(config, "TAMPER_ENFORCE", True):
            logger.critical(
                f"Lockdown SUPPRESSED (monitor-only mode) for tamper on: {filepath}"
            )
            return

        # Trigger emergency lockdown for files that gate or grant authority.
        # Previously a substring test that missed .env entirely -- the file that
        # holds KAIA_CAPABILITY_TOKEN_SECRET, i.e. the ability to mint any
        # capability token. Tampering with it is the most severe case, not an
        # exempt one. Matched on resolved paths rather than substrings.
        lockdown_paths = {
            os.path.join(config.WORKSPACE_DIR, ".env"),
            os.path.join(config.WORKSPACE_DIR, "core", "config.py"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh"),
            "/etc/systemd/system/kaia-policy-gate.service",
        }
        in_security_pkg = os.path.dirname(os.path.abspath(filepath)) == os.path.join(
            os.path.abspath(config.WORKSPACE_DIR), "security"
        )
        if in_security_pkg or os.path.abspath(filepath) in {os.path.abspath(p) for p in lockdown_paths}:
            # Circuit breaker. Latching stops one unchanged file re-firing, but a
            # sequence of changed files could still produce a burst -- and every
            # lockdown flushes the ruleset, so a burst leaves an operator with no
            # network to investigate through. Cap the rate; keep alerting after.
            now = time.time()
            window = getattr(config, "LOCKDOWN_WINDOW_SECONDS", 3600)
            limit = getattr(config, "LOCKDOWN_MAX_PER_WINDOW", 3)
            self._lockdown_times = [t for t in self._lockdown_times if now - t < window]
            if len(self._lockdown_times) >= limit:
                logger.critical(
                    f"LOCKDOWN RATE LIMIT REACHED ({limit} in {window}s). Tamper on "
                    f"{filepath} recorded but lockdown SUPPRESSED so the host stays "
                    f"reachable for investigation. Review immediately."
                )
                return
            self._lockdown_times.append(now)

            try:
                from security.policy_gate import trigger_lockdown
                trigger_lockdown(f"Tamper detected on core file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to trigger lockdown for tamper: {e}")
