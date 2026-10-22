import os
import json
import sys
import time
import hashlib
import logging
import threading
import re
import stat
from pathlib import Path
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

        # Protect code that can make, enforce, or reset a security decision.
        workspace = os.path.abspath(config.WORKSPACE_DIR)
        self.immutable_files = [
            os.path.join(workspace, ".env"),
            os.path.join(workspace, "core", "config.py"),
            os.path.join(workspace, "kaia_dashboard.py"),
            os.path.join(workspace, "main.py"),
            os.path.join(workspace, "scripts", "kaia-lockdown.sh"),
            os.path.join(workspace, "scripts", "kaia-baseline.sh"),
            os.path.join(workspace, "scripts", "install_services.sh"),
            os.path.join(workspace, "scripts", "kaia-unlock.sh"),
            os.path.join(workspace, "scripts", "kaia-rotate-ledger.sh"),
        ]

        # Automatically cover new security modules instead of maintaining a
        # brittle hand-written list.
        security_pkg = os.path.join(workspace, "security")
        try:
            for name in sorted(os.listdir(security_pkg)):
                if name.endswith(".py"):
                    path = os.path.join(security_pkg, name)
                    if path not in self.immutable_files:
                        self.immutable_files.append(path)
        except OSError as e:
            raise RuntimeError(
                f"Could not enumerate protected security modules in {security_pkg}; "
                "refusing to continue integrity monitoring."
            ) from e

        # The installed service loads its signing key from /etc/kaia/secret.env;
        # /opt/kaia/.env is deliberately removed by the installer. Watch the
        # actual production secret and all installed Kaia unit files.
        for path in (
            "/etc/kaia/secret.env",
            "/etc/systemd/system/kaia-policy-gate.service",
            "/etc/systemd/system/kaia-lockdown.service",
            "/etc/systemd/system/kaia-restore-blocks.service",
        ):
            if os.path.exists(path) and path not in self.immutable_files:
                self.immutable_files.append(path)
        self.immutable_files = list(dict.fromkeys(self.immutable_files))

        # 2. Append-only security files
        self.append_only_files = [
            config.AUDIT_LOG_PATH,
        ]
        
        # 3. SQLite append-only files
        self.sqlite_append_only_files = [
            config.SECURITY_DB_PATH,
        ]

    def _load_trusted_baseline(self) -> dict:
        """Load the operator-approved file hashes, failing closed on corruption.

        A missing baseline is an explicit bootstrap condition handled by start().
        An existing but unreadable or malformed baseline must never be converted
        into that weaker state, or an attacker could bypass trusted verification
        by corrupting the trust record.
        """
        path = getattr(config, "TRUSTED_BASELINE_PATH", None)
        if not path or not os.path.exists(path):
            return {}

        trusted = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) != 2:
                        raise ValueError(f"malformed entry at line {line_number}")
                    digest, filepath = parts
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                        raise ValueError(f"invalid SHA-256 digest at line {line_number}")
                    filepath = filepath.lstrip("*").strip()
                    if not filepath or not os.path.isabs(filepath):
                        raise ValueError(f"expected an absolute path at line {line_number}")
                    normalized = os.path.abspath(filepath)
                    if normalized in trusted:
                        raise ValueError(f"duplicate path at line {line_number}")
                    trusted[normalized] = digest.lower()
        except Exception as e:
            raise RuntimeError(
                f"Cannot trust integrity baseline {path}; refusing to start: {e}"
            ) from e

        if not trusted:
            raise RuntimeError(
                f"Integrity baseline {path} exists but contains no hashes; refusing to start."
            )
        logger.info(f"Loaded {len(trusted)} trusted hashes from {path}")
        return trusted

    def _compute_sha256(self, filepath: str, limit: int = None) -> str:
        """Hash a regular file via a no-follow descriptor with bounded memory."""
        if limit is not None and limit < 0:
            logger.error("Invalid negative SHA-256 byte limit %r for %s", limit, filepath)
            return ""

        digest = hashlib.sha256()
        chunk_size = 1024 * 1024
        fd = None
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            fd = os.open(filepath, flags)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError(f"protected path is not a regular file: {filepath}")

            if limit is not None:
                remaining = limit
                while remaining > 0:
                    chunk = os.read(fd, min(chunk_size, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    remaining -= len(chunk)
            else:
                while True:
                    chunk = os.read(fd, chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception as e:
            logger.error("Error computing SHA-256 for %s: %s", filepath, e)
            return ""
        finally:
            if fd is not None:
                os.close(fd)


    def _compute_sqlite_prefix_hash(self, db_path: str, row_count: int) -> str:
        """Hash a logical prefix of the SQLite ledger with unambiguous fields."""
        import sqlite3

        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            logger.error("Invalid SQLite baseline row count %r for %s", row_count, db_path)
            return ""

        digest = hashlib.sha256()
        conn = None
        try:
            # The detector only observes the ledger; it must not create or mutate
            # the database as a side effect of checking integrity.
            uri = Path(db_path).resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, timestamp, type, source, actor, payload_hash, disposition, session_id
                FROM security_events
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (row_count,),
            )
            for row in cursor:
                encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        except Exception as e:
            logger.error(f"Error computing SQLite content hash for {db_path}: {e}")
            return ""
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        return digest.hexdigest()

    def _get_sqlite_row_count(self, db_path: str) -> int:
        """Return row count, or -1 if the ledger cannot be read."""
        import sqlite3

        conn = None
        try:
            uri = Path(db_path).resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events")
            row = cursor.fetchone()
            return row[0] if row and isinstance(row[0], int) and not isinstance(row[0], bool) else -1
        except Exception as e:
            logger.error(f"Error counting rows in {db_path}: {e}")
            return -1
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def start(self):
        """Establish trusted baselines; do not start monitoring after a mismatch."""
        logger.info("Initializing tamper detection baseline hashes...")

        # Read the trust record before checking file presence so deletion of a
        # previously baselined critical file is itself recorded as tampering,
        # rather than merely aborting startup without an alert.
        trusted = self._load_trusted_baseline()
        baseline_path = getattr(config, "TRUSTED_BASELINE_PATH", None)
        if not trusted and getattr(config, "STATE_DIR_IS_SYSTEM", False):
            raise RuntimeError(
                "Trusted integrity baseline is missing or unavailable in system mode "
                f"({baseline_path}); refusing to start. Review the deployed files, "
                "create the baseline with scripts/kaia-baseline.sh, then start the Policy Gate."
            )

        for filepath in self.immutable_files:
            # The installer deliberately removes /opt/kaia/.env. In production,
            # the actual signing secret comes from /etc/kaia/secret.env.
            if filepath.endswith(".env") and not os.path.exists(filepath):
                continue
            if not os.path.exists(filepath):
                if os.path.abspath(filepath) in trusted:
                    self._trigger_tamper_alert(
                        filepath, "Trusted protected file is missing at startup", "missing"
                    )
                raise RuntimeError(f"FATAL: Critical system file missing: {filepath}")

        if not getattr(config, "TAMPER_ENFORCE", True):
            logger.critical(
                "TAMPER ENFORCEMENT IS DISABLED (KAIA_TAMPER_ENFORCE=0). "
                "Modifications will be detected and logged but will NOT trigger "
                "emergency lockdown. This is monitor-only mode."
            )

        if not trusted:
            # Development-only bootstrap. System mode refuses startup above because
            # reconstructing trust from the current disk would erase evidence of
            # edits made while the privileged service was stopped.
            logger.warning(
                "No trusted integrity baseline at %s -- development bootstrap only: "
                "modifications made while stopped cannot be detected.",
                baseline_path,
            )
        else:
            current_paths = {
                os.path.abspath(path) for path in self.immutable_files if os.path.exists(path)
            }
            missing_hashes = sorted(current_paths - set(trusted))
            stale_hashes = sorted(set(trusted) - current_paths)
            if missing_hashes or stale_hashes:
                details = []
                if missing_hashes:
                    details.append("unbaselined protected files: " + ", ".join(missing_hashes[:8]))
                if stale_hashes:
                    details.append("stale or missing protected paths: " + ", ".join(stale_hashes[:8]))
                message = "Trusted integrity baseline coverage mismatch; refusing to start: " + "; ".join(details)
                logger.critical(message)
                for stale in stale_hashes:
                    if not os.path.exists(stale):
                        self._trigger_tamper_alert(stale, "Trusted protected file missing", "missing")
                raise RuntimeError(message)

        mismatches = []
        for filepath in self.immutable_files:
            if not os.path.exists(filepath):
                continue
            current_hash = self._compute_sha256(filepath)
            if not current_hash:
                raise RuntimeError(f"Cannot hash protected file; refusing to start: {filepath}")
            expected = trusted.get(os.path.abspath(filepath))
            if expected and expected != current_hash:
                self._trigger_tamper_alert(
                    filepath,
                    "Startup hash does not match the trusted baseline "
                    "(file was modified while the daemon was stopped)",
                    current_hash,
                )
                mismatches.append(filepath)
            self._baselines[filepath] = (expected or current_hash, "immutable")

        if mismatches:
            raise RuntimeError(
                "Protected files do not match trusted baseline; startup aborted: "
                + ", ".join(mismatches[:8])
            )

        # Create the JSONL ledger before hashing it so its first record does not
        # evade integrity monitoring when the file was absent at daemon startup.
        for filepath in self.append_only_files:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                fd = os.open(filepath, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
                os.close(fd)
                size = os.path.getsize(filepath)
                digest = self._compute_sha256(filepath, limit=size)
                if not digest:
                    raise RuntimeError("empty or unreadable prefix hash")
                self._baselines[filepath] = (digest, size)
            except Exception as e:
                raise RuntimeError(f"Cannot baseline append-only ledger {filepath}: {e}") from e

        for filepath in self.sqlite_append_only_files:
            if not os.path.exists(filepath):
                continue
            row_count = self._get_sqlite_row_count(filepath)
            if row_count < 0:
                raise RuntimeError(f"Cannot read SQLite audit ledger; refusing to start: {filepath}")
            digest = self._compute_sqlite_prefix_hash(filepath, row_count)
            if not digest:
                raise RuntimeError(f"Cannot hash SQLite audit ledger; refusing to start: {filepath}")
            self._baselines[filepath] = (digest, ("sqlite", row_count))

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
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-baseline.sh"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "install_services.sh"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-unlock.sh"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-rotate-ledger.sh"),
            "/etc/kaia/secret.env",
            "/etc/systemd/system/kaia-policy-gate.service",
            "/etc/systemd/system/kaia-lockdown.service",
            "/etc/systemd/system/kaia-restore-blocks.service",
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
