"""Find credentials where Kaia keeps data: its logs, its state, its deployed tree.

The capability-token signing secret is the authorization root (it was once
committed to this public repository). It belongs only in the gate's
environment and ``/etc/kaia/secret.env``. This check looks for:

* the live signing secret itself, byte for byte;
* high-precision credential formats: private key blocks, AWS access key ids,
  GitHub, Slack, Anthropic and OpenAI tokens, and a 64-hex
  ``KAIA_CAPABILITY_TOKEN_SECRET=`` assignment (a stray ``.env`` copy).

Scopes are files Kaia writes or ships: an operator command typed into the HUD,
a traceback that printed the environment, or a backup ``.env`` left in the
deployed tree would all land there. Findings name the file, line and pattern;
**the matched text is never recorded, logged or returned.** Alert-only.

Must stay importable without ``core.config``.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import time
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PATTERNS: Tuple[Tuple[str, "re.Pattern[bytes]"], ...] = (
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----")),
    ("aws_access_key_id", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{60,255})\b")),
    ("slack_token", re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("anthropic_api_key", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_api_key", re.compile(rb"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{40,}")),
    ("kaia_secret_assignment", re.compile(rb"KAIA_CAPABILITY_TOKEN_SECRET\s*=\s*['\"]?[0-9a-fA-F]{64}\b")),
)
LIVE_SECRET = "live_signing_secret"

SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"})
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FILES = 200_000
MAX_FINDINGS = 200
CHUNK = 1024 * 1024
OVERLAP = 4096          # longer than any pattern or secret we look for
MIN_EXACT_LEN = 16      # never search for a short "secret": it would match noise


def scan_file(path: str, exact: Dict[str, bytes]) -> List[Tuple[str, int]]:
    """[(pattern name, 1-based line)] in one regular file, no-follow, size-bounded."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    hits: Dict[str, int] = {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE_BYTES:
            return []
        carry = b""
        lines_before = 0      # newlines in the data preceding `carry`
        while True:
            chunk = os.read(fd, CHUNK)
            if not chunk and not carry:
                break
            data = carry + chunk
            for name, needle_re in _matchers(exact):
                if name in hits:
                    continue
                for m in needle_re.finditer(data):
                    if name != LIVE_SECRET and _is_documentation_example(m.group(0)):
                        continue
                    hits[name] = lines_before + data.count(b"\n", 0, m.start()) + 1
                    break
            if not chunk:
                break
            keep = data[-OVERLAP:]
            lines_before += data.count(b"\n", 0, len(data) - len(keep))
            carry = keep
    finally:
        os.close(fd)
    return sorted(hits.items(), key=lambda kv: kv[1])


# Vendor documentation values (AWS's AKIA...EXAMPLE) and obvious placeholders,
# as used by the honeypot's decoy content. The live secret is never exempt.
_EXAMPLE = re.compile(rb"EXAMPLE|example|XXXXXXXX|xxxxxxxx|0{16}")


def _is_documentation_example(match: bytes) -> bool:
    return bool(_EXAMPLE.search(match))


def _matchers(exact: Dict[str, bytes]):
    yield from PATTERNS
    for name, value in exact.items():
        if len(value) >= MIN_EXACT_LEN:
            yield name, re.compile(re.escape(value))


def iter_files(scopes: Iterable[str], skip_paths: Iterable[str] = ()) -> Iterable[str]:
    """Regular files under each scope, never following symlinks or crossing filesystems."""
    skip = {os.path.realpath(p) for p in skip_paths}
    seen_dirs = set()
    for scope in scopes:
        try:
            root_st = os.lstat(scope)
        except OSError:
            continue
        if stat.S_ISREG(root_st.st_mode):
            if os.path.realpath(scope) not in skip:
                yield scope
            continue
        if not stat.S_ISDIR(root_st.st_mode):
            continue
        stack = [scope]
        while stack:
            directory = stack.pop()
            key = os.path.realpath(directory)
            if key in seen_dirs or key in skip:
                continue
            seen_dirs.add(key)
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode):
                    if entry.name not in SKIP_DIRS and st.st_dev == root_st.st_dev:
                        stack.append(entry.path)
                elif stat.S_ISREG(st.st_mode) and os.path.realpath(entry.path) not in skip:
                    yield entry.path


def scan(scopes: Iterable[str], live_secret: Optional[str] = None, skip_paths: Iterable[str] = ()) -> dict:
    """One pass. Findings carry path, line and pattern name, never the match."""
    started = time.time()
    exact = {}
    if live_secret and len(live_secret) >= MIN_EXACT_LEN:
        exact[LIVE_SECRET] = live_secret.encode("utf-8")
    findings: List[dict] = []
    files = total = 0
    truncated = False
    for path in iter_files(scopes, skip_paths):
        files += 1
        if files > MAX_FILES or total > MAX_TOTAL_BYTES:
            truncated = True
            break
        try:
            total += os.lstat(path).st_size
            hits = scan_file(path, exact)
        except OSError:
            continue
        for name, line in hits:
            if len(findings) >= MAX_FINDINGS:
                truncated = True
                break
            findings.append({"path": path, "line": line, "pattern": name})
    return {"files": files, "bytes": total, "findings": findings, "truncated": truncated,
            "duration": round(time.time() - started, 1)}


EVENT_EXPOSURE = "secret_exposure"


class SecretExposureMonitor:
    """Scan every ``interval`` seconds; record each (path, pattern) once. Alert-only."""

    def __init__(self, scopes, live_secret: Optional[str], skip_paths=(), interval: float = 6 * 3600,
                 first_delay: float = 900, log_event=None, clock=time.time):
        self.scopes = list(scopes)
        self._secret = live_secret
        self.skip_paths = list(skip_paths)
        self.interval = interval
        self.first_delay = first_delay
        self._log_event = log_event
        self._clock = clock
        self._reported = set()
        self._stop = None

    def run_once(self) -> dict:
        from security import control_health
        try:
            report = scan(self.scopes, self._secret, self.skip_paths)
        except Exception as e:
            logger.error("Secret exposure scan failed: %s", e)
            control_health.report("secret_exposure", "failed", f"last scan raised: {e}", heartbeat_interval=60.0)
            return {}
        current = {(f["path"], f["pattern"]): f["line"] for f in report["findings"]}
        log = self._log_event
        if log is None:
            from security.db import log_security_event as log
        for (path, pattern), line in current.items():
            if (path, pattern) in self._reported:
                continue
            logger.critical("Credential exposed: %s pattern in %s at line %d (value not logged)", pattern, path, line)
            try:
                log(event_type=EVENT_EXPOSURE, source="secret_exposure", actor=path[:200],
                    payload_hash=f"{pattern} at line {line}", disposition="observed",
                    session_id="system_protection")
            except Exception as e:
                logger.error("Could not record secret exposure event: %s", e)
        self._reported = set(current)
        state = "degraded" if report["truncated"] else "active"
        detail = (f"{report['files']} files, {report['bytes'] // (1024 * 1024)} MiB scanned; "
                  f"{len(current)} exposures" + ("; scan hit its limits" if report["truncated"] else ""))
        if not self._secret:
            state, detail = "degraded", detail + "; live signing secret unavailable to compare"
        control_health.report("secret_exposure", state, detail, heartbeat_interval=60.0)
        return report

    def _run(self) -> None:
        from security import control_health
        next_run = self._clock() + self.first_delay
        while not self._stop.wait(min(60.0, max(0.0, next_run - self._clock()))):
            control_health.heartbeat("secret_exposure")
            if self._clock() >= next_run:
                self.run_once()
                next_run = self._clock() + self.interval

    def start(self) -> None:
        import threading
        from security import control_health
        self._stop = threading.Event()
        control_health.report("secret_exposure", "active",
                              f"first scan in {int(self.first_delay)} s, then every {int(self.interval / 3600)} h",
                              heartbeat_interval=60.0)
        threading.Thread(target=self._run, daemon=True, name="secret-exposure").start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
