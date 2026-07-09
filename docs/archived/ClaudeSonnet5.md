Good catch bringing this here — this is a real bug, and it's one the previous review passes (including Gemini 3.5's) missed entirely. Here's the root-cause breakdown.

## Bug 1: The tamper alert is a guaranteed false positive, not a real intrusion

`TamperDetector` treats `security_events.db` as an "append-only" file and verifies it with a **prefix hash**: at startup it hashes the first N bytes (N = file size at boot), and on every 30s check it re-hashes those same first N bytes and compares.

That technique works fine for `audit_ledger.json` (plain JSONL — appends genuinely only add bytes after the existing content). It does **not** work for `security_events.db`, because that's a SQLite file, not a flat log. SQLite's fixed 100-byte file header (which lives at offset 0, always inside your hashed "prefix") contains fields that mutate on **every single write transaction**, regardless of WAL mode:

- the file change counter (offset 24)
- the schema cookie
- the freelist page count/pointer
- the "version-valid-for" number

On top of that, since `security/db.py` sets `PRAGMA journal_mode=WAL`, committed transactions get checkpointed back into the main `.db` file periodically, which can physically rewrite/reorganize pages that fall within your original byte range even though no row you're tracking actually changed.

Net effect: the very next `log_security_event()` call after `TamperDetector.start()` establishes its baseline will alter bytes inside the hashed region. This isn't an edge case — it's deterministic. The tamper detector is guaranteed to fire on its own database, every time, shortly after startup. That's exactly what you're seeing at 07:31:43: no intrusion, just the detector eating its own tail.

**Fix direction:** stop hashing raw file bytes for the SQLite target. Instead, baseline should be `(row_count, hash of concatenated logical content of the first row_count rows via SQL SELECT ... ORDER BY rowid)`. On each check, re-query the *same first N rows* and re-hash — this is invariant to WAL checkpoints, page splits, and header counters, and will only fire if a previously-committed row's actual content changes or disappears. `audit_ledger.json` can keep the current byte-prefix approach since it's genuinely append-only at the OS level. If you want, I can write this as a proper patch to `tamper_detection.py`.

## Bug 2: The "mitigation" for the false alarm also fails, cascading into log noise

Once the false tamper alert fires, `trigger_lockdown()` tries two things, and both fail:

1. `systemctl start kaia-lockdown.service` → *"Access denied... requires interactive authentication"*
2. Fallback: run `scripts/kaia-lockdown.sh` directly → script explicitly checks `EUID -ne 0` and refuses

Both failures trace to the same misconfiguration: `scripts/kaia-policy-gate.service` runs as `User=ekco` / `Group=kaiacord`, but master_plan Appendix A and §3.4 explicitly require `User=root` for this daemon — precisely so it can invoke `systemctl start` and root-level scripts without hitting polkit or EUID checks. The deployed unit file drifted from spec at some point in the agent cycles and none of the prior reviews (including Gemini's) caught it, because none of them actually exercised the lockdown path under a non-root service account.

Practically, this means: **your lockdown mechanism does not currently work at all.** The good news is that in this instance nothing was actually locked down (nftables was never flushed, since both paths failed before reaching that point) — so there's no accidental self-inflicted network outage. The bad news is that if a *real* tamper event or honeypot trigger occurs, the emergency response will silently no-op the same way.

**Fix direction:** revert `User=root` (and drop `AmbientCapabilities=` in favor of running as root, since `CAP_SYS_ADMIN` alone doesn't grant polkit authorization for `systemctl start`, and doesn't satisfy the script's `EUID` check either way). This also matches why the service needs `CAP_NET_RAW`/`CAP_SYS_ADMIN` for eBPF and raw sockets in the first place — running as root sidesteps both problems at once.

## Immediate remediation, no code changes needed

Given lockdown didn't actually engage, you don't need to run `nft flush ruleset` recovery. To stop the noise right now without touching code:
- `sudo systemctl stop kaia-policy-gate.service` (or restart it — `TamperDetector` will just re-baseline and eventually re-trigger on the next write, since the underlying bug is still there)
- Or temporarily comment out the `TamperDetector.start()` call in `policy_gate.py`'s `__main__` until the SQL-based rehash fix lands.

Want me to write the actual patch — the SQL-content-hash rewrite for `tamper_detection.py` plus the `User=root` correction for `kaia-policy-gate.service` — as the next action-plan doc for the coding agent?

user: yes.

# Kaia — Tamper Detection False Positive & Lockdown Escalation Fix
**Date:** 2026-07-05
**Source of truth:** `docs/master_plan.md`
**Repo:** `Ekco-S64QTN6/Kaia`
**Reported by:** Ekco, via live syslog output from a running `kaia-policy-gate` instance (not caught by any prior review pass, including `docs/Gemini35.md`)
**Agent role:** implement both fixes below; do not touch anything else this cycle

---

## Context

Live logs show `TamperDetector` firing a CRITICAL tamper alert against `storage/security/security_events.db` shortly after the Policy Gate daemon starts, with reason `"Historical prefix hash mismatch"`. This then triggers `trigger_lockdown()`, which fails on both its systemd path and its fallback shell-script path. No file has actually been tampered with. This is two separate bugs, both must be fixed.

---

## Bug 1 — `security/tamper_detection.py`: prefix-hash technique is fundamentally wrong for a live SQLite database

**Problem:** `TamperDetector` baselines `security_events.db` (an append-only file per §1.2 INV-003) using the same technique it correctly uses for `audit_ledger.json`: hash the first `size_at_baseline` bytes, and on every 30-second check, re-hash that same byte range and compare.

This works for `audit_ledger.json` because it's a flat JSONL file — appends only ever add bytes after existing content, so the same prefix bytes are stable forever.

It does **not** work for `security_events.db`, because that file is a live SQLite database opened in WAL mode (`PRAGMA journal_mode=WAL`, set in `security/db.py`). The first 100 bytes of any SQLite file are the file header, which contains fields that mutate on **every write transaction**, independent of which rows are touched:

- the file change counter (header offset 24)
- the schema cookie (offset 40, on schema-affecting operations)
- the freelist trunk page pointer and freelist page count
- the "version-valid-for" number tied to the WAL sequence

In addition, WAL checkpoints (which happen automatically as `-wal` file segments fill and get folded back into the main `.db` file) can physically move or rewrite pages that fall within whatever byte range was hashed at baseline time, even when no logical row content in that range has changed.

Net effect: any `log_security_event()` call after `TamperDetector.start()` establishes its baseline is very likely to change the prefix-hashed bytes of `security_events.db`. This is deterministic, not intermittent — hence the CRITICAL alert firing almost immediately on every daemon start.

**Fix:** Replace the byte-prefix hash technique for `security_events.db` specifically with a content-based hash over a fixed set of logical rows. This is invariant to SQLite header churn, WAL checkpoints, and VACUUM/page reorganization, and will only fire on genuine tampering (a row in the hashed range being altered or deleted).

In `security/tamper_detection.py`:

1. Remove `config.SECURITY_DB_PATH` from `self.append_only_files` (the byte-prefix list). Keep `config.AUDIT_LOG_PATH` in that list — the JSONL technique is correct for it and must not change.

2. Add a new baseline category specifically for the SQLite ledger:

```python
# In __init__, alongside self.append_only_files:
self.sqlite_append_only_files = [
    config.SECURITY_DB_PATH,
]
```

3. Add a helper to compute a content hash over the first N committed rows by `rowid`:

```python
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
```

4. In `start()`, baseline the SQLite file separately from the byte-prefix files:

```python
# Alongside the existing append_only_files baselining loop:
for filepath in self.sqlite_append_only_files:
    if os.path.exists(filepath):
        row_count = self._get_sqlite_row_count(filepath)
        h = self._compute_sqlite_prefix_hash(filepath, row_count)
        self._baselines[filepath] = (h, ("sqlite", row_count))
```

Note the file_type tuple now distinguishes `("sqlite", row_count)` from the existing plain-integer byte-size type used for `audit_ledger.json`, so `check_integrity()` must branch on this.

5. Update `check_integrity()` to handle the new type:

```python
def check_integrity(self):
    for filepath, (baseline_hash, file_type) in list(self._baselines.items()):
        if not os.path.exists(filepath):
            self._trigger_tamper_alert(filepath, "File deleted / missing")
            continue

        if file_type == "immutable":
            current_hash = self._compute_sha256(filepath)
            if current_hash != baseline_hash:
                self._trigger_tamper_alert(filepath, "Baseline hash mismatch")

        elif isinstance(file_type, tuple) and file_type[0] == "sqlite":
            row_count = file_type[1]
            current_hash = self._compute_sqlite_prefix_hash(filepath, row_count)
            if current_hash != baseline_hash:
                self._trigger_tamper_alert(filepath, "SQLite historical row content mismatch (rows altered or deleted)")

        else:
            # Existing byte-prefix technique, unchanged — used for audit_ledger.json
            size_limit = file_type
            current_hash = self._compute_sha256(filepath, limit=size_limit)
            if current_hash != baseline_hash:
                self._trigger_tamper_alert(filepath, "Historical prefix hash mismatch (file altered or truncated)")
```

**Verify:**
- Start the Policy Gate, let it run for 2+ minutes while normal traffic writes new rows to `security_events.db` (e.g. run `tests/verify_security.py` in another terminal). No `tamper_detected` event should appear in the logs or in `security_events.db` itself as a result of ordinary appends.
- Manually run `UPDATE security_events SET disposition = 'approved' WHERE rowid = 1;` against the DB while the daemon is running (simulating real tampering of a historical row) — a `tamper_detected` CRITICAL alert must fire within 30 seconds.
- `grep -n "sqlite_append_only_files\|_compute_sqlite_prefix_hash" security/tamper_detection.py` returns matches.

---

## Bug 2 — `scripts/kaia-policy-gate.service`: wrong service user breaks the lockdown escalation path

**Problem:** The deployed unit file runs the Policy Gate as `User=ekco` / `Group=kaiacord`, with `AmbientCapabilities=CAP_NET_RAW CAP_SYS_ADMIN`. But master_plan Appendix A and §3.4 explicitly specify `User=root`. This isn't cosmetic — capabilities alone do not satisfy two independent checks in the escalation path that both require actual root:

1. `subprocess.run(["/usr/bin/systemctl", "start", "kaia-lockdown.service"])` in `trigger_lockdown()` fails with *"Access denied... requires interactive authentication"* because a non-root, non-logind-session caller triggers a polkit prompt that a headless daemon can never answer.
2. The fallback path, `/bin/bash scripts/kaia-lockdown.sh`, explicitly checks `if [ "$EUID" -ne 0 ]` and refuses to run for the same reason — it was written assuming root execution, consistent with the spec.

Both failures observed in the live logs trace to this single misconfiguration. The practical impact: **the emergency lockdown mechanism currently cannot execute under any trigger path** — tamper detection, honeypot triggers, or FIM YARA matches on core files all call the same `trigger_lockdown()` function and will all silently no-op the same way.

**Fix:** In `scripts/kaia-policy-gate.service`, change:

```ini
User=ekco
Group=kaiacord
```

to:

```ini
User=root
Group=kaiacord
```

Leave `AmbientCapabilities=CAP_NET_RAW CAP_SYS_ADMIN` and `CapabilityBoundingSet` in place — they remain relevant for eBPF/fanotify initialization even under root in case `NoNewPrivileges`/capability-drop tooling is added later, but the immediate fix is the `User=` line. Also update `scripts/install_services.sh` if it does any user-specific templating (grep first to confirm — as of the current version it does not, it only substitutes `KAIA_PROJECT_DIR`, so no change needed there).

**Verify:**
- `grep -n "^User=" scripts/kaia-policy-gate.service` returns `User=root`.
- After reinstalling the service (`sudo scripts/install_services.sh`) and restarting it, manually invoke `python3 -c "from security.policy_gate import trigger_lockdown; trigger_lockdown('manual_test')"` from within a root-owned shell context matching the service's environment, or trigger it via the dashboard's `> lockdown` command — confirm `nft list ruleset` shows drop policies on all three chains afterward, and no `"Access denied"` or `"must be executed as root"` errors appear in the journal.
- `systemctl status kaia-policy-gate` shows the process running as `root` (check the `Main PID` owner via `ps -o user= -p <pid>`).

---

## Part A — Run after both fixes

```bash
sudo systemctl restart kaia-policy-gate.service
python -m pytest tests/test_negative_security.py -v
python -m pytest tests/test_advanced_security.py -v
python -m pytest tests/test_tier5_security.py -v
python tests/verify_security.py
```

Let the daemon run idle for at least 5 minutes after restart and confirm no spurious `tamper_detected` events appear in `security_events.db`:

```bash
sqlite3 storage/security/security_events.db "SELECT COUNT(*) FROM security_events WHERE type='tamper_detected' AND timestamp >= datetime('now', '-5 minutes');"
```
Expected: `0`, assuming no actual file was modified during that window.

---

## Part B — Acceptance grep checks

```bash
# SQLite content-hash path exists and is wired for security_events.db only
grep -n "sqlite_append_only_files" security/tamper_detection.py
# Expected: at least 2 matches (init, start())

# audit_ledger.json still uses the byte-prefix technique unchanged
grep -n "append_only_files" security/tamper_detection.py
# Expected: still present, now containing only AUDIT_LOG_PATH

# Service runs as root
grep -n "^User=" scripts/kaia-policy-gate.service
# Expected: User=root
```

---

## What is NOT in scope this cycle

- No changes to `security/policy_gate.py`'s `trigger_lockdown()` logic itself — the function is correct, it was only ever failing due to the caller's privilege level.
- No changes to `kaia-lockdown.sh` or `kaia-lockdown.service` — both are correct as written and assume root, which Bug 2's fix now satisfies.
- No changes to WAL mode configuration in `security/db.py` — WAL is correct and required; the bug was in how tamper detection interpreted a WAL-mode file, not in using WAL itself.
- Do not extend the SQLite content-hash technique to any other file — `security_events.db` is the only current SQLite target in `self.immutable_files`/`append_only_files`. If a future file is added to the monitored perimeter that is also a live SQLite database, it must use the same content-hash pattern, not the byte-prefix one.
