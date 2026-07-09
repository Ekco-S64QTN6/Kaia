# Kaia — Full Project Review & Coding Agent Prompt
**Date:** 2026-06-28  
**Source of truth:** `docs/master_plan.md`  
**Repo:** `Ekco-S64QTN6/Kaia`  
**Reviewed by:** live `git clone` + grep/cat — every finding below has a confirmed line number or file location  
**Agent role:** implement all fixes in Part A before doing anything in Part B

---

## Overall State

All Tier 3 and Tier 5 structural work is complete and verified. The system has correct dashboard panes, working Policy Gate, eBPF/fanotify/honeypot/YARA/discovery subsystems, systemd units, negative test suite, and tamper detection. Three gaps from the last cycle were not fixed. Five new issues were found in this review pass. All are listed below with exact fix instructions.

---

## What Is Confirmed Complete (do not re-touch)

- AuditLogCollector crash fix, 4-pane security dashboard, Sec filter wired  
- Cognitive wiring fully removed from `main.py` and `security/`  
- Storage paths unified to `storage/security/` per §2.2  
- `schemas.py` import order fixed, `IocRuleRequest` model present  
- Tamper detection (`security/tamper_detection.py`) wired into `policy_gate.py` `__main__`  
- FIM daemon (`security/fim_daemon.py`) with `FAN_CREATE` + `FAN_ATTRIB` + `FAN_MODIFY` + YARA scan  
- eBPF telemetry (`security/ebpf_telemetry.py`) with all 6 hook points + `sys_enter_openat2`  
- Passive L2/L3 discovery (`security/network_discovery.py`) with ARP/mDNS/LLMNR/SSDP/NetBIOS  
- Honeypot coordinator (`security/honeypot.py`) with filesystem decoys + network namespace listeners  
- YARA rule engine (`security/rule_engine.py`) with `add_rule` policy gate action  
- GeoLite2 `lookup_geoip` wired into Pane 2 rendering correctly (country string, not tags)  
- Emergency lockdown script + service files + `install_services.sh` with path parametrization  
- `.git` explicit in `host_executor.py` blocklist + segment-level path check  
- Negative test suite expanded to 14 tests including `.git`, `storage/`, sandbox mask, executor bypass  
- `telemetry_sanitizer.py` wired into `ebpf_telemetry.py` exec/connect/privilege/honeypot callbacks  
- Command panel worker thread liveness fix (`_cmd_stop` event)  
- YARA rules count uses file count not `len(engine.scanner)`  
- All service files use `${KAIA_PROJECT_DIR}` env var, not hardcoded paths  
- `test_tier5_security.py` has proper `sys.path` and `KAIA_CAPABILITY_TOKEN_SECRET` setup  
- `add_rule` in both permission sets and after lattice check in `policy_gate.py`  
- Lattice uses `max()` in both `policy_gate.py` and `ContainmentCollector`

---

## Part A — Fixes Required (in priority order)

Work top to bottom. Do not skip to Part B until all 8 fixes below are complete and the acceptance grep checks at the end of each pass.

---

### Fix 1 — `security/db.py`: WAL mode not set (§8, Appendix D)

**Problem:** master_plan §8 specifies the security event ledger as "append-only SQLite with WAL mode." `initialize_db()` creates the table but never sets WAL mode. Without WAL, concurrent reads from `AuditLogCollector` (250ms polling) while `log_security_event` writes will cause `database is locked` errors that silently swallow events.

**Fix:** Add a WAL pragma immediately after opening the connection in `initialize_db()`:

```python
def initialize_db():
    """Initializes the append-only security events database."""
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")   # ADD THIS LINE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
        ...
    """)
    conn.commit()
    conn.close()
```

**Verify:** `python -c "import sqlite3; import config; conn = sqlite3.connect(config.SECURITY_DB_PATH); print(conn.execute('PRAGMA journal_mode').fetchone())"` prints `('wal',)`.

---

### Fix 2 — `security/host_executor.py`: `sandbox-exec` tier missing INV-011 masking

**Problem:** INV-011 states "any script execution inside Bubblewrap must mask `.env` and `storage/`." The `bwrap`, fallback, and `systemd-nspawn` tiers all include the `--bind /dev/null .env` and `--tmpfs storage/` flags. The `sandbox-exec` tier (lines ~128-139) uses `bwrap` too but is missing both flags, creating a masking gap if that lattice level is ever selected.

**Fix:** In `execute_script`, in the `elif effective_level == "sandbox-exec":` block, add the masking flags before the `--ro-bind script_path` line:

```python
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
        "--bind", "/dev/null", os.path.join(workspace_abs, ".env"),        # ADD
        "--tmpfs", os.path.join(workspace_abs, "storage"),                  # ADD
        "--ro-bind", script_path, "/tmp/run_script.sh",
        "--",
        "/tmp/run_script.sh"
    ]
```

**Verify:** `grep -A30 'sandbox-exec' security/host_executor.py` shows both masking flags present.

---

### Fix 3 — `security/policy_gate.py`: diagnostics capability name mismatch

**Problem:** When a capability token is present for a `diagnostics` request, `policy_gate.py` line 316 verifies it against `"view_logs"` as the required capability. But `generate_capability_token` is called with `action` as the capability name — so any caller that generates a token with `capability="diagnostics"` will have it rejected. Since diagnostics is Green tier (token optional), tests pass without tokens, hiding this mismatch. But it is wrong and will silently reject any operator who does supply a token.

**Fix:** In `evaluate_and_execute`, change the verify call to use `"diagnostics"`:

```python
# line 316 — change "view_logs" to "diagnostics"
ok_tok, err_tok = verify_capability_token(req.capability_token, "diagnostics", req.query_type)
```

**Verify:** Generate a token with `generate_capability_token("diagnostics", "ss")` and pass it in a diagnostics request — it must succeed. A token with `generate_capability_token("view_logs", "ss")` must fail.

---

### Fix 4 — `security/fim_daemon.py` and `security/network_discovery.py`: DB connections not closed in `finally` (Appendix E)

**Problem:** Appendix E requires every database operation to use `try/except/finally` to guarantee connection closure. Both files close connections inside `try` blocks — if an exception occurs between `conn = sqlite3.connect(...)` and `conn.close()`, the connection leaks. Under load (rapid FIM events), this can exhaust SQLite file descriptors.

**Fix for `security/fim_daemon.py`** — the `_handle_event` DB write block (around line 289):

```python
try:
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fim_events (timestamp, event_type, pid, comm, path, yara_matches, sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ts, event_type, pid, comm, filepath, yara_str, sha))
    conn.commit()
except Exception as e:
    logger.error(f"FIM audit database insert failed: {e}")
finally:                           # ADD finally block
    try:
        conn.close()
    except Exception:
        pass
```

Apply the same `finally: conn.close()` pattern to the `get_recent_alerts` method and the `initialize` DB block in `start()`.

**Fix for `security/network_discovery.py`** — apply `finally: conn.close()` pattern to `start()` DB init, `get_recent_assets()`, and `_log_asset()`.

**Verify:** `grep -n "finally" security/fim_daemon.py security/network_discovery.py` — both files show at least 3 `finally` entries each.

---

### Fix 5 — `security/ebpf_telemetry.py`: `_handle_tcp_retransmit` not sanitized (INV-004)

**Problem:** The `_handle_tcp_retransmit` callback (around line 312) appends raw `saddr_str` and `daddr_str` directly to `self.retrans_deque` without calling `sanitize_telemetry`. All other callbacks were fixed last cycle but this one was missed. INV-004 covers all incoming telemetry streams with no exceptions.

**Fix:** In `_handle_tcp_retransmit`:

```python
def _handle_tcp_retransmit(self, cpu, data, size):
    event = self.bpf["tcp_retransmit_events"].event(data)
    saddr_str = socket.inet_ntoa(struct.pack("<I", event.saddr))
    daddr_str = socket.inet_ntoa(struct.pack("<I", event.daddr))
    from security.telemetry_sanitizer import sanitize_telemetry
    raw = {
        "ip": saddr_str, "port": str(event.sport),
    }
    clean_src = sanitize_telemetry(raw)
    raw2 = {"ip": daddr_str, "port": str(event.dport)}
    clean_dst = sanitize_telemetry(raw2)
    with self.lock:
        self.retrans_deque.append({
            "saddr": clean_src.get("ip", ""),
            "daddr": clean_dst.get("ip", ""),
            "sport": int(clean_src.get("port", 0) or 0),
            "dport": int(clean_dst.get("port", 0) or 0),
            "state": event.state,
            "timestamp": time.time()
        })
```

**Verify:** `grep -n "sanitize_telemetry" security/ebpf_telemetry.py` — appears in 5 callback functions (exec, tcp_connect, tcp_retransmit, privilege, honeypot).

---

### Fix 6 — `security/network_discovery.py`: `_log_asset` not sanitized (INV-004)

**Problem:** `_log_asset` receives `ip`, `hostname`, and `vendor` from raw Ethernet frame parsing. These fields come directly from untrusted broadcast packets — exactly the injection vector INV-004 is designed to block. A crafted mDNS hostname containing shell metacharacters or SQL fragments would go directly into `assets.db` unsanitized.

**Fix:** In `_log_asset`, after receiving ip/mac/hostname/vector, apply sanitization before any DB write:

```python
def _log_asset(self, ip: str, mac: str, hostname: str, vector: str):
    now = time.time()
    key = (mac, ip)
    
    # Sanitize all fields from untrusted network input (INV-004)
    from security.telemetry_sanitizer import sanitize_telemetry
    raw = {"ip": ip, "hostname": hostname}
    clean = sanitize_telemetry(raw)
    ip = clean.get("ip", "")
    hostname = clean.get("hostname", "")
    if not ip:   # Drop events with unparseable IPs
        return

    with self.cache_lock:
        # ... rest of method unchanged
```

**Verify:** `grep -n "sanitize_telemetry" security/network_discovery.py` returns at least one match.

---

### Fix 7 — `kaia_dashboard.py` `AuditLogCollector`: audit ledger polled at wrong interval (Appendix D)

**Problem:** Appendix D specifies security_events.db at 250ms and `audit_ledger.json` at 500ms. The current `_run` loop polls both on every 250ms tick.

**Fix:** Add a `ledger_tick` counter to `_run`:

```python
def _run(self) -> None:
    """Poll security_events.db (250ms) and audit_ledger.json (500ms)."""
    ledger_tick = 0
    while not self._stop.is_set():
        try:
            new_entries = []
            new_entries.extend(self._poll_security_db())
            ledger_tick += 1
            if ledger_tick >= 2:
                new_entries.extend(self._poll_audit_ledger())
                ledger_tick = 0
            if new_entries:
                now = time.time()
                with self._lock:
                    if not self._paused:
                        for entry in new_entries:
                            self._logs.append(entry)
                            self._lps_timestamps.append(now)
                            self._check_split_lock(entry.message, entry.timestamp)
        except Exception:
            pass
        self._stop.wait(self.AUDIT_POLL_INTERVAL)
```

**Verify:** `grep -n "ledger_tick" kaia_dashboard.py` returns at least 3 matches.

---

### Fix 8 — `kaia_dashboard.py` `check threat` command: CVE details not enriched (§Appendix C)

**Problem:** `> check threat <IP>` shows CVE IDs from Shodan but never calls `lookup_cve_details()` to surface CVSS scores and descriptions from the local `cve.db`. The data exists — it just isn't fetched.

**Fix:** In `_command_worker`, replace the vuln display block in the `check threat` handler:

```python
if shodan.get("vulns"):
    self._add_response(f"Vulnerabilities ({len(shodan['vulns'])} total):")
    for cve_id in shodan["vulns"][:3]:
        details = threat_intel.lookup_cve_details(cve_id)
        cvss = details.get("cvss", "N/A")
        desc = str(details.get("details", ""))[:80]
        self._add_response(f"  • {cve_id} — CVSS: {cvss} — {desc}")
    if len(shodan["vulns"]) > 3:
        self._add_response(f"  ... and {len(shodan['vulns']) - 3} more")
```

**Verify:** `grep -n "lookup_cve_details" kaia_dashboard.py` returns one match.

---

## Minor Cleanups (low priority, same commit is fine)

**schemas.py dead import:** `IPvAnyAddress` is imported from pydantic but never used in any model. Remove it from the import line. This is cosmetic but keeps the import clean.

```python
# Change:
from pydantic import BaseModel, Field, IPvAnyAddress
# To:
from pydantic import BaseModel, Field
```

**`_handle_privilege_event` uid key mapping:** The uid field is sanitized by mapping it to the `"pid"` schema key (`raw_uid = {"pid": str(event.uid)}`). This works because both are digit-only fields, but it's confusing. Rename the mapping key to make intent clear — add a `"uid"` entry to `FIELD_SCHEMAS` in `telemetry_sanitizer.py` mirroring the `"pid"` definition, then use `"uid"` as the key. This is spec hygiene, not a bug fix, but worth doing cleanly.

---

## Part B — Run Full Test Suite

After all 8 fixes are complete:

```bash
python -m pytest tests/test_negative_security.py -v
python -m pytest tests/test_advanced_security.py -v
python -m pytest tests/test_database_utils.py -v
python -m pytest tests/test_kaia_cli.py -v
python -m pytest tests/test_tier5_security.py -v
python tests/verify_security.py
```

All must pass. If any fail, report the exact error — do not attempt to fix silently.

---

## Part C — Acceptance Grep Checks

Run all of these. Every check must return the stated result before declaring this cycle complete.

```bash
# WAL mode in db.py
grep -n "journal_mode=WAL" security/db.py
# Expected: one match

# sandbox-exec has masking flags
grep -A35 "sandbox-exec" security/host_executor.py | grep "dev/null\|tmpfs"
# Expected: two matches (.env and storage/)

# diagnostics token uses correct capability name
grep -n '"diagnostics"' security/policy_gate.py | grep "verify_capability"
# Expected: one match

# fim_daemon uses finally for DB ops
grep -c "finally" security/fim_daemon.py
# Expected: 3 or more

# network_discovery uses finally for DB ops
grep -c "finally" security/network_discovery.py
# Expected: 3 or more

# tcp_retransmit sanitized
grep -n "sanitize_telemetry" security/ebpf_telemetry.py
# Expected: 5 matches (exec, tcp_connect, tcp_retransmit, privilege, honeypot)

# network_discovery sanitizes before DB write
grep -n "sanitize_telemetry" security/network_discovery.py
# Expected: one match

# ledger_tick counter present
grep -n "ledger_tick" kaia_dashboard.py
# Expected: 3 or more matches

# CVE enrichment wired in check threat
grep -n "lookup_cve_details" kaia_dashboard.py
# Expected: one match

# IPvAnyAddress removed from schemas.py
grep "IPvAnyAddress" security/schemas.py
# Expected: zero matches
```

---

## What Is Deferred — Do Not Implement This Cycle

Per master_plan §12 and Appendix F:
- DuckDB delta computation for InternetDB snapshots (Appendix C)
- `dnsdb/dns.db` passive hostname ingestion pipeline (schema exists, population deferred)
- Supply chain monitoring, baseline drift, secrets scanning (§11 optional)
- GVisor / Firecracker containment tiers
- AuditLogCollector persistent-error → lockdown signaling (Appendix D last paragraph) — architectural change requiring UI thread coordination, deferred to next cycle
- Multi-agent or distributed architectures (§12, out of scope)
