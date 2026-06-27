# Kaia Coding Agent — Tier 3/5 Correction & Next Action Plan
**Date:** 2026-06-27  
**Source of truth:** `docs/master_plan.md`  
**Repo:** `Ekco-S64QTN6/Kaia`  
**Agent role:** implement corrections and remaining items; Claude reviews next cycle

---

## Context

The previous agent pass completed the bulk of Tier 3 and Tier 5 implementation. A joint review by two agents (one Claude, one Gemini 3.5) verified that all major structural work landed correctly: `AuditLogCollector` crash fix, SQL schema correction, four security panes, Sec filter, cognitive wiring removal, storage path unification, schemas import fix, `IocRuleRequest`, tamper detection, FIM daemon, eBPF telemetry, network discovery, honeypot coordinator, YARA rule engine, lockdown scripts, and both negative test suites all exist and are broadly correct.

**Seven confirmed gaps remain from that pass.** Fix all seven before doing anything else. They are listed below in priority order — gaps 1 and 5 are security invariant violations that must be resolved before the system can be considered spec-compliant.

---

## Part A — Gap Fixes (correct before anything else)

Work through these in order. Each entry specifies the file, the exact problem, and the required fix.

---

### Gap 1 — `security/host_executor.py`: `.git` missing from write_file blocklist (INV-010 violation)

**Problem:** INV-010 and §3.3 of master_plan explicitly list `.git` as a blocked hidden file target alongside `.env`. The current `execute_state_modification` method only checks `filename.startswith(".")` for hidden files generally, which does catch `.git`, BUT it only checks the final path component, not intermediate path segments. A path like `legitimate.txt` inside a symlinked directory pointing to `.git/` would bypass this. More critically, the blocklist comment in the code and the invariant table both call `.git` out by name — it should be explicit, not implicit.

**Fix:** In `security/host_executor.py` `execute_state_modification`, add `.git` as an explicit check alongside `.env` in the hidden file block:

```python
BLOCKED_HIDDEN = {".env", ".git"}
if filename.startswith("."):
    if filename in BLOCKED_HIDDEN:
        return False, "", f"Path modification violation: writing to protected file '{filename}' is blocked."
    return False, "", f"Path modification violation: writing to hidden files ({filename}) is blocked."
```

Also add a path-segment check to catch traversal into `.git/` subdirectories:
```python
# Block any path containing .git as a directory component
if ".git" in parts:
    return False, "", "Path modification violation: writing into .git directory tree is blocked."
```

Add this check before the `BLOCKED_DIRS` check. The `.git` segment check must inspect all `parts`, not just `parts[0]`.

**Verify:** `HostExecutor.execute_state_modification(".git/config", "content")` returns `(False, "", "...violation...")`. `HostExecutor.execute_state_modification("subdir/.git/config", "content")` also returns denied.

---

### Gap 2 — `tests/test_negative_security.py`: four §10.4-required test cases missing

**Problem:** §10.4 of master_plan specifies the following test scenarios that are not in the current suite:
1. Writing to `.git` — not tested (gap 1 above exposes this)
2. Writing to `storage/anything.txt` — not tested (storage/ is in `BLOCKED_DIRS` but has no test)
3. Sandbox escape attempts — trying to access masked paths (`.env`, `storage/`) from inside Bubblewrap — not tested at all
4. Automated-bypass attempt — sending a valid action directly to `HostExecutor` without going through `PolicyGate` — not tested

**Fix:** Add four new test functions to `tests/test_negative_security.py`:

**Test: .git write blocked**
```python
def test_blocklisted_git(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, ".git", "config")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "malicious",
        "justification": "Testing .git blocklist",
        "capability_token": token,
        "session_id": "test_neg_git"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    row = check_db_blocked()
    assert row is not None and row[1] == "blocked"
```

**Test: storage/ write blocked**
```python
def test_protected_dir_storage(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "storage", "anything.txt")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "text",
        "justification": "Testing storage dir blocklist",
        "capability_token": token,
        "session_id": "test_neg_storage"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    row = check_db_blocked()
    assert row is not None and row[1] == "blocked"
```

**Test: sandbox escape — .env masked inside Bubblewrap**  
This test creates a minimal script that tries to `cat .env` and confirms the output is empty (because `.env` is bind-mounted to `/dev/null` inside the sandbox):
```python
def test_sandbox_env_masked(gate_server):
    """Verify .env is masked to /dev/null inside the Bubblewrap sandbox (INV-011)."""
    script_path = os.path.expanduser("~/free-space.sh")
    env_path = os.path.join(config.WORKSPACE_DIR, ".env")
    try:
        with open(script_path, "w") as f:
            f.write(f"#!/bin/bash\ncat {env_path} 2>/dev/null\necho EXIT_CODE_$?\n")
        os.chmod(script_path, 0o755)
    except Exception as e:
        pytest.skip(f"Cannot create test script: {e}")

    token = generate_capability_token("run_script", "free-space.sh")
    payload = {
        "action": "run_script",
        "script_name": "free-space.sh",
        "justification": "Sandbox masking test",
        "capability_token": token,
        "session_id": "test_sandbox_mask"
    }
    try:
        resp = send_framed_request(gate_server.socket_path, payload)
        stdout = resp.get("executor_response", {}).get("stdout", "")
        # .env content must not appear — output should only be EXIT_CODE_0 (cat succeeded but file is empty)
        # Check that no key=value patterns appear (would indicate .env leaked)
        import re
        assert not re.search(r"[A-Z_]+=\S+", stdout), f".env content leaked into sandbox stdout: {stdout}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
```

**Test: automated bypass — HostExecutor called without PolicyGate**  
This verifies that a direct call to `HostExecutor.execute_service_control` for a blocked service still returns a denial (the allowlist enforcement is in HostExecutor itself, not only PolicyGate):
```python
def test_executor_allowlist_enforced_directly():
    """HostExecutor must enforce its own allowlist even when called without PolicyGate."""
    from security.host_executor import HostExecutor
    success, stdout, stderr = HostExecutor.execute_service_control("apache2")
    assert not success
    assert "allowlist" in stderr.lower() or "not in" in stderr.lower()
```

**Verify:** `python -m pytest tests/test_negative_security.py -v` — all 14 tests green.

---

### Gap 3 — `security/fim_daemon.py`: FAN_CREATE and FAN_ATTRIB missing from fanotify mark (§4.1)

**Problem:** §4.1 specifies the fanotify mark must monitor `FAN_MODIFY`, `FAN_CREATE`, `FAN_ATTRIB`, and `FAN_ONDIR`. Current implementation only marks `FAN_MODIFY | FAN_CLOSE_WRITE | FAN_ONDIR`. `FAN_CREATE` is required to catch new file drops. `FAN_ATTRIB` catches permission and ownership changes (a key indicator of privilege escalation setup).

**Fix:** In `security/fim_daemon.py`, add the missing constants and include them in the fanotify mark:

```python
FAN_CREATE = 0x00000100
FAN_ATTRIB = 0x00000004
```

Update the mark call:
```python
mask = FAN_MODIFY | FAN_CLOSE_WRITE | FAN_CREATE | FAN_ATTRIB | FAN_ONDIR
```

Update `_handle_event` to classify the `event_type` correctly from `mask`:
```python
if mask & FAN_CREATE:
    event_type = "create"
elif mask & FAN_ATTRIB:
    event_type = "attrib"
elif mask & FAN_MODIFY:
    event_type = "modify"
else:
    event_type = "close_write"
```

**Verify:** Creating a new file in `WORKSPACE_DIR` while the daemon runs produces a row with `event_type = "create"` in `/var/lib/secdaemon/fim_audit.db`.

---

### Gap 4 — `security/ebpf_telemetry.py`: `sys_enter_openat2` tracepoint missing (§4.2)

**Problem:** §4.2 explicitly calls out `sys_enter_openat` / `sys_enter_openat2` as the hook targets for file system modification tracking. The current BPF C program only has a probe on `sys_enter_openat`. `openat2` is the newer syscall used by many modern processes and is the one the spec calls for honeypot detection on.

**Fix:** In `security/ebpf_telemetry.py`, add `sys_enter_openat2` as a second tracepoint in the BPF C program string, reusing the same `open_event_t` struct and the same `honeypot_paths` BPF hash map lookup:

```c
TRACEPOINT_PROBE(syscalls, sys_enter_openat2) {
    struct open_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    data.flags = 0;  // openat2 flags are in a struct, not a flat int — set 0 as safe default

    struct honeypot_key_t key = {};
    __builtin_memcpy(key.path, data.filename, sizeof(key.path));
    u32 *val = honeypot_paths.lookup(&key);
    if (val) {
        honeypot_events.perf_submit(args, &data, sizeof(data));
    }

    open_events.perf_submit(args, &data, sizeof(data));
    return 0;
}
```

No changes to the Python side are needed — events flow through the same `open_events` and `honeypot_events` perf buffers and the existing callbacks handle them.

**Verify:** `grep "sys_enter_openat2" security/ebpf_telemetry.py` returns a match. On a system with BCC installed, opening a file via a process that uses `openat2` (e.g., newer `cat` builds) appears in `get_recent_connections()` within 2 seconds. On systems without BCC, the fallback still functions unchanged.

---

### Gap 5 — `security/telemetry_sanitizer.py` never called anywhere (INV-004 violation)

**Problem:** INV-004 is non-negotiable: "All incoming telemetry streams must pass through character allowlists and hard string truncation fields." `telemetry_sanitizer.py` with its `sanitize_telemetry()` function exists but is imported nowhere and called nowhere. This means every field arriving from eBPF event buffers, network discovery frames, and passive L2/L3 parse results goes directly into security_events.db and the dashboard without sanitization — a direct violation of INV-004.

**Fix:** Call `sanitize_telemetry()` at every telemetry ingestion point:

**In `security/ebpf_telemetry.py`**, import and apply after decoding each perf buffer event. In `_handle_exec_event`:
```python
from security.telemetry_sanitizer import sanitize_telemetry
raw = {"pid": str(event.pid), "uid": str(event.uid), "comm": event.comm.decode("utf-8", "ignore"), "filename": event.filename.decode("utf-8", "ignore")}
clean = sanitize_telemetry(raw)
# use clean["comm"], clean["filename"] etc. when appending to deque
```

Apply the same pattern in `_handle_tcp_connect` (fields: `pid`, `comm`, `daddr`, `dport`), `_handle_privilege_event` (fields: `pid`, `comm`, `uid`), and `_handle_honeypot_event` (fields: `pid`, `comm`, `filename`).

Note: `sanitize_telemetry` drops fields not in `FIELD_SCHEMAS`. The current FIELD_SCHEMAS covers `ip`, `port`, `pid`, `comm`, `path`, `hostname`, `bytes`, `timestamp`, `state`. Map eBPF fields to these keys before sanitizing: `filename` → `path`, `daddr` → `ip`, `dport` → `port`.

**In `security/network_discovery.py`**, import and apply in `_log_asset` before writing to the DB:
```python
from security.telemetry_sanitizer import sanitize_telemetry
raw = {"ip": ip, "hostname": hostname, "comm": vendor}
clean = sanitize_telemetry(raw)
ip = clean.get("ip", "")
hostname = clean.get("hostname", "")
```

**Verify:** `grep -rn "sanitize_telemetry" security/` returns matches in `ebpf_telemetry.py` and `network_discovery.py`. A telemetry event with a malicious comm field (e.g., `"bash; rm -rf /"`) arrives sanitized as `"bash rm-rf "` (shell metacharacters stripped by the `comm` allowlist `[a-zA-Z0-9_\-\.]`).

---

### Gap 6 — `kaia_dashboard.py` `AuditLogCollector`: ledger polled at wrong interval (Appendix D)

**Problem:** Appendix D specifies two distinct polling intervals: security_events.db at 250ms, `audit_ledger.json` at 500ms. The current `_run` loop polls both on the same 250ms cycle. This doubles the I/O load on the ledger file for no benefit and diverges from the spec.

**Fix:** Add a separate counter to `_run` that only calls `_poll_audit_ledger()` every other tick:

```python
def _run(self) -> None:
    """Poll security_events.db (250ms) and audit_ledger.json (500ms) separately."""
    ledger_tick = 0
    while not self._stop.is_set():
        try:
            new_entries = []
            new_entries.extend(self._poll_security_db())
            ledger_tick += 1
            if ledger_tick >= 2:   # every 2 × 250ms = 500ms
                new_entries.extend(self._poll_audit_ledger())
                ledger_tick = 0
            # ... rest of loop unchanged
```

**Verify:** The change is mechanical. Confirm with a code review — no runtime test needed beyond ensuring the dashboard still receives audit events.

---

### Gap 7 — `kaia_dashboard.py` `check threat` command: CVE details not enriched (§Appendix C)

**Problem:** The `> check threat <IP>` command calls `lookup_internetdb()` which returns a list of CVE IDs in `shodan.get("vulns")`. These IDs are displayed raw (e.g., `['CVE-2026-1234']`) but `threat_intel.lookup_cve_details(cve_id)` is never called to get the CVSS score and description from the local `cve.db`. This leaves the CVE data only half-surfaced.

**Fix:** In `_command_worker`, in the `check threat` handler, after displaying the CVE ID list, iterate the vulns and call `lookup_cve_details` for each (cap at 3 to avoid flooding the response area):

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

**Verify:** `> check threat 203.0.113.42` (after seeding a test block event) shows CVSS scores alongside CVE IDs if the CVE DB has entries for those IDs. Falls back gracefully to `"Details not available locally."` when the CVE is not in the local DB.

---

## Part B — Run the full test suite after all fixes

Run in this order:

```bash
python -m pytest tests/test_negative_security.py -v
python -m pytest tests/test_advanced_security.py -v
python -m pytest tests/test_database_utils.py -v
python -m pytest tests/test_kaia_cli.py -v
python -m pytest tests/test_tier5_security.py -v
python tests/verify_security.py
```

All must pass before doing anything else. Document any remaining failure with the exact error output — do not attempt to fix new failures without first reporting them.

---

## Part C — Final acceptance grep checks

Every one of these must return the stated result:

```bash
# .git is explicitly blocked in host_executor
grep -n "\.git" security/host_executor.py
# Expected: at least one match containing "git" in a blocklist context

# telemetry_sanitizer is now imported and used
grep -rn "sanitize_telemetry" security/
# Expected: matches in ebpf_telemetry.py and network_discovery.py

# openat2 tracepoint exists in eBPF
grep "sys_enter_openat2" security/ebpf_telemetry.py
# Expected: one match

# FAN_CREATE and FAN_ATTRIB defined in FIM
grep "FAN_CREATE\|FAN_ATTRIB" security/fim_daemon.py
# Expected: both defined and used in the mask

# No FIMDaemon() instantiation anywhere in dashboard
grep -n "FIMDaemon()" kaia_dashboard.py
# Expected: zero matches

# AuditLogCollector uses ledger_tick counter
grep -n "ledger_tick" kaia_dashboard.py
# Expected: at least 3 matches (init, increment, reset)
```

---

## What is explicitly deferred — do NOT implement this cycle

Per master_plan §12 and Appendix F:

- DuckDB delta computation for InternetDB snapshot diffing (Appendix C) — deferred
- Supply chain monitoring, baseline drift, secrets scanning (§11) — optional future
- GVisor / Firecracker containment tiers (no KVM kernel module assumed)
- DNS passive hostname DB (`dnsdb/dns.db`) population — schema exists, ingestion pipeline deferred
- Multi-agent consensus or distributed architectures
