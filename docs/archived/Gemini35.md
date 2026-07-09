# Kaia: System Security & Action Plan Verification Audit (Gemini 3.5 Flash)

**Audit Date:** 2026-07-09 (Updated)  
**Initial Audit Date:** 2026-07-04  
**Target Repository:** `Ekco-S64QTN6/Kaia`  
**OS Environment:** Arch Linux (Python 3.14)  
**Cross-Referenced Against:** [ClaudeSonnet5.md](file:///home/ekco/github/Kaia/docs/ClaudeSonnet5.md), [master_plan.md](file:///home/ekco/github/Kaia/docs/master_plan.md)  
**Status:** **Comprehensive Audit v2 — Critical Open Bugs Identified**

---

## 1. Executive Summary

This is a second-pass audit of the Kaia security administration system, cross-referenced against the findings in `ClaudeSonnet5.md` (which identified two live-production bugs missed by the original pass of this report) and re-verified against the `master_plan.md` authoritative specification.

**Test suite status:** All **33 automated tests pass** cleanly (9.38s runtime). However, the test suite produces **117 deprecation warnings** related to `datetime.utcnow()` in Python 3.14 — these are scheduled for removal in Python 3.16 and represent a ticking time bomb.

**Critical finding:** Two bugs identified by the Claude review remain **unfixed** in the codebase. Additionally, the systemd service file has **multiple specification drift** issues against the Appendix A requirements in `master_plan.md`.

---

## 2. CRITICAL: Unfixed Bugs from ClaudeSonnet5.md

The following bugs were discovered by the Claude Sonnet 5 review on 2026-07-05 and were **not caught by the original Gemini 3.5 audit**. As of this update, both remain **unfixed in the codebase**.

### 2.1 Bug CS-1: Tamper Detection False Positive on SQLite Database

**File:** [tamper_detection.py](file:///home/ekco/github/Kaia/security/tamper_detection.py)  
**Status:** ❌ **UNFIXED** — Code at lines 35-37 still uses byte-prefix hashing for `security_events.db`  
**Severity:** CRITICAL (deterministic false alarm on every daemon restart)

**Root Cause:** `TamperDetector` applies the same byte-prefix hash technique to both `audit_ledger.json` (a flat JSONL file, where this technique is correct) and `security_events.db` (a live SQLite database in WAL mode). The SQLite file header (first 100 bytes) contains fields that mutate on every write transaction — file change counter, schema cookie, freelist pointers, version-valid-for number. WAL checkpoints also rewrite pages within the hashed byte range.

**Effect:** Every `log_security_event()` call after baseline establishment changes the hashed byte region, guaranteeing a false `tamper_detected` CRITICAL alert. This then triggers `trigger_lockdown()`, which cascades into Bug CS-2.

**Current code (still present):**
```python
# Line 35-37 in tamper_detection.py
self.append_only_files = [
    config.SECURITY_DB_PATH,   # ← THIS IS THE BUG: SQLite file, not flat log
    config.AUDIT_LOG_PATH,
]
```

**Required fix:** Replace byte-prefix hashing for `security_events.db` with a content-hash technique that queries logical row content via SQL (`SELECT ... ORDER BY rowid LIMIT N`), which is invariant to SQLite header churn and WAL checkpoints. See `ClaudeSonnet5.md` for the complete implementation specification.

### 2.2 Bug CS-2: Service User Breaks Lockdown Escalation Path

**File:** [kaia-policy-gate.service](file:///home/ekco/github/Kaia/scripts/kaia-policy-gate.service)  
**Status:** ❌ **UNFIXED** — Line 19 still reads `User=ekco`  
**Severity:** CRITICAL (emergency lockdown mechanism is completely non-functional)

**Root Cause:** The deployed service unit file runs the Policy Gate daemon as `User=ekco` / `Group=kaiacord`. The `master_plan.md` Appendix A (line 594) explicitly requires `User=root`. This isn't cosmetic — it breaks two independent checks:

1. `systemctl start kaia-lockdown.service` fails with "Access denied... requires interactive authentication" (polkit prompt in headless daemon)
2. Fallback `kaia-lockdown.sh` explicitly checks `if [ "$EUID" -ne 0 ]` and refuses to run

**Effect:** The emergency lockdown mechanism **cannot execute under any trigger path** — tamper detection, honeypot triggers, or FIM YARA matches all call `trigger_lockdown()` and will all silently fail.

**Current code (still present):**
```ini
# Line 19 in kaia-policy-gate.service
User=ekco    # ← MUST BE: User=root (per master_plan.md Appendix A)
```

---

## 3. Service File Specification Drift

Beyond Bug CS-2, the [kaia-policy-gate.service](file:///home/ekco/github/Kaia/scripts/kaia-policy-gate.service) has additional deviations from the Appendix A specification in `master_plan.md` (lines 586-604):

| Directive | Appendix A Spec | Actual in Service File | Impact |
| :--- | :--- | :--- | :--- |
| `User=` | `root` | `ekco` | **CRITICAL** — lockdown path broken (Bug CS-2) |
| `ProtectHome=` | `true` | **MISSING** | Moderate — daemon can read `/home` contents |
| `PrivateDevices=` | `true` | **MISSING** | Low — daemon has access to device nodes |
| `RestartSec=` | `1` | `3` | Low — slower recovery from crashes |
| `CapabilityBoundingSet=` | `CAP_NET_RAW CAP_SYS_ADMIN` | **MISSING** (only `AmbientCapabilities` set) | Low — no bounding set ceiling defined |
| `ExecStart=` | `/usr/bin/python3 /path/to/...` | Hardcoded `.venv/bin/python` path | Cosmetic — works but not portable |
| Hardcoded paths | Should use `${KAIA_PROJECT_DIR}` | Hardcoded `/home/ekco/github/Kaia` in `ExecStart`, `WorkingDirectory` | Cosmetic — `install_services.sh` handles at deployment |

**Note:** The `kaia-lockdown.service` file at line 10 hardcodes `KAIA_PROJECT_DIR=/home/ekco/github/Kaia` but correctly uses `${KAIA_PROJECT_DIR}` in its `ExecStart`, which is the right pattern. The policy gate service should follow the same convention.

---

## 4. Python 3.14 Deprecation Warnings (117 total)

The test suite produces **117 deprecation warnings**. The most critical are Kaia's own code:

| Location | Deprecated API | Replacement |
| :--- | :--- | :--- |
| [policy_gate.py#L30](file:///home/ekco/github/Kaia/security/policy_gate.py#L30) | `datetime.utcnow()` | `datetime.now(datetime.UTC)` |
| [policy_gate.py#L68](file:///home/ekco/github/Kaia/security/policy_gate.py#L68) | `datetime.utcnow()` | `datetime.now(datetime.UTC)` |
| [policy_gate.py#L394](file:///home/ekco/github/Kaia/security/policy_gate.py#L394) | `datetime.utcnow()` | `datetime.now(datetime.UTC)` |
| [db.py#L36](file:///home/ekco/github/Kaia/security/db.py#L36) | `datetime.utcnow()` | `datetime.now(datetime.UTC)` |
| [schemas.py#L54](file:///home/ekco/github/Kaia/security/schemas.py#L54) | `datetime.utcnow` (via Pydantic default_factory) | `lambda: datetime.now(datetime.UTC)` |

`datetime.utcnow()` is scheduled for removal in Python 3.16. Since Kaia is running on Python 3.14, these should be migrated before the next Python upgrade cycle.

The remaining warnings come from third-party dependencies (`chromadb`, `llama_index`, `pydantic`, `gi`) using deprecated `asyncio.iscoroutinefunction`.

---

## 5. Structural Observations

### 5.1 Missing `security/__init__.py`

The `security/` directory has **no `__init__.py` file**. Python imports currently work because `config.py` is imported directly from the `core/` package (added to `PYTHONPATH`), and security modules are imported as `from security.X import Y` with the project root on the path. However, the lack of `__init__.py` means `security/` is technically an implicit namespace package rather than a regular package — this can cause subtle import issues with tools like `mypy`, IDEs, and some `pytest` configurations.

### 5.2 Legacy File Cleanup — Already Done

The master plan (§2.1 note, §8) calls for removal of `security/cognitive_wiring.py` and `storage/cognition/beliefs.json`. **Both files have already been deleted** — they no longer exist anywhere in the repository. References to them remain only in documentation (master_plan.md, archived docs), which is correct since those documents describe what was removed.

### 5.3 `IPvAnyAddress` Unused Import — Already Resolved

My original report (§4.2) noted an unused `IPvAnyAddress` import in [schemas.py](file:///home/ekco/github/Kaia/security/schemas.py). **This has been cleaned up** — the current `schemas.py` does not import `IPvAnyAddress`. The `MitigationRequest.target_ip` field is correctly typed as `str` with validation happening in `host_executor.py` via `socket.inet_aton()`.

---

## 6. Resolution Status of Original Bug Reports

### 6.1 10 Bugs from `02_Prompt_For_Agent.md` — All Resolved ✅

| Item | Description | Status | Verification |
| :--- | :--- | :--- | :--- |
| **Bug 1** | `honeypot.py`: `log_security_event` not imported | **Resolved** | [honeypot.py#L13](file:///home/ekco/github/Kaia/security/honeypot.py#L13) — import present |
| **Bug 2** | `kaia_dashboard.py`: `logger` undefined | **Resolved** | [kaia_dashboard.py#L49](file:///home/ekco/github/Kaia/kaia_dashboard.py#L49) — defined at module level |
| **Bug 3** | Lattice uses `min()` instead of `max()` | **Resolved** | [kaia_dashboard.py#L416](file:///home/ekco/github/Kaia/kaia_dashboard.py#L416) — `eff_idx = max(g_idx, w_idx)` |
| **Bug 4** | `FIMDaemon()` instantiated in polling loops | **Resolved** | [kaia_dashboard.py#L450](file:///home/ekco/github/Kaia/kaia_dashboard.py#L450) — queries SQLite directly |
| **Bug 5** | Geo field from reputation tags not GeoIP | **Resolved** | [kaia_dashboard.py#L331](file:///home/ekco/github/Kaia/kaia_dashboard.py#L331) — calls `lookup_geoip()` |
| **Bug 6** | Systemd units contain hardcoded paths | **Resolved** | `Environment=KAIA_PROJECT_DIR` set in both services |
| **Bug 7** | `test_tier5_security.py` lacks path/secret setup | **Resolved** | [test_tier5_security.py#L5-L9](file:///home/ekco/github/Kaia/tests/test_tier5_security.py#L5-L9) — configured |
| **Bug 8** | `add_rule` bypasses lattice check | **Resolved** | [policy_gate.py#L327](file:///home/ekco/github/Kaia/security/policy_gate.py#L327) — inside lattice gate |
| **Bug 9** | Command worker exits immediately | **Resolved** | [kaia_dashboard.py#L1415](file:///home/ekco/github/Kaia/kaia_dashboard.py#L1415) — uses `_cmd_stop` event |
| **Bug 10** | `len(engine.scanner)` TypeError | **Resolved** | [kaia_dashboard.py#L623](file:///home/ekco/github/Kaia/kaia_dashboard.py#L623) — counts `.yar` files |

### 6.2 8 Fixes from `02_2_kaia_full_review.md` — All Resolved ✅

1. **WAL Mode** — [db.py#L13](file:///home/ekco/github/Kaia/security/db.py#L13): `PRAGMA journal_mode=WAL` set ✅
2. **Sandbox Masking** — [host_executor.py#L136-L137](file:///home/ekco/github/Kaia/security/host_executor.py#L136-L137): `.env` → `/dev/null`, `storage/` → `tmpfs` ✅
3. **Diagnostics Capability Name** — [policy_gate.py#L316](file:///home/ekco/github/Kaia/security/policy_gate.py#L316): Target is `"diagnostics"` ✅
4. **Safe DB Closure** — Both [fim_daemon.py](file:///home/ekco/github/Kaia/security/fim_daemon.py) and [network_discovery.py](file:///home/ekco/github/Kaia/security/network_discovery.py) use try/except/finally ✅
5. **TCP Retransmit Sanitization** — [ebpf_telemetry.py#L311-L317](file:///home/ekco/github/Kaia/security/ebpf_telemetry.py#L311-L317): Address fields sanitized ✅
6. **Network Asset Sanitization** — [network_discovery.py#L279-L284](file:///home/ekco/github/Kaia/security/network_discovery.py#L279-L284): `sanitize_telemetry` called ✅
7. **Ledger Polling** — [kaia_dashboard.py#L1305-L1313](file:///home/ekco/github/Kaia/kaia_dashboard.py#L1305-L1313): Alternate 250ms/500ms cycle ✅
8. **CVE Enrichment** — [kaia_dashboard.py#L1571-L1574](file:///home/ekco/github/Kaia/kaia_dashboard.py#L1571-L1574): `lookup_cve_details` mapped ✅

---

## 7. Operational Subsystem Status

All major subsystems implemented and wired into the Policy Gate daemon's `__main__` block ([policy_gate.py#L508-L573](file:///home/ekco/github/Kaia/security/policy_gate.py#L508-L573)):

| Subsystem | Module | Status | Notes |
| :--- | :--- | :--- | :--- |
| Policy Gate IPC | `policy_gate.py` | ✅ Operational | Unix socket with 4-byte framed JSON |
| Tamper Detection | `tamper_detection.py` | ⚠️ Buggy | False positive on SQLite (Bug CS-1) |
| FIM Daemon | `fim_daemon.py` | ✅ Operational | fanotify + YARA, falls back to watchdog |
| eBPF Telemetry | `ebpf_telemetry.py` | ✅ Operational | Graceful fallback if not root |
| Passive Discovery | `network_discovery.py` | ✅ Operational | AF_PACKET raw frames, ARP/mDNS/LLMNR |
| Honeypot Coordinator | `honeypot.py` | ✅ Operational | File & port tripwires |
| YARA Rule Engine | `rule_engine.py` | ✅ Operational | systemd-run sandboxed validation |
| Threat Intelligence | `threat_intel.py` | ✅ Operational | Shodan InternetDB, GeoIP, CVE lookup |
| Dashboard TUI | `kaia_dashboard.py` | ✅ Operational | 4-pane curses UI with command interface |
| Security DB | `db.py` | ✅ Operational | WAL mode, append-only |
| Telemetry Sanitizer | `telemetry_sanitizer.py` | ✅ Operational | Character allowlist + truncation |
| Host Executor | `host_executor.py` | ✅ Operational | Path blocklist, cgroup wrapper, bwrap |

---

## 8. Comprehensive Gap Analysis vs. master_plan.md

### 8.1 Items Completed

- [x] Policy Gate out-of-process IPC daemon (§3)
- [x] Risk Tier classification and action routing (§3.3)
- [x] Lattice permission intersection — `max()` for stricter constraint (§3.3, INV-007)
- [x] Capability token HMAC verification with expiry (INV-002)
- [x] Path blocklist enforcement for write_file (INV-010, §3.3)
- [x] Bubblewrap sandbox with `.env` and `storage/` masking (INV-011, §3.3)
- [x] cgroup resource ceilings: CPU 25%, MEM 2G, PID 128 (§6.3)
- [x] eBPF telemetry hooks with fallback (§4.2)
- [x] fanotify FIM daemon with YARA scanning (§4.1)
- [x] Raw packet passive discovery — ARP, mDNS, LLMNR, SSDP (§5)
- [x] Honeypot file and port tripwires (§6.1)
- [x] YARA rule ingestion with schema validation and sandboxed compilation (§7)
- [x] Tamper detection perimeter (§6.4) — structure is correct, but Bug CS-1 undermines it
- [x] Dashboard 4-pane layout with command interface (§9)
- [x] AuditLogCollector replaces LogCollector (§10.3 item 1, Appendix D)
- [x] InternetDB caching with 7-day TTL (§10.2)
- [x] Legacy cognitive files removed (§8)
- [x] Negative test suite (§10.4) — covers path traversal, blocklists, token expiry, injection

### 8.2 Items Requiring Fixes (Bugs)

- [ ] **Bug CS-1:** Tamper detection byte-prefix hash on SQLite DB (§6.4) — **CRITICAL**
- [ ] **Bug CS-2:** Service `User=ekco` instead of `User=root` (Appendix A) — **CRITICAL**
- [ ] **Service directive drift:** Missing `ProtectHome`, `PrivateDevices`, `CapabilityBoundingSet` — **MODERATE**
- [ ] **Python 3.14 `utcnow()` deprecation:** 5 call sites across `policy_gate.py`, `db.py`, `schemas.py` — **MODERATE**
- [ ] **Missing `security/__init__.py`:** Implicit namespace package — **LOW**

### 8.3 Items Explicitly Deferred (per master_plan.md §11)

These are marked as optional future enhancements and are **not blocking**:

1. **DuckDB delta computation** for Shodan InternetDB snapshot diffing (Appendix C)
2. **`dnsdb/dns.db` population** — schema exists but no active packet ingestion
3. **Advanced sandboxing** — gVisor/Firecracker (require KVM)
4. **Supply chain monitoring** — package/dependency tracking (§11)
5. **Baseline drift detection** — behavioral baselines (§11)
6. **Alert confidence framework** — severity classification (§11)
7. **Secrets exposure detection** — API key scanning (§11)
8. **Backup awareness** — backup validation (§11)
9. **Self-health monitoring** — service watchdog integration (§11)
10. **Multi-agent consensus** — explicitly out of scope (§12)

---

## 9. Test Suite Summary

```
33 passed, 117 warnings in 9.38s
```

| Suite | Tests | Status |
| :--- | :--- | :--- |
| [test_negative_security.py](file:///home/ekco/github/Kaia/tests/test_negative_security.py) | 12 | ✅ All pass |
| [test_advanced_security.py](file:///home/ekco/github/Kaia/tests/test_advanced_security.py) | 7 | ✅ All pass |
| [test_database_utils.py](file:///home/ekco/github/Kaia/tests/test_database_utils.py) | 2 | ✅ All pass |
| [test_kaia_cli.py](file:///home/ekco/github/Kaia/tests/test_kaia_cli.py) | 2 | ✅ All pass |
| [test_tier5_security.py](file:///home/ekco/github/Kaia/tests/test_tier5_security.py) | 7 | ✅ All pass |
| [test_heuristics.py](file:///home/ekco/github/Kaia/tests/test_heuristics.py) | 3 | ✅ All pass |

**Note:** The test suite does **not** exercise the tamper detection false positive (Bug CS-1) or the lockdown escalation path under a non-root service account (Bug CS-2). These are integration/runtime bugs that only manifest in production when the daemon is running as a systemd service.

---

## 10. Priority-Ordered Action Items

1. **🔴 Fix Bug CS-1** — Rewrite `tamper_detection.py` to use SQL content-hash for `security_events.db` (see `ClaudeSonnet5.md` for implementation spec)
2. **🔴 Fix Bug CS-2** — Change `User=ekco` to `User=root` in `kaia-policy-gate.service`
3. **🟡 Add missing service directives** — `ProtectHome=true`, `PrivateDevices=true`, `CapabilityBoundingSet=CAP_NET_RAW CAP_SYS_ADMIN`, `RestartSec=1`
4. **🟡 Migrate `datetime.utcnow()`** — Replace all 5 call sites with `datetime.now(datetime.UTC)` before Python 3.16
5. **🟢 Add `security/__init__.py`** — Create empty init file for proper package semantics
6. **🟢 Add integration test** for tamper detection (verify no false positives over 2+ minutes of normal operation)

---

## Appendix: Corrections to Original Gemini 3.5 Audit (v1)

The following items from the original v1 audit (2026-07-04) were incorrect or incomplete:

1. **§4.2 `IPvAnyAddress` cosmetic cleanup** — Reported as "retained for backwards-compatibility." **Correction:** The import has already been removed from `schemas.py`. This item is resolved.
2. **Missing coverage of Claude's Bug CS-1 and CS-2** — The original audit verified all items from `02_Prompt_For_Agent.md` and `02_2_kaia_full_review.md` but did not independently discover the tamper detection false positive or the service user misconfiguration. These were runtime/integration bugs not exercised by the test suite.
3. **Service file analysis was incomplete** — The original audit noted hardcoded paths (§4.1) but did not compare the full service file against the Appendix A specification, missing the `User=`, `ProtectHome=`, and `PrivateDevices=` deviations.
