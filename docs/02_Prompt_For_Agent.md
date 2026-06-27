# Kaia Coding Agent — Tier 3/5 Correction & Next Action Plan
**Date:** 2026-06-27  
**Source of truth:** `docs/master_plan.md`  
**Repo:** `Ekco-S64QTN6/Kaia`  
**Agent role:** implement corrections and remaining items; Claude reviews next cycle

---

## Context

The previous agent pass completed the bulk of Tier 3 and Tier 5 implementation. A joint review by two agents (one Claude, one Gemini 3.5) verified that all major structural work landed correctly: `AuditLogCollector` crash fix, SQL schema correction, four security panes, Sec filter, cognitive wiring removal, storage path unification, schemas import fix, `IocRuleRequest`, tamper detection, FIM daemon, eBPF telemetry, network discovery, honeypot coordinator, YARA rule engine, lockdown scripts, and both negative test suites all exist and are broadly correct.

**Eight confirmed bugs remain from that pass.** Fix all eight before doing anything else. They are listed below in priority order — bugs 1 and 4 are runtime crashers that will prevent any testing.

---

## Part A — Bug Fixes (correct before anything else)

Work through these in order. Each entry specifies the file, the exact problem, and the required fix.

---

### Bug 1 — `security/honeypot.py`: `log_security_event` never imported (runtime crash)

**Problem:** `log_security_event()` is called at two points (honeypot file access handler ~line 171, port trigger handler ~line 238) but the import is absent. Any decoy trigger will raise `NameError: name 'log_security_event' is not defined` and crash the honeypot thread.

**Fix:** Add the import at the top of `security/honeypot.py`, alongside the other imports:

```python
from security.db import log_security_event
```

**Verify:** `grep -n "log_security_event\|from security.db" security/honeypot.py` must show the import present before the first call site.

---

### Bug 2 — `kaia_dashboard.py`: `logger` referenced but never defined (runtime crash)

**Problem:** Multiple collector classes (`ThreatIntelCollector`, `ContainmentCollector`, `SystemSecurityCollector`) call `logger.error(...)` but no module-level `logger` is defined in `kaia_dashboard.py`. This raises `NameError` the first time any collector hits an exception path — which happens immediately on systems where `nft` or the Policy Gate socket is absent.

**Fix:** Add one line at module level in `kaia_dashboard.py`, directly after the final `import` statement and before the constant definitions:

```python
logger = logging.getLogger(__name__)
```

**Verify:** `grep -n "^logger\|^import logging" kaia_dashboard.py` confirms both `import logging` and the `logger =` assignment exist.

---

### Bug 3 — `kaia_dashboard.py` `ContainmentCollector.run()`: lattice direction inverted

**Problem:** `ContainmentCollector.run()` computes the effective lattice level as:
```python
eff_idx = min(g_idx, w_idx)
```
The spec (master_plan §1.1 axiom, archive §2, and `policy_gate.py` line ~321) is explicit: the lattice resolves to `max()` — the **stricter** level wins. `min()` produces the least-restrictive level, which is the opposite of the intended security behaviour and means the dashboard displays a falsely permissive containment level.

**Fix:** Change `min` to `max` in `ContainmentCollector.run()`:

```python
eff_idx = max(g_idx, w_idx)
eff_lvl = config.LATTICE_LEVELS[eff_idx]
```

**Verify:** On a system where `GLOBAL_LATTICE_LEVEL = "bwrap"` (index 3) and `WORKSPACE_LATTICE_LEVEL = "none"` (index 0), the displayed effective level must be `bwrap`, not `none`.

---

### Bug 4 — `kaia_dashboard.py` `ContainmentCollector.run()`: `FIMDaemon` instantiated fresh every 2 seconds

**Problem:** The collector calls `FIMDaemon()` inside its polling loop:
```python
fim_daemon = FIMDaemon()
alerts = fim_daemon.get_recent_alerts(5)
```
Each call creates a brand-new, unstarted `FIMDaemon` instance. Its `get_recent_alerts()` method queries `/var/lib/secdaemon/fim_audit.db`, so the method itself will work, but this is wasteful (instantiates libc/ctypes bindings on every tick) and brittle. More importantly, `FIMDaemon.__init__` may attempt to allocate resources or log on construction, making the 2-second loop noisy and slow.

**Fix:** Replace the `FIMDaemon()` instantiation inside the loop with a direct SQLite query against the FIM audit database. `ContainmentCollector` should not own or start a `FIMDaemon` — it should only read the audit DB that the already-running daemon writes to.

Replace the FIM section inside `ContainmentCollector.run()` with:

```python
fim_alerts = []
fim_db_path = "/var/lib/secdaemon/fim_audit.db"
if os.path.exists(fim_db_path):
    try:
        conn = sqlite3.connect(fim_db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, event_type, pid, comm, path, yara_matches, sha256
            FROM fim_events
            ORDER BY rowid DESC LIMIT 5
        """)
        rows = cursor.fetchall()
        for row in rows:
            fim_alerts.append({
                "timestamp": row[0],
                "event_type": row[1],
                "pid": row[2],
                "comm": row[3],
                "path": row[4],
                "yara_matches": row[5],
                "sha256": row[6]
            })
        conn.close()
    except Exception as e:
        logger.error(f"Failed to query FIM audit DB: {e}")
```

Remove the `from security.fim_daemon import FIMDaemon` import from inside the `run()` body if it was placed there. `FIMDaemon` should not be imported in `kaia_dashboard.py` at all — the dashboard reads data, it does not manage the daemon.

**Verify:** The `ContainmentCollector.run()` loop contains no `FIMDaemon()` instantiation. `grep -n "FIMDaemon" kaia_dashboard.py` returns zero matches.

---

### Bug 5 — `ThreatIntelCollector`: geo field populated from reputation tags, not country data

**Problem:** In `ThreatIntelCollector.run()`, the `recent_blocks` list is built like this:
```python
rep = threat_intel.get_ip_reputation(ip)
recent_blocks.append({
    ...
    "geo": rep.get("tags", ["US"]) if rep.get("tags") else ["US"],
    ...
})
```
`rep["tags"]` contains Shodan threat classification tags (e.g. `["scanner", "c2"]`), not geographic data. The dashboard Pane 2 mockup (master_plan §9.2) explicitly shows country codes like `(US)`, `(RU)`, `(CN)` — these should come from GeoIP lookup, not threat tags.

**Fix:** Call `threat_intel.lookup_geoip(ip)` and use the returned country field for geo display. Keep the existing `tags` field for Shodan threat labels.

Replace the block-building section in `ThreatIntelCollector.run()`:

```python
for ip in unique_ips:
    rep = threat_intel.get_ip_reputation(ip)
    shodan = threat_intel.lookup_internetdb(ip)
    geo = threat_intel.lookup_geoip(ip)
    country = geo.get("country", "Unknown")
    recent_blocks.append({
        "ip": ip,
        "geo": country,                          # single string now, not a list
        "tags": shodan.get("tags", []),
        "ports": shodan.get("ports", [])
    })
```

Update `_draw_threat_intel_pane()` in the UI class accordingly — `item.get("geo")` is now a string, so the join call must be removed:

```python
line = f" • {ip:<15} ({geo})"
```

**Verify:** With a valid GeoLite2 MMDB present, blocked IPs show a country name in parentheses, not a list of Shodan tags. Without the MMDB, the field shows `"Unknown"` gracefully (the existing `lookup_geoip` fallback already handles this).

---

### Bug 6 — `scripts/kaia-policy-gate.service` and `scripts/kaia-lockdown.service`: hardcoded user paths

**Problem:** Both service files contain:
```
ExecStart=/usr/bin/python3 /home/ekco/github/Kaia/security/policy_gate.py
ExecStart=/bin/bash /home/ekco/github/Kaia/scripts/kaia-lockdown.sh
```
These break on any machine where the repo is not at that exact path.

**Fix:** Replace the hardcoded path with a systemd `Environment=` variable so the operator only needs to edit one line.

In `scripts/kaia-policy-gate.service`, add to the `[Service]` block:
```ini
Environment=KAIA_PROJECT_DIR=/home/ekco/github/Kaia
ExecStart=/usr/bin/python3 ${KAIA_PROJECT_DIR}/security/policy_gate.py
WorkingDirectory=${KAIA_PROJECT_DIR}
```
Remove the old hardcoded `ExecStart` line.

In `scripts/kaia-lockdown.service`, same pattern:
```ini
Environment=KAIA_PROJECT_DIR=/home/ekco/github/Kaia
ExecStart=/bin/bash ${KAIA_PROJECT_DIR}/scripts/kaia-lockdown.sh
```

Add a comment block at the top of both files:
```ini
# IMPORTANT: Set KAIA_PROJECT_DIR to the absolute path of your Kaia repo.
# Edit the Environment= line below before running: systemctl enable --now <service>
```

**Verify:** `grep -n "ekco\|home" scripts/kaia-policy-gate.service scripts/kaia-lockdown.service` returns only the `Environment=` line, not any `ExecStart` line.

---

### Bug 7 — `tests/test_tier5_security.py`: missing `sys.path` setup and capability secret env var

**Problem:** The file contains no `sys.path` manipulation and no `KAIA_CAPABILITY_TOKEN_SECRET` environment variable assignment before imports. Every other test file in the suite (e.g. `test_advanced_security.py`, `verify_security.py`, `test_kaia_cli.py`) handles this at the top. Running `pytest tests/test_tier5_security.py` on a clean shell will fail with `ImportError` or `KeyError` before the first test runs.

**Fix:** Insert the following block at the very top of `tests/test_tier5_security.py`, before any project imports, mirroring the pattern in `test_advanced_security.py`:

```python
import os
import sys
import pathlib

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))
```

The `import os`, `import sys`, and `import pathlib` lines that are already in the file should be deduplicated — keep only one block at the top.

**Verify:** `python -m pytest tests/test_tier5_security.py -v --collect-only` completes collection without `ImportError` or `KeyError`.

---

### Bug 8 — `security/policy_gate.py`: `add_rule` action bypasses the lattice permission check

**Problem:** In `evaluate_and_execute()`, the `add_rule` handler is placed before the lattice/capability-intersection block. This means `add_rule` is never filtered by `effective_permissions` — any caller can invoke it regardless of what `GLOBAL_PERMISSIONS` or `WORKSPACE_PERMISSIONS` contain. This is inconsistent with the permission model in INV-007.

**Fix — two-part:**

1. Add `"add_rule"` to both permission sets in `core/config.py`:

```python
GLOBAL_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script", "add_rule"}
WORKSPACE_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script", "add_rule"}
```

2. In `security/policy_gate.py`, move the `add_rule` handling to **after** the lattice block (after the `effective_permissions` check), not before it. The `lockdown` action may remain before the lattice check since it is an emergency override, but `add_rule` is a normal privileged action and must respect the permission intersection.

The restructured flow in `evaluate_and_execute()` should be:
```
1. Handle "lockdown" early-exit (before lattice — emergency override)
2. Compute effective_permissions (lattice intersection)
3. Check action in effective_permissions — deny if absent
4. Schema validation + token verification + execution (all actions including add_rule fall here)
```

**Verify:** With `"add_rule"` removed from `WORKSPACE_PERMISSIONS`, an `add_rule` request returns `{"status": "denied", "message": "Lattice violation: action 'add_rule' is blocked by security policy."}`. With it present in both sets, the request proceeds to token validation.

---

## Part B — What to Do Next

Complete the bug fixes above first. Then proceed through these blocks in order.

---

### Block N1 — Run the full test suite; establish a green baseline

After applying all eight fixes, run every test suite and confirm green before touching anything else:

```bash
python -m pytest tests/test_negative_security.py -v
python -m pytest tests/test_advanced_security.py -v
python -m pytest tests/test_database_utils.py -v
python -m pytest tests/test_kaia_cli.py -v
python -m pytest tests/test_tier5_security.py -v
python tests/verify_security.py
```

Any remaining failures must be investigated and fixed before Block N2. Document what still fails and why if anything does not pass after the Bug 5 fixes above.

**Acceptance:** All five pytest suites report zero failures. `verify_security.py` prints `=== ALL SECURITY TESTS PASSED! ===`.

---

### Block N2 — GeoLite2 wiring completion (T5-7 finishing work)

Bug 5 fixes the geo field in `ThreatIntelCollector`. This block completes the GeoLite2 integration so that Pane 2 shows country annotation on blocked IPs end-to-end.

**What is already done:** `lookup_geoip()` in `security/threat_intel.py` already exists and reads from `GEOIP_DB_PATH`. The `scripts/update_geoip.sh` download script already exists. `config.py` already defines `GEOIP_DB_PATH`.

**What still needs wiring:**

1. **`threat_intel.py` `lookup_geoip()` debug logging:** The function currently logs at `WARNING` level when the MMDB is absent, which floods the dashboard logs on fresh installs. Change the "GeoIP MMDB lookup failed or not configured" log call from `logger.debug` (check current level) to explicitly `logger.debug` — not `warning`. Confirm with `grep -n "logger\." security/threat_intel.py`.

2. **Dashboard `_draw_threat_intel_pane()` geo annotation for audit-log-sourced blocks:** After Bug 5 fix, `ThreatIntelCollector` provides geo per IP in its snapshot. Confirm the `_draw_threat_intel_pane()` render correctly uses `item.get("geo")` as a string (not a list join). The line should read:
   ```python
   line = f" • {ip:<15} ({item.get('geo', '?')})"
   ```

3. **Add install note to `README.md`:** In the Getting Started section, add a step after the PostgreSQL setup block:
   ```markdown
   ### 5. (Optional) GeoIP Database
   To enable country annotations on blocked IPs, obtain a free MaxMind license key
   from https://www.maxmind.com and run:
   ```bash
   export MAXMIND_LICENSE_KEY="your_key_here"
   ./scripts/update_geoip.sh
   ```
   Without this the geo field displays "Unknown" — all other functionality is unaffected.
   ```

**Acceptance:** With a valid MMDB at `storage/threat_intel/geoip/GeoLite2-City.mmdb`, the Threat Intel pane shows `• 203.0.113.42 (United States)`. Without it, the pane shows `• 203.0.113.42 (Unknown)` with no errors or warnings in the log.

---

### Block N3 — Policy Gate systemd service install script

The `kaia-policy-gate.service` and `kaia-lockdown.service` files exist (and will have correct `Environment=` variables after Bug 6 fix). What's missing is the installation step.

**Create `scripts/install_services.sh`:**

```bash
#!/bin/bash
# Install and enable Kaia systemd services.
# Run with sudo from the repository root.

set -e

KAIA_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for SERVICE in kaia-policy-gate kaia-lockdown; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed "s|KAIA_PROJECT_DIR=/home/ekco/github/Kaia|KAIA_PROJECT_DIR=${KAIA_PROJECT_DIR}|g" \
        "$SRC" > "$DEST"
    echo "Installed $DEST"
done

systemctl daemon-reload
systemctl enable --now kaia-policy-gate.service
echo "kaia-policy-gate.service enabled and started."
echo "kaia-lockdown.service installed (start manually when needed)."
```

Make it executable (`chmod +x scripts/install_services.sh`).

**Update `activate_kaia_env.sh`:** The existing script already has a fallback nohup block. Update the comment above the systemd detection block to reference `scripts/install_services.sh`:

```bash
# To run as a systemd service instead of nohup, run: sudo scripts/install_services.sh
```

**Acceptance:** Running `sudo scripts/install_services.sh` from the repo root produces correct service files in `/etc/systemd/system/` with the actual repo path substituted, and `systemctl is-active kaia-policy-gate.service` returns `active`.

---

### Block N4 — Dashboard manual verification checklist

After Blocks N1–N3, do a full manual launch verification. This is not automated — you are checking visual correctness of the TUI.

Launch:
```bash
python kaia_dashboard.py
```

Verify each item:

- [ ] Dashboard launches without any Python exception or traceback
- [ ] All four panes render with borders and titles (Security Audit Log, Threat Intelligence, Containment & Sentinel, System Security)
- [ ] Command interface pane renders at the bottom with `>` input line
- [ ] Pressing `F` cycles filter modes: `All → Sec → Net → Hw → All`
- [ ] Pane 1 shows audit events from `security_events.db` (run `python tests/verify_security.py` in another terminal to generate events)
- [ ] Pane 2 shows firewall rule counts and "No LAN assets discovered yet" or actual assets if on LAN
- [ ] Pane 3 shows the lattice calculation string with `max` (not `min`) resolution
- [ ] Pane 3 shows FIM alerts as "No FIM alerts" (if daemon not running) or actual entries (if running)
- [ ] Pane 4 shows Policy Gate as `RUNNING` or `OFFLINE` correctly based on actual socket state
- [ ] Removing the policy gate socket (kill the gate process) causes Pane 2 to show `⚠ POLICY GATE OFFLINE` banner
- [ ] Typing `show audit --since 1h` in the command pane and pressing Enter returns audit rows or "No audit events found"
- [ ] Typing `check threat 8.8.8.8` returns Shodan/geo data or graceful "not found" message
- [ ] `Q` exits cleanly with terminal restored

Document any visual glitches or pane rendering issues as comments in the code — do not fix layout issues beyond what blocks N1–N3 address. Layout iteration is deferred.

---

### Block N5 — Final acceptance grep checks

Run these greps and confirm all return zero matches:

```bash
# No cognitive wiring
grep -rn "AffectiveState\|beliefs.json\|cognitive_wiring\|dream_cycle\|update_system_prompts" . --include="*.py"

# No LogCollector live references
grep -n "LogCollector" kaia_dashboard.py | grep -v "^.*#"

# No hardcoded user paths in service files
grep -n "/home/ekco" scripts/kaia-policy-gate.service scripts/kaia-lockdown.service | grep "ExecStart"

# No FIMDaemon instantiation in dashboard
grep -n "FIMDaemon()" kaia_dashboard.py

# log_security_event import present in honeypot
grep -n "from security.db import" security/honeypot.py

# logger defined in dashboard
grep -n "^logger = " kaia_dashboard.py
```

Every grep must return zero results (for the first five checks) or at least one result (for the last two). If any check fails, fix before declaring this cycle complete.

---

## What is explicitly deferred — do NOT implement this cycle

Per master_plan §12 and the tier 5 plan dependency notes:

- DuckDB delta computation for InternetDB snapshot diffing (Appendix C)
- Supply chain monitoring, baseline drift, secrets scanning (§11)
- GVisor / Firecracker containment tiers (no KVM kernel module assumed)
- Multi-agent consensus or distributed architectures
- Any changes to `main.py` beyond what the cognitive wiring removal in Bug Fixes already covered

---
Here's what the live grep actually found versus what Gemini reported:
Gemini was correct on 8 bugs — all confirmed with exact line numbers.
Gemini missed 2 bugs that the live repo exposed:

Bug 9 (command panel completely non-functional): _command_worker starts in __init__ when self._running = False. The loop condition while self._running or not self._cmd_queue.empty() evaluates to False immediately and the thread exits. Every command typed by the user is silently dropped. _running only becomes True later inside _main_loop. This is the most impactful new finding — the command panel is dead on arrival.
Bug 10 (len(engine.scanner) TypeError): yara.Rules objects don't implement __len__. This crashes SystemSecurityCollector every 5 seconds silently. The file-count fallback is already written two lines below — it just needs to replace the broken call.

Gemini was right that these are already done correctly: cognitive wiring gone, schemas import fixed, AuditLogCollector wired, SQL schema correct, Sec filter wired in both places, storage paths right, THREAT_INTEL_DIR from config, block_ip logging present, TamperDetector wired.
The prompt now has all 10 bugs with verified line numbers and no invented findings.

# Kaia Coding Agent — Verified Bug Fix Prompt
**Date:** 2026-06-27  
**Source of truth:** `docs/master_plan.md`  
**Repo:** `Ekco-S64QTN6/Kaia`  
**Verified by:** live `git clone` + grep/cat — every finding below has a confirmed line number

---

## What was verified correct from the last pass — do not touch

The following items were confirmed complete and correct in the live repo:

- `security/cognitive_wiring.py` deleted, zero references remain in `main.py`
- `security/schemas.py` — `from typing import Any` is at the top of the import block
- `kaia_dashboard.py` `CollectorManager.__init__` — correctly uses `AuditLogCollector` (line 1333)
- `AuditLogCollector._poll_security_db` — SELECT query correctly uses `payload_hash` and `disposition` columns (line 1181)
- Sec filter mode wired in both `_handle_input` (line 1745) and `_draw_logs_pane` (line 1885)
- `core/config.py` storage paths — `SECURITY_STORAGE_DIR`, `SECURITY_DB_PATH`, `AUDIT_LOG_PATH`, `THREAT_INTEL_DIR` all correct
- `security/threat_intel.py` — `THREAT_INTEL_DIR` imported from `config`, not derived from `SECURITY_DB_PATH`
- `security/policy_gate.py` — `block_ip` approved events logged to `security_events.db` at line 370
- `security/tamper_detection.py` exists and is wired into `policy_gate.py` `__main__` at line 526

---

## Part A — 10 confirmed bugs to fix

All line numbers verified against live repo on 2026-06-27.

---

### Bug 1 — `security/honeypot.py`: `log_security_event` called but never imported

**Confirmed:** `log_security_event()` called at lines 171 and 238. No import of it appears in the file (imports end at line 12).

**Fix:** Add to the imports at the top of `security/honeypot.py`:
```python
from security.db import log_security_event
```

**Verify:** `grep -n "from security.db import\|log_security_event" security/honeypot.py` — import line must appear before line 171.

---

### Bug 2 — `kaia_dashboard.py`: module-level `logger` never defined

**Confirmed:** `logger.error()` used at lines 342 and 454 inside collector classes. No `logger = logging.getLogger(...)` exists at module level (confirmed via `grep -n "^logger" kaia_dashboard.py` returning nothing).

**Fix:** Add one line directly after the final `import` statement at the top of `kaia_dashboard.py`, before the constant definitions:
```python
logger = logging.getLogger(__name__)
```

**Verify:** `grep -n "^logger = " kaia_dashboard.py` returns exactly one result.

---

### Bug 3 — `kaia_dashboard.py` line 411: lattice uses `min()` instead of `max()`

**Confirmed:** `ContainmentCollector.run()` at line 411:
```python
eff_idx = min(g_idx, w_idx)
```
The spec (master_plan §1.1 axiom, archive §2) and `policy_gate.py` line ~321 all use `max()` — stricter level wins. `min()` silently displays the least-restrictive level.

**Fix:**
```python
eff_idx = max(g_idx, w_idx)
```

**Verify:** With `GLOBAL_LATTICE_LEVEL = "bwrap"` (index 3) and `WORKSPACE_LATTICE_LEVEL = "none"` (index 0), the dashboard Containment pane shows `EFFECTIVE(3)`, not `EFFECTIVE(0)`.

---

### Bug 4 — `kaia_dashboard.py` lines 450–451 and 1626–1627: `FIMDaemon()` instantiated fresh inside polling loops

**Confirmed:** Two separate locations:
- Line 450–451 in `ContainmentCollector.run()`: `from security.fim_daemon import FIMDaemon` / `fim_daemon = FIMDaemon()`
- Line 1626–1627 in the command worker's `show fim alerts` handler: same pattern

Each call creates a new, unstarted `FIMDaemon` object. `FIMDaemon.__init__` loads ctypes/libc bindings on every tick. The 2-second `ContainmentCollector` loop is particularly bad.

**Fix for `ContainmentCollector.run()`:** Replace the FIMDaemon instantiation with a direct SQLite query against the FIM audit database. The dashboard reads data written by the daemon — it must not own or instantiate the daemon.

Replace lines 450–455 with:
```python
fim_alerts = []
fim_db_path = "/var/lib/secdaemon/fim_audit.db"
if os.path.exists(fim_db_path):
    try:
        conn = sqlite3.connect(fim_db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, event_type, pid, comm, path, yara_matches, sha256
            FROM fim_events ORDER BY rowid DESC LIMIT 5
        """)
        rows = cursor.fetchall()
        for row in rows:
            fim_alerts.append({
                "timestamp": row[0], "event_type": row[1], "pid": row[2],
                "comm": row[3], "path": row[4], "yara_matches": row[5], "sha256": row[6]
            })
        conn.close()
    except Exception as e:
        logger.error(f"Failed to query FIM audit DB: {e}")
```

**Fix for command worker `show fim alerts` handler (lines 1626–1628):** Same approach — query `fim_audit.db` directly with the same SELECT, do not instantiate `FIMDaemon`.

**Verify:** `grep -n "FIMDaemon()" kaia_dashboard.py` returns zero results.

---

### Bug 5 — `kaia_dashboard.py` line 329: geo field populated from reputation tags, not GeoIP

**Confirmed:** Line 329:
```python
"geo": rep.get("tags", ["US"]) if rep.get("tags") else ["US"],
```
`rep["tags"]` is a list of Shodan threat classification strings (e.g. `["scanner", "c2"]`), not geographic data. The Pane 2 render at line 1934 then does `",".join(item.get("geo", []))` and displays threat tags inside the country parentheses.

**Fix:** Call `threat_intel.lookup_geoip(ip)` and use the country field. In `ThreatIntelCollector.run()`, replace the block-building section:
```python
for ip in unique_ips:
    rep = threat_intel.get_ip_reputation(ip)
    shodan = threat_intel.lookup_internetdb(ip)
    geo = threat_intel.lookup_geoip(ip)
    recent_blocks.append({
        "ip": ip,
        "geo": geo.get("country", "Unknown"),   # string, not list
        "tags": shodan.get("tags", []),
        "ports": shodan.get("ports", [])
    })
```

Update `_draw_threat_intel_pane()` at line ~1934 — `geo` is now a string, remove the join:
```python
geo = item.get("geo", "Unknown")
line = f" • {ip:<15} ({geo})"
```

**Verify:** With no MMDB present, blocked IPs show `(Unknown)`. With a valid MMDB, they show `(United States)` or appropriate country name. Shodan threat tags appear only in the tags field, not the geo field.

---

### Bug 6 — `scripts/kaia-policy-gate.service` and `scripts/kaia-lockdown.service`: hardcoded paths

**Confirmed:**  
- `kaia-policy-gate.service` line 13: `ExecStart=/usr/bin/python3 /home/ekco/github/Kaia/security/policy_gate.py`
- `kaia-lockdown.service` line 8: `ExecStart=/bin/bash /home/ekco/github/Kaia/scripts/kaia-lockdown.sh`

**Fix:** In both service files, add an `Environment=` variable to the `[Service]` block and reference it in `ExecStart`. Add a comment at the top instructing the operator to edit exactly one line.

`kaia-policy-gate.service` `[Service]` block becomes:
```ini
# IMPORTANT: Set KAIA_PROJECT_DIR to the absolute path of your Kaia repository
# before running: sudo scripts/install_services.sh  (or systemctl enable --now)
Environment=KAIA_PROJECT_DIR=/home/ekco/github/Kaia
ExecStart=/usr/bin/python3 ${KAIA_PROJECT_DIR}/security/policy_gate.py
WorkingDirectory=${KAIA_PROJECT_DIR}
```

`kaia-lockdown.service` `[Service]` block becomes:
```ini
# IMPORTANT: Set KAIA_PROJECT_DIR to the absolute path of your Kaia repository
Environment=KAIA_PROJECT_DIR=/home/ekco/github/Kaia
ExecStart=/bin/bash ${KAIA_PROJECT_DIR}/scripts/kaia-lockdown.sh
```

Also **create `scripts/install_services.sh`** — a one-time installer that substitutes the correct repo path via sed, so operators do not have to manually edit service files:

```bash
#!/bin/bash
# Install Kaia systemd services with the correct project path.
# Run with sudo from the repository root: sudo scripts/install_services.sh
set -e
KAIA_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SERVICE in kaia-policy-gate kaia-lockdown; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed "s|KAIA_PROJECT_DIR=/home/ekco/github/Kaia|KAIA_PROJECT_DIR=${KAIA_PROJECT_DIR}|g" \
        "$SRC" > "$DEST"
    echo "Installed $DEST"
done
systemctl daemon-reload
systemctl enable --now kaia-policy-gate.service
echo "Done. kaia-lockdown.service installed but not enabled (start manually on breach)."
```

**Verify:** `grep "ExecStart" scripts/kaia-policy-gate.service scripts/kaia-lockdown.service` — neither ExecStart line contains a literal `/home/` path.

---

### Bug 7 — `tests/test_tier5_security.py`: missing `sys.path` setup and env var

**Confirmed:** File begins with `import os` / `import sys` etc and immediately imports `import config` at line 9 with no `sys.path` manipulation and no `KAIA_CAPABILITY_TOKEN_SECRET` assignment before it. Running from any directory outside the repo root or in a clean shell will raise `ModuleNotFoundError` or `KeyError`/`sys.exit(1)` from config.

**Fix:** Insert the following block at the very top of `tests/test_tier5_security.py`, before all other imports, matching the pattern in `tests/test_advanced_security.py`:
```python
import os
import sys
import pathlib

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))
```

Deduplicate the `import os` and `import sys` lines that already exist in the file — keep only the block above.

**Verify:** `python -m pytest tests/test_tier5_security.py -v --collect-only` completes collection without `ModuleNotFoundError` or `KeyError`.

---

### Bug 8 — `security/policy_gate.py` line 284: `add_rule` bypasses lattice permission check

**Confirmed:** The `add_rule` handler starts at line 284, which is before the lattice computation block at line ~323 and the `effective_permissions` check at line 328–330. `add_rule` is also absent from `GLOBAL_PERMISSIONS` and `WORKSPACE_PERMISSIONS` in `config.py` lines 202–203.

**Fix — two parts:**

1. `core/config.py` — add `"add_rule"` to both permission sets:
```python
GLOBAL_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script", "add_rule"}
WORKSPACE_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script", "add_rule"}
```

2. `security/policy_gate.py` — move the `add_rule` handler to after the lattice block. The `lockdown` early-exit may remain before the lattice check (it is an emergency override). `add_rule` is a normal privileged action and must respect the permission intersection. The restructured order in `evaluate_and_execute()` should be:
   1. `lockdown` early-exit
   2. Lattice computation + `effective_permissions` intersection check
   3. All other action handlers including `add_rule`

**Verify:** With `"add_rule"` removed from `WORKSPACE_PERMISSIONS` (temporary test), an `add_rule` request returns `{"status": "denied", "message": "Lattice violation: action 'add_rule' is blocked by security policy."}`. Restore the permission set afterwards.

---

### Bug 9 — `kaia_dashboard.py` line 1451: command worker exits immediately on startup

**Confirmed:** `_command_worker` is started in `KaiamonUI.__init__` (line 1401) when `self._running = False` (line 1390). The worker loop condition at line 1451 is:
```python
while self._running or not self._cmd_queue.empty():
```
At thread start: `_running` is `False` and the queue is empty → `False or not True` → `False`. The thread exits before the UI starts. `self._running` is set to `True` only at line 1711 inside `_main_loop`, which runs later. Every command typed by the user is silently dropped — the command panel is completely non-functional.

**Fix:** Replace the loop condition with one that does not depend on `_running` for liveness. Add a dedicated `threading.Event` for shutdown, or simplest: use a sentinel-based blocking get that keeps the thread alive until explicitly stopped.

Change the `__init__` to add a stop event:
```python
self._cmd_stop = threading.Event()
```

Change the worker loop:
```python
while not self._cmd_stop.is_set():
    try:
        cmd_line = self._cmd_queue.get(timeout=0.5)
    except queue.Empty:
        continue
    # ... rest of handler unchanged
```

In the `run()` method's `finally` block (where `self._running = False` is set), also set:
```python
self._cmd_stop.set()
```

**Verify:** After launching the dashboard, typing any command (e.g. `show rules`) and pressing Enter produces a response in the command pane within 2 seconds.

---

### Bug 10 — `kaia_dashboard.py` line 608: `len(engine.scanner)` — `yara.Rules` does not support `len()`

**Confirmed:** `SystemSecurityCollector.run()` at line 607–608:
```python
if engine.scanner:
    yara_rules_count = len(engine.scanner)
```
`yara.Rules` objects are not sequences and do not implement `__len__`. This raises `TypeError: object of type 'yara.Rules' has no len()` every 5 seconds, crashing the `SystemSecurityCollector` thread silently (caught by the outer `except Exception`).

**Fix:** Replace `len(engine.scanner)` with a file count from the rules directory — this is already used as the fallback on line 610:
```python
if engine.scanner:
    # yara.Rules has no len(); count the .yar files instead
    yara_rules_count = len([
        f for f in os.listdir(config.YARA_RULES_DIR) if f.endswith(".yar")
    ])
```

Remove the now-unreachable second `except` branch that does the same thing.

**Verify:** `SystemSecurityCollector` runs for 30+ seconds without any `TypeError` in logs. Pane 4 shows `Active Rules: N` where N matches the number of `.yar` files in `storage/threat_intel/rules/`.

---

## Part B — Run the test suite after all fixes

Run in this order:

```bash
python -m pytest tests/test_tier5_security.py -v
python -m pytest tests/test_negative_security.py -v
python -m pytest tests/test_advanced_security.py -v
python -m pytest tests/test_database_utils.py -v
python -m pytest tests/test_kaia_cli.py -v
python tests/verify_security.py
```

All must pass before doing anything else. Document any remaining failure with the exact error output — do not attempt to fix new failures without first reporting them.

---

## Part C — Final acceptance grep checks

Every one of these must return the stated result:

```bash
# No cognitive wiring anywhere
grep -rn "AffectiveState\|beliefs.json\|cognitive_wiring\|dream_cycle" . --include="*.py"
# Expected: zero matches

# No FIMDaemon instantiation in dashboard
grep -n "FIMDaemon()" kaia_dashboard.py
# Expected: zero matches

# logger defined at module level in dashboard
grep -n "^logger = " kaia_dashboard.py
# Expected: exactly one match

# log_security_event imported in honeypot
grep -n "from security.db import" security/honeypot.py
# Expected: one match, before line 171

# No hardcoded /home/ in ExecStart lines of service files
grep "ExecStart" scripts/kaia-policy-gate.service scripts/kaia-lockdown.service
# Expected: both lines use ${KAIA_PROJECT_DIR}, not /home/ekco

# Worker uses stop event not _running for liveness
grep -n "_cmd_stop\|cmd_stop" kaia_dashboard.py
# Expected: at least 3 matches (init, loop, cleanup)
```

---

## What is deferred — do not implement this cycle

- GeoLite2 MMDB download integration beyond what Bug 5 already wires
- DuckDB delta computation (Appendix C)
- Multi-agent or distributed architectures
- GVisor/Firecracker tiers
- Any new features not listed above
