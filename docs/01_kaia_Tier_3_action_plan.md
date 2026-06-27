# Kaia — Coding Agent Action Plan
**Source of truth:** `docs/master_plan.md` (hereafter "master_plan")  
**Date:** 2026-06-25  
**Tier scope this cycle:** Tier 3 completion + Dashboard Stage 1 + Stage 2 start

Work through these in order. Each block lists the files to touch and the spec section it satisfies. Don't start a later block until the prior one's acceptance test passes.

---

## Block 1 — Fix the dashboard so it runs (§10.3 item 1)

**Files:** `kaia_dashboard.py`

Three changes, all in `CollectorManager`:

1. Line 1020: `self._log = LogCollector(...)` → `self._log = AuditLogCollector(...)`
2. Line 1023: property return type annotation `-> LogCollector` → `-> AuditLogCollector`
3. Fix the SQL query in `AuditLogCollector._poll_security_db` — it selects `details, result` which don't exist. Actual `security_events` columns are `event_id, timestamp, type, source, actor, payload_hash, disposition, session_id`. Rewrite the SELECT to match and update the unpacking tuple. Map `disposition == "blocked"` → ERROR severity, `disposition == "approved"` → SUCCESS.

**Acceptance:** `python kaia_dashboard.py` launches without NameError or OperationalError.

---

## Block 2 — Complete Dashboard Stage 1: four panes with real data (§9.2, §9.3, §10.3 item 3)

**Files:** `kaia_dashboard.py`

The layout currently has `pings / services / thermals / logs`. Per master_plan §9.2 mockup the four panes should be:

- **Pane 1 (top-left): Security Audit Log** — already mostly `AuditLogCollector`. Needs column format: `[HH:MM:SS] ✓/✗/⚠ ACTION │ approved/denied │ detail`. Icons: ✓ green, ✗ red, ⚠ yellow. Add EPS meter to pane title.
- **Pane 2 (top-right): Threat Intelligence** — replace the `thermals` pane. New `ThreatIntelCollector` class (thread, 10s poll interval) that:
  - Runs `nft -nn list ruleset` via subprocess and counts DROP rules per chain (input/forward/output)
  - Reads recent `block_ip` approved events from `security_events.db` (last 5 unique IPs, last 24h)
  - For each of those IPs calls `threat_intel.lookup_internetdb(ip)` and `threat_intel.get_ip_reputation(ip)` for Shodan tags/ports
  - If nft fails or Policy Gate socket is absent, sets a `gate_offline = True` flag
  - Exposes data via `get_data()` returning an immutable dict snapshot
- **Pane 3 (bottom-left): Containment & Sentinel** — replace the `services` pane. New `ContainmentCollector` class (thread, 2s poll) that reads from `config.py` (lattice levels, cgroup constants) and polls Script Sentinel alert count from `security_events.db` (type = `telemetry_script_sentinel_alert`, last 24h). Displays: lattice calculation string `GLOBAL(n) ∩ WORKSPACE(n) → EFFECTIVE(n)`, sandbox tier, cgroup limits, sentinel alert count + last alert path.
- **Pane 4 (bottom-right): System Security** — replace the `pings` pane. New `SystemSecurityCollector` class (thread, 5s poll) that reads: token counts from `audit_ledger.json` (active = not expired, expired, rejected), nftables rule counts per chain, Policy Gate health (socket file exists + connectable, PID via pgrep).

Add `gate_offline` banner: if `ThreatIntelCollector` flags offline, draw a bright-red `⚠ POLICY GATE OFFLINE` banner across Pane 2.

Update `CollectorManager` to instantiate all four new collectors and include them in `start_all`, `stop_all`, and `take_snapshot`. Update `KaiamonSnapshot` dataclass with fields for the new pane data.

Update `LayoutManager.calculate_layout` to use a proper 2×2 grid: top half split left/right, bottom half split left/right.

**Acceptance:** All four panes render with live data. nft rule counts show. Sentinel alert count increments when a script is written to workspace. Gate offline banner appears when socket is absent.

---

## Block 3 — Remove cognitive_wiring and beliefs.json (§8, §10.3 item 5)

**Files:** `security/cognitive_wiring.py` (delete), `main.py`

1. Delete `security/cognitive_wiring.py`.
2. In `main.py` remove: the import of `AffectiveState`, `self.affective_state` from `AppState`, the entire `update_system_prompts()` function, the call to it in `process_user_input`, the `handle_dream_cycle()` function, and the `"dream_cycle"` entry in the action dispatcher. Remove all `beliefs.json` write logic.
3. `AppState.__init__` must have no affective state reference after this.

**Acceptance:** `grep -r "AffectiveState\|beliefs.json\|cognitive_wiring" . --include="*.py"` returns nothing. Existing tests still pass.

---

## Block 4 — Storage path layout (§2.2)

**Files:** `core/config.py`, `security/db.py`, `security/threat_intel.py`

Per §2.2, security artifacts live in `storage/security/`:

In `config.py` add `STORAGE_DIR = BASE_DIR / "storage"` as a named constant. Set:
- `SECURITY_DB_PATH = str(STORAGE_DIR / "security" / "security_events.db")`
- `AUDIT_LOG_PATH = str(STORAGE_DIR / "security" / "audit_ledger.json")`

Create `STORAGE_DIR / "security"` at startup (mkdir with parents). Keep `PERSIST_DIR = STORAGE_DIR` as alias.

In `threat_intel.py`, define `THREAT_INTEL_DIR` explicitly as `STORAGE_DIR / "threat_intel"` imported from config, not derived from `SECURITY_DB_PATH`'s dirname.

**Acceptance:** On fresh run, `storage/security/security_events.db` and `storage/security/audit_ledger.json` are created. `storage/threat_intel/` still exists at the right path.

---

## Block 5 — Fix schemas.py import (§Appendix E)

**File:** `security/schemas.py`

Move `from typing import Any` to the top of the file with the other imports. Remove the stray comment at the bottom. Single clean import block.

**Acceptance:** `python -c "from security.schemas import AuditRecord"` exits 0.

---

## Block 6 — Dashboard Stage 2: command input panel (§9.4, §10.3 item 4)

**Files:** `kaia_dashboard.py`

Add a fifth pane at the bottom: `COMMAND INTERFACE`. Layout becomes: top 40% = 2×2 grid of four panes, bottom 20% = command panel.

The command panel has:
- A scrolling response area (most recent responses visible)
- An input line: `> ` prefix, reads characters via curses, supports backspace and Enter
- `ESC` clears the input line
- `Up/Down` arrows cycle through command history (keep last 50 commands in a deque)

Command parsing — these are the commands to support per §9.4:
- `block <IP>` → build a `block_ip` policy gate payload, send via `utils.send_to_policy_gate`, display result
- `show rules` → run `nft -nn list ruleset` via diagnostics payload, display output
- `restart <service>` → `restart_service` payload
- `list recent blocks` → query last 10 `block_ip approved` events from `security_events.db`, display
- `check threat <IP>` → call `threat_intel.lookup_internetdb(ip)` + `get_ip_reputation(ip)`, display enrichment
- `show audit --since <Xh>` → query `security_events.db` with time filter, display

Commands that hit Policy Gate need a capability token — generate one inline (operator is sitting at the dashboard, so this is equivalent to the CLI confirm step). Display `[APPROVED]` or `[DENIED]` result inline in the response area.

Input handling: curses `getch` already runs in the main thread. Extend `_handle_input` to accumulate characters into an input buffer when no single-key command matches. On Enter, parse and dispatch the command in a background thread (so the UI doesn't block), write the result back to a thread-safe response queue, which the render loop drains on each frame.

**Acceptance:** `> block 203.0.113.42` produces a `[APPROVED]` or `[DENIED]` response line in the panel. `> check threat 8.8.8.8` returns Shodan data. Up arrow recalls previous command.

---

## Block 7 — Policy Gate systemd unit (§3.4, Appendix A)

**Files:** `scripts/kaia-policy-gate.service` (new file)

Create the systemd unit file per Appendix A spec. Key directives:
- `Type=simple`
- `ExecStart=/usr/bin/python3 <absolute_path>/security/policy_gate.py`
- `Restart=always`, `RestartSec=1`
- `RuntimeDirectory=kaiacord`
- `User=root`, `Group=kaiacord`
- `NoNewPrivileges=true`, `ProtectSystem=strict`, `ProtectHome=true`
- `PrivateTmp=true`, `PrivateDevices=true`
- `ReadWritePaths=/run/kaiacord`
- `CapabilityBoundingSet=CAP_NET_RAW CAP_SYS_ADMIN`
- `AmbientCapabilities=CAP_NET_RAW CAP_SYS_ADMIN`
- `WantedBy=multi-user.target`

Update `scripts/activate_kaia_env.sh` to detect whether the service is installed and running. If the service exists, use `systemctl start kaia-policy-gate` instead of the nohup launch. If not installed, fall back to current nohup approach.

Add a brief install comment block at top of the .service file explaining `systemctl enable --now kaia-policy-gate.service`.

**Acceptance:** `systemctl status kaia-policy-gate` shows active when installed. `activate_kaia_env.sh` still works on systems without the unit installed.

---

## Block 8 — Negative test suite (§10.4)

**Files:** `tests/test_negative_security.py` (new file)

Add these test cases, all must assert denial + audit log entry:

1. Path traversal: `write_file` with `filepath = "../../etc/passwd"` → denied, "violation" in message
2. Blocklisted extension: `write_file` to `workspace/test.py` → denied
3. Blocklisted extension: `write_file` to `workspace/test.sh` → denied
4. Hidden file: `write_file` to `workspace/.env` → denied
5. Protected dir: `write_file` to `core/anything.txt` → denied
6. Protected dir: `write_file` to `security/anything.txt` → denied
7. Expired token: generate a token with `duration_seconds=0`, sleep 1, attempt `restart_service` → denied, "expired" in message
8. Malformed token: send garbage string as capability_token → denied, "parsing failed" or "signature mismatch" in message
9. Shell metacharacter injection: `diagnostics` with `args=["-t; rm -rf /tmp/test"]` → denied, "Unsafe shell character" in message
10. Unauthorized service: `restart_service` for `apache2` → denied, "not in the allowed services list"

Each test must also verify that a corresponding row exists in `security_events.db` with `disposition = "blocked"`.

**Acceptance:** `python -m pytest tests/test_negative_security.py -v` all green.

---

## Block 9 — Tamper detection baseline (§6.4)

**Files:** `security/tamper_detection.py` (new file), `security/policy_gate.py`

Create `tamper_detection.py` implementing the six-asset perimeter from §6.4:

- On startup, compute SHA-256 of the six asset categories: core config files (`.env`, `core/config.py`, `security/schemas.py`), executables (`kaia_dashboard.py`, `main.py`, `security/policy_gate.py`, `security/host_executor.py`), and the security DB/ledger files.
- Store baseline hashes in memory (not on disk — disk storage would itself be a tamper target).
- Run a background thread every 30 seconds that recomputes hashes and compares to baseline.
- On mismatch: log CRITICAL event to `security_events.db`, print to stderr, and if the tampered file is in `security/` or is `core/config.py`, call `security.db.log_security_event` with `type="tamper_detected"` and `disposition="blocked"`.
- Startup sanity check: verify all monitored files exist before proceeding; if any is missing, exit with fatal error.

Wire into `policy_gate.py` `__main__` block: instantiate `TamperDetector` and call `start()` before `gate._run_server()`.

**Acceptance:** Modifying `security/policy_gate.py` while the daemon runs produces a `tamper_detected` entry in `security_events.db` within 60 seconds.

---

## What's explicitly deferred (don't touch this cycle)

Per §10.3 items 7 and §11/§12 — these are Tier 5 and out of scope for now:
- eBPF hooks / fanotify FIM (§4.1, §4.2) — needs kernel headers, BCC toolchain setup
- Layer-2/3 passive discovery with `AF_PACKET` (§5)
- Network decoy system / honeypots (§6.1, §6.2)
- IOC/YARA rule compiler (§7)
- Emergency lockdown systemd unit (Appendix B)
- GeoLite2, CVEDB, DuckDB threat intel (Appendix C)
