# Kaia — Claude Handoff Document

**Original handoff:** 2026-09-18  
**Reviewed by:** Antigravity (Gemini) agent reviewing docs/, tests/, and source  
**Authoritative spec:** [`docs/master_plan.md`](docs/master_plan.md) — read this first  
**Pending ops doc:** [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md)  
**Last agent report:** [`docs/03_Gemini35_Execution_Report.md`](docs/03_Gemini35_Execution_Report.md)

**Status reconciliation:** 2026-10-22 — read `docs/master_plan.md` §10.5 and `docs/NEXT_STEPS.md` for current status. This is a historical handoff, not evidence of current host deployment.

---

## What Kaia Is

Kaia is a **hardened security daemon** for Arch Linux. It is **not** an AI chat assistant or a Tamagotchi. It:

- Runs as a `root` systemd service (`kaia-policy-gate.service`)
- Exposes a Unix socket (`/run/kaiacord/policy_gate.sock`) that validates signed capability tokens before executing any host action
- Monitors filesystem integrity (fanotify/watchdog), network traffic (raw sockets), honeypot decoys, and privilege escalation events (eBPF telemetry)
- Enforces a capability lattice: `GLOBAL ∩ WORKSPACE → EFFECTIVE`
- Runs all agent-requested scripts inside Bubblewrap sandboxes with cgroup limits
- Presents telemetry via the loopback web HUD by default and the legacy curses TUI (`kaia_dashboard.py --tui`)

**Operator UI:** local loopback web HUD by default; `kaia_dashboard.py --tui` keeps the curses dashboard available. They share the collector/snapshot architecture.  
**`main.py`** is a debugging fallback only, not for normal use.

---

## Features Present in the Codebase — review snapshot 2026-10-22

The following code paths were present in the reviewed source as of 2026-10-22. Their presence does not imply that the privileged host deployment or kernel integrations have been verified end to end:

### Security Core
- [x] Policy Gate Unix socket server with length-framed IPC protocol
- [x] Capability token generation (`KAIA_CAPABILITY_TOKEN_SECRET` env var, HMAC-signed, expiring)
- [x] Capability lattice intersection enforcement (INV-007)
- [x] Fail-closed on any socket/parsing error (INV-001)
- [x] Append-only audit DB (`security_events.db`) — agent has zero write/delete capability (INV-003)
- [x] Audit ledger (`audit_ledger.json`) isolated from agent working tree (INV-008)
- [x] Host executor with **no `shell=True`** ever — all commands via absolute-path arrays (INV-005)
- [x] Path blocklist for `write_file`: blocks `.py`, `.sh`, `.env`, `.git`, `core/`, `security/`, `tests/`, `scripts/`, `storage/` (INV-010)
- [x] Bubblewrap sandbox masking `.env` to `/dev/null` and `storage/` to `tmpfs` (INV-011)
- [x] Telemetry sanitizer (character allowlist, string truncation) for all incoming telemetry (INV-004)
- [x] SQLite tamper detection uses logical content-hash (not raw byte prefix) — eliminates WAL false positives
- [x] Tamper detection watchlist expanded from 8 to 20 files (including `tamper_detection.py`, `db.py`, `kaia-lockdown.sh`)
- [x] Lockdown circuit breaker: capped at 3 per hour; latching prevents single unchanged file re-firing
- [x] NFT ruleset snapshotted to `/var/lib/kaia/pre-lockdown.nft` before lockdown; `kaia-unlock.sh` lifts it
- [x] `KAIA_TAMPER_ENFORCE=0` monitor-only mode (logged CRITICAL at startup)
- [x] `.env` tampering now triggers lockdown (was silently exempt before)
- [x] `kaia-restore-blocks.service` — folds approved block/unblock ledger events in order before the Policy Gate accepts requests
- [x] Panic recovery: `~/kaia-panic-unlock.sh` (standalone, no repo dependency)

### Firewall / Network
- [x] `block_ip` writes to dedicated `inet kaia_block` nftables table at hook priority -10 (survives `ufw reload`)
- [x] IPv6 targets handled (`ip6` selector — `ip saddr` with v6 was a syntax error)
- [x] `execute_mitigation` reconciled with live firewall (old implementation wrote to `ip filter` = ufw's table, silently discarded)
- [x] `get_default_interface()` checks `/sys/class/net` for actual existence before trusting route result

### Dashboard Stage 1 (read-only panes)
- [x] Four security panes: Audit Log, Threat Intel, Containment & Sentinel, System Security
- [x] `AuditLogCollector` polls `security_events.db` (250ms) and `audit_ledger.json` (500ms)
- [x] Ping/Service/Telemetry collectors targeting Kaia services (Ollama, Policy Gate socket, etc.)
- [x] EPS meter (events-per-second)
- [x] `s` key toggles SECURITY-only filter
- [x] Pane 2: active firewall blocks, Shodan enrichment, recent blocked IPs
- [x] Pane 4: capability token counts, firewall rule stats, honeypot status, Policy Gate health

### Dashboard Command Interface (Stage 1 commands implemented)
- [x] `block <ip>` — sends `block_ip` intent to Policy Gate
- [x] `unblock <ip>` — requests exact Kaia-managed rule removal via capability-gated Policy Gate action
- [x] Web command console returns structured one-shot responses; incremental response streaming remains unfinished
- [x] `show rules` — queries nftables via Policy Gate diagnostics
- [x] `restart <service>` — sends `restart_service` intent
- [x] `list recent blocks` — queries `security_events.db`
- [x] `check threat <ip>` — enriches via Shodan InternetDB + CVE lookup
- [x] `show audit --since <Nh>` — queries DB by time window
- [x] `lockdown` — confirmation-gated emergency lockdown
- [x] `add rule <name> <indicator> [mitre:<id>]` — YARA rule compilation via Policy Gate
- [x] `show assets` — queries `PassiveDiscoveryEngine` for LAN assets
- [x] `show fim alerts` — queries `/var/lib/secdaemon/fim_audit.db`

### Threat Intel
- [x] InternetDB lookup with 7-day SQLite cache, per-process serialized request rate limiting, and negative caching for authoritative 404 results
- [x] CVE detail lookups
- [x] IP reputation scoring

### Systemd Hardening
- [x] `User=root` (was `User=ekco`, broke lockdown path)
- [x] Historical unit capability set was reviewed and later tightened; confirm the active unit before changing it. The 2026-09-08 host note recorded CAP_SYS_ADMIN/CAP_DAC_OVERRIDE dropped unless current kernel probes justify them.
- [x] `ProtectHome=read-only`, `PrivateDevices=true`
- [x] `KAIA_CAPABILITY_TOKEN_SECRET` loaded from `/etc/kaia/secret.env` (not in unit file)
- [x] Honeypot files pre-created on host; daemon truncates (not deletes) on shutdown
- [x] `vulkaninfo --json` CWD pollution fixed (runs in temp dir)
- [x] `cgroup` wrapper: `systemd-run --scope` selects system vs user scope correctly for root daemon
- [x] `execute_script` checks `SCRIPT_ALLOWLIST` itself (was trusting Policy Gate entirely)
- [x] Service allowlist is now a single `config.SERVICE_RESTART_ALLOWLIST` (no drift between files)
- [x] `StartLimitIntervalSec`/`StartLimitBurst` moved from `[Service]` to `[Unit]`

### Other
- [x] `security/__init__.py` exists (standard package, not implicit namespace)
- [x] All 11 `datetime.utcnow()` calls migrated to `datetime.now(UTC)` (Python 3.14 compat)
- [x] Legacy files `security/cognitive_wiring.py` and `storage/cognition/beliefs.json` removed
- [x] `.gitignore` excludes `/core/personal_context` and `/core/storage`
- [x] FIM daemon degrades across 4 mask tiers on EINVAL (was retrying same failing mask)
- [x] FIM hashes and scans the same bounded byte snapshot read from the original event FD; this is notification-mode observation, not pre-execution blocking
- [x] `sys_enter_unlinkat` eBPF events are consumed and exposed in the syscall waterfall as attempts
- [x] YARA generation escapes string literals; sandbox validation fails closed when prerequisites are missing
- [x] Passive pacman transaction monitor for completed package operations, with persistent cursor and rotation handling
- [x] Strict CSP and GitHub Actions regression workflow (latest recorded main run: 156 passed)
- [x] Trusted baseline via `scripts/kaia-baseline.sh` -> `/var/lib/kaia/baseline.sha256`

---

## What Is NOT Done / Still Needs Work — updated 2026-10-22

### 1. Incremental command-response streaming

Interactive command entry is built in core/web/static/js/terminal.js; /api/command
dispatches known commands through the Policy Gate and returns structured output.
The telemetry WebSocket streams snapshots, but command output itself is still one
completed response. Long-running diagnostics and sandboxed operations do not publish
partial output chunks while executing. Completing this needs a bounded, framed
streaming protocol, cancellation/timeout behavior and UI back-pressure tests.

### 2. Execution-blocking FIM

The current FIMDaemon uses FAN_CLASS_NOTIF. The October hardening keeps the
original event FD open and scans/hashes the same captured byte snapshot (bounded to
64 MiB), but notification mode does not block the originating process pending a
verdict. A separate permission-event design is required if pre-execution prevention
is a goal.

### 3. Expand and verify kernel telemetry

The sys_enter_unlinkat perf callback and waterfall lane now exist. The record is
captured at syscall entry and means attempt observed, not confirmed successful
deletion. More hook coverage, successful-operation correlation and verification
against the deployed kernel/BCC/toolchain remain unfinished.

### 4. Network discovery gaps

NetBIOS Name Service parsing is still a target requirement and is not implemented.
The existing passive discovery only sees traffic actually observed on the configured
interface; it is not a complete asset inventory or active scanner.

### 5. Broader supply-chain and resilience features

Basic pacman-log transaction observation is implemented. Still missing are AUR build
input tracking, repository/metadata mutation detection, dependency-graph drift and
package signature/installed-content attestation. Other Tier 5 backlog items include
behavioral baselines, alert-confidence scoring, secrets-exposure detection, backup
restorability checks, independent service/data-freshness supervision and expanded
GeoLite2/CVE/passive-DNS/historical threat-intelligence stores.

### 6. Target-host deployment / acceptance tests

The deployment section below reflects an old host checkpoint, not a current live
probe. Before relying on Kaia, verify the actual systemd unit, ExecStart,
WorkingDirectory, state directory and trusted integrity baseline, then run the
firewall block → UFW reload → reboot replay → unblock → second reboot test described
in docs/NEXT_STEPS.md. FIM marks and BCC probes must be checked on the target kernel.

## Architecture Quick Reference

| Component | File | Status |
|-----------|------|--------|
| Policy Gate daemon | `security/policy_gate.py` | Complete |
| Host executor + sandbox | `security/host_executor.py` | Complete |
| Pydantic schemas | `security/schemas.py` | Complete |
| Tamper detection | `security/tamper_detection.py` | Complete |
| FIM daemon | `security/fim_daemon.py` | Complete (degrades gracefully) |
| eBPF telemetry | `security/ebpf_telemetry.py` | Complete (passive + fallback sampler) |
| Network discovery | `security/network_discovery.py` | Complete |
| Threat intel | `security/threat_intel.py` | Complete (Shodan + CVE) |
| YARA rule engine | `security/rule_engine.py` | Built; yara optional |
| Honeypot | `security/honeypot.py` | Complete |
| Telemetry sanitizer | `security/telemetry_sanitizer.py` | Complete |
| Audit DB | `security/db.py` | Complete |
| Cyber-Sentinel Web Bridge | `core/web_bridge.py` | **Complete & Verified (Zero-Trust)** |
| Cyber-Sentinel Web HUD | `core/web/static/` | **Complete & Verified (HTML5/CSS3/Canvas)** |
| Dashboard CLI / TUI | `kaia_dashboard.py` | Complete (Dual-mode: Web HUD default, `--tui` fallback) |
| Config | `core/config.py` | Complete |
| CLI intent generator | `core/kaia_cli.py` | Complete |
| Systemd unit | `scripts/kaia-policy-gate.service` | Written; NOT installed to /opt |
| Installer | `scripts/install_services.sh` | Written; NOT run |
| Baseline script | `scripts/kaia-baseline.sh` | Written; NOT run |
| Block restore unit | `scripts/kaia-restore-blocks.service` | Written; NOT installed |

---

## Cyber-Sentinel UI Upgrade: Completed Groundwork & Claude Opus 5.5 Handoff

As specified in [`2026-10-07-kaia-hardened-security-ui-upgrade-design-spec.md`](2026-10-07-kaia-hardened-security-ui-upgrade-design-spec.md), Kaia's interface has been upgraded from a terminal-only curses TUI into a **Zero-Trust Cyber-Sentinel Operations Center (SOC)** web dashboard.

### 1. Division of Responsibilities

* **Antigravity (Completed Groundwork)**:
  - Built the zero-trust async Python web server daemon (`core/web_bridge.py` and `core/web/server.py`) using `aiohttp`.
  - Enforced all **Zero-Trust Security Invariants**: loopback-only binding (`127.0.0.1`, rejecting `0.0.0.0`), strict Content Security Policy (`default-src 'self' ...`), read-only WebSocket outbound push, air-gapped local asset serving, and deterministic Policy Gate command dispatch.
  - Implemented immutable `KaiamonSnapshot` JSON serialization with live hardware telemetry (CPU, RAM, GPU, thermals), service status LED tracking, and active token details.
  - Implemented the userspace process/file fallback sampler (`ProcessSyscallSampler`) when BCC kernel headers are absent.
  - Implemented the dual-mode CLI argument parser in `kaia_dashboard.py` (`--web` by default, `--tui` / `--curses` legacy terminal).
  - Designed the **Cyber-Sentinel UI Design System** (`core/web/static/css/sentinel.css`): CRT scanline overlay, 32px cyber grid, tactical corner brackets, glowing neon accents (`#00f0ff`, `#3dff8a`, `#ffe45c`, `#ff3860`, `#a259ff`), LED meters.
  - Implemented the **process-event waterfall** (`core/web/static/js/waterfall.js`) as a Canvas timeline of actual `ebpf_waterfall` snapshot events, including `sys_enter_execve` plus the now-wired `sys_enter_openat2` perf-buffer path, with disposition colors and a cursor inspector. It intentionally does not stream every high-volume `openat` call. Animation is drawn at display cadence; underlying snapshot delivery remains 5 Hz, and `observed` events are not labeled safe.
  - Implemented the **LPS oscillogram** in `core/web/static/js/charts.js` (there is no separate `oscillogram.js` module): rolling 60-second Canvas curve, optional gradient area fill, adaptive scale, and visible 35 LPS surge threshold. The plot can redraw at ~60 FPS, but new measurements still arrive at the bridge's 5 Hz rate.
  - Implemented the **passive-discovery network radar** (`core/web/static/js/radar.js`): rotating sweep, observed LAN asset nodes, recent blocked IP nodes, and reputation-score bars from collector data. Node placement is a visual distribution, not geographic topology; scores are omitted when source data is absent.
  - Implemented the **capability token ledger** (`core/web/static/js/tokens.js`): compact cards for token fingerprint, scope and target, with SVG TTL rings based on snapshot expiry timestamps.
  - Built the **Interactive Command Terminal Console** (`core/web/static/js/terminal.js`): Monospace terminal prompt (`KAIA_SENTINEL:~$ `) with quick-action buttons, command history, and structured response rendering.
  - Built the **Sentinel Avatar State Machine Controller** (`core/web/static/js/operative.js`): Manages the 7 security states (`SentinelWatch`, `PolicyEvaluating`, `AnomalyDetected`, `ContainmentAlert`, `LockdownMode`, `AuditSurge`, `SleepyIdle`), updating biometric HUD (Pulse BPM, Conduit Glow, Defense Stance), and exposing the clean mount hook for Opus 5.5.
  - The original 55-test result was recorded at handoff time; it has not been re-established by this UI audit.

* **Still Remaining / Not Yet Verified**:
  - The WebGL2 codec portrait is now mounted automatically by `SentinelAvatarController` and reacts to all seven sentinel states through `KaiaAnimator` presets. This is a rigged **portrait**, not the specified full-body transparent crystal-polymer android. Full-body artwork/mesh or a full-body sprite sequence, including limbs, optical spine and material treatment, remains unfinished; the portrait assets cannot satisfy that part of Section 5.
  - Run browser-level verification for all seven mood transitions, WebGL2 failure fallback, resize handling, resource use, and the dashboard under strict CSP.

  - Orbitron and JetBrains Mono are referenced as local-first CSS font choices, but the repository has no bundled WOFF/TTF/OTF assets. It remains visually functional offline with system-font fallbacks, but Section 3's exact locally-packaged-font requirement is not met.
  - Optional Web Audio alert/tick synthesizer remains unimplemented.

---

### 2. Integration Hooks Ready for Claude Opus 5.5

#### DOM Mounting Target:
Inside [`core/web/static/index.html`](../core/web/static/index.html). The image fallback from the original example has since been removed; the current default renderer is a rigged portrait mounted into this container:
```html
<div class="operative-stage" id="kaia-operative-stage">

  
  <!-- DEDICATED OPUS 5.5 RENDER MOUNT -->
  <div id="kaia-character-container"></div>

  <div class="reticle-hud"><div class="reticle-center"></div></div>
  <div class="conduit-glow-border" id="conduit-glow"></div>
</div>
```

#### JavaScript API Hook (`core/web/static/js/operative.js`):
Opus 5.5 can mount their custom renderer directly into `window.KaiaAvatar`:
```javascript
window.KaiaAvatar.mountAnimationViewport({
  init: function(containerElement, initialState) {
    // 1. Initialize Three.js WebGL Scene / 2D WebM loop / Canvas Sprite engine
    // containerElement is document.getElementById('kaia-character-container')
  },
  onStateTransition: function(nextState, oldState) {
    // 2. React to security state changes:
    //    'SentinelWatch'   -> Idle gentle breathing, soft emerald glow
    //    'PolicyEvaluating'-> Focused, cyan upward fiber conduits
    //    'AnomalyDetected' -> Skeptical, head tilt, amber micro-sparks
    //    'ContainmentAlert'-> Defensive recoil, angry stern scowl, crimson glow
    //    'LockdownMode'    -> Combat shield stance, strobe red volumetric lights
    //    'AuditSurge'      -> Binary matrix stream wash down crystal torso
    //    'SleepyIdle'      -> Low resource standby, dim conduits, 35 BPM
  }
});
```

---

### 3. How to Run & Verify the Cyber-Sentinel UI

1. **Launch the Web Operations Center HUD (Default)**:
   ```bash
   python3 kaia_dashboard.py
   # OR standalone:
   python3 -m core.web_bridge --host 127.0.0.1 --port 8765
   ```
   Open `http://127.0.0.1:8765/` in your browser to view the Cyber-Sentinel Operations Center HUD.

2. **Launch the Legacy Terminal Curses TUI**:
   ```bash
   python3 kaia_dashboard.py --tui
   ```

3. **Run the Full Test Suite**:
   ```bash
   pytest tests/test_web_bridge.py tests/test_tier5_security.py tests/test_advanced_security.py -v
   ```

---

## Known Credentials / Security Notes

| Issue | Status |
|-------|--------|
| `KAIA_CAPABILITY_TOKEN_SECRET` was committed in 4 git commits to a public repo | Rotated 2026-09-08. Old value rejected by installer outright. History NOT scrubbed (scrubbing doesn't un-publish a value that was already public). |
| `sdb` backup drive reportedly had 2 pending sectors | Historical note from the 2026-09-18 handoff. Re-check device health before treating it as current. |

---

## Tier 5 Stretch Goals (Future / Optional)

These are a mixture of remaining roadmap items and optional visual polish. Current implementation status is tracked in `docs/master_plan.md` §10.5; the list below is not a claim that every item is untouched:

- [ ] Expanded eBPF coverage and target-kernel validation
- [ ] Execution-blocking FIM, plus broader path/rule coverage
- [ ] NetBIOS in passive LAN discovery
- [ ] Broader supply-chain monitoring (AUR/repositories/dependency graphs)
- [ ] Behavioral baselines, alert-confidence scoring and secrets exposure detection
- [ ] Backup restorability and independent service/data-freshness monitoring
- [ ] DuckDB analysis, richer GeoLite2/CVE and passive-DNS datasets
- [ ] WebGL2 fallback, resize/resource checks, bundled local fonts and optional Web Audio alert synthesis
- [ ] Full-body transparent crystal-polymer Kaia artwork/mesh; the current renderer is a rigged portrait

