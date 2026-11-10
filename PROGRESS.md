# Progress log

Handoffs between autonomous maintenance runs. Append a section per run; never rewrite one.

## Run 2026-11-09 21:40 UTC

**HEAD at start:** 2a65e5a
**HEAD at end:** 4de7cd7 (four stacked PRs were left open; all merged in the next run)
**Batches landed this run:** 1 merged (#97); #98, #99, #100, #107 merged in the next run

### Merged
- #97 — HUD and curses dashboard show a lockdown in force: state from `table inet kaia_lockdown` in the ruleset, not the oneshot unit's state; "unknown" when the gate cannot be asked — evidence: CI-passing

### Open, stacked (merge in this order; each retargets to `main` when its base merges)
- #98 — eBPF: gains of root from `commit_creds()` replace the setuid/setreuid syscall-entry probes (268 false `privilege_escalation` events in 8 days on the host); local BPF compile check; rate cap on routine setuid-exec records — evidence: test-covered, compile-checked locally, CI-passing; **not loaded on the host**
- #99 — Control health: every control reports active/degraded/off/failed with heartbeats; `kaia-health-check.timer` checks it from outside the gate; HUD control strip; the tamper loop survives an exception — evidence: test-covered, run unprivileged end to end; **not deployed**
- #100 — HUD token cards read redacted ledger rows (they could not show any token issued since the redaction change) and no signature reaches the browser; bounded/incremental ledger reads (the HUD parsed 18 MB at 5 Hz) — evidence: test-covered, counts match the old parser on the host ledger
- #107 — docs: ROADMAP, master_plan §10.5/§11, README for the stack

**Why not merged:** the auto-mode permission classifier refused `gh pr merge` on #98 (a security-critical change with no independent review). `docs/security_review_process.md` expects a non-author approval for these paths, although the `main` ruleset requires 0. They need the owner's review.

### Issues closed
- none (there were no open issues at start)

### Issues filed
- #101 — deploy and host-verify #98–#100 — type: host-verification
- #102 — `/root/.ssh` decoy impossible under `ProtectHome=tmpfs` — type: code (owner decision)
- #103 — one failed probe attach disables every eBPF probe — type: code
- #104 — BCC runtime compile breaks after a kernel upgrade until reboot; libbpf CO-RE — type: code (design)
- #105 — every dashboard nft poll is audited (~17k ledger rows/day) — type: code
- #106 — openat2 ring buffer drops — type: code (low)

### Docs updated
- CHANGELOG (in each PR); OPERATIONS: privilege gains, control health (#98, #99); DEPLOYMENT_CHECKLIST: compile check, privilege-probe host test, control-health check (#98, #99); ROADMAP, master_plan, README (#107). New host claims are limited to what was read on the host this run (journal, kernel config, LSM list, BTF, pacman mtree).

### Host observations (read-only, 2026-11-09)
- `/opt/kaia` matches `main` @ 2a65e5a for `security/`, `core/` and `scripts/`; the gate has been up since 2026-11-03 11:14; eBPF loaded; honeypot inode watches 4 of 4; the privilege watch is active; FIM falls back to watchdog (no `CAP_SYS_ADMIN`); `kaia-restore-blocks` succeeded at boot (no blocks to restore).
- The audit ledger is 18 MB / 50,107 rows, 49,890 of them dashboard `diagnostics` polls (#105).
- BPF LSM active, BTF present, `CONFIG_FANOTIFY_ACCESS_PERMISSIONS=y`, Landlock active: inputs to the execution-blocking FIM design.

### In progress / blocked
- #98–#100 and #107: blocked on owner review and merge, not on code. Branches `fix/privilege-gain-probe`, `feat/control-health`, `fix/hud-ledger-reads`, `docs/roadmap-2026-11`.
- #101: host-verification blocker; needs sudo on the deployed host, after the merges.

### Baseline for next run
- `scripts/run_ci_tests.sh`: green on `main` (412 tests with #97); 484 at the top of the stack
- CI on main (`security-regressions.yml`): green
- Lint/type: not run (the repo has no lint/type gate)

### Recommended first action next run
- If #98–#100 are merged: verify CI on `main`, then do #103 (partial eBPF attach reported as degraded); it is code-only and builds on control health. If they are still open, address review comments on them; do not stack more work on unmerged security branches. Do not attempt #101 without the deployed host.

## Run 2026-11-09 23:30 UTC

**HEAD at start:** 4de7cd7
**HEAD at end:** 770ed56 (+ this docs PR)
**Batches landed this run:** 10 PRs merged

### Merged
- #98 — eBPF: gains of root from `commit_creds()`; rate cap on routine records — evidence: CI-passing, compile-checked locally; not loaded on host
- #99 — Control health + `kaia-health-check.timer`; tamper loop survives exceptions — evidence: CI-passing; not deployed
- #100 — HUD token cards read redacted rows; bounded/incremental ledger reads — evidence: CI-passing; counts match on host ledger
- #107 — docs: roadmap/status refresh with host-kernel research — evidence: host facts read this run
- #109 — eBPF kprobes attached individually, partial load = degraded; openat2 ring 1 MiB (closes #103, #106) — evidence: CI-passing, compile-checked locally
- #110 — installed-content attestation against pacman mtree, every 6 h, alert-only; `kaia_package_attest.py` — evidence: CI-passing; run read-only on host (155 files, clean)
- #111 — dashboard: one cached ruleset query per 10 s (~3x fewer audited polls; refs #105) — evidence: CI-passing
- #112 — alert levels for every event type; feed + HUD critical/high counts — evidence: CI-passing
- #113 — secret-exposure self-check (live secret + credential formats, values never recorded) — evidence: CI-passing; host scopes scanned read-only (clean)
- #114 — pacman monitor degraded while pacman.log unreadable — evidence: CI-passing

### Issues closed
- #103, #106 (by #109)

### Issues filed
- none new; #101 updated with the attestation and secret-scan host checks

### Docs updated
- CHANGELOG (each PR), OPERATIONS (attestation, secrets), DEPLOYMENT_CHECKLIST (attestation), ROADMAP, master_plan §10.5/§11, README. New features are stated as test-covered and not host-verified.

### In progress / blocked
- #101 — host verification of everything above. **Host-verification blocker**: needs sudo on the deployed host. Deploy order: `sudo scripts/install_services.sh`, review, `sudo /opt/kaia/scripts/kaia-baseline.sh` **before the next boot** (new protected modules: control_health, package_attestation, secret_exposure, alert_levels).
- #102 — `/root/.ssh` decoy: owner decision.
- #104 — libbpf CO-RE: design-sized; not started.
- #105 — partly addressed (#111); summarising repeated read-only diagnostics at the gate is still open.
- #108 — closed, superseded by this docs PR.

### Baseline for next run
- `scripts/run_ci_tests.sh`: green (542 tests)
- CI on main (`security-regressions.yml`): green
- Lint/type: no gate in the repo

### Recommended first action next run
- Check whether #101 has host results; act on any finding. Then code items: #105 (gate-side summary rows for identical read-only diagnostics, keeping every mutation and denial individual), signed-package attestation (ROADMAP Tier 5 #2), alert reasons in the HUD (Tier 5 #1). Gotcha: `cmd | tail` hides the suite's exit status; check `$?` of the suite itself.

## Run 2026-11-09 (third)

**HEAD at start:** ebb900e
**HEAD at end:** 27f87f8 (+ this docs PR)
**Batches landed this run:** 4 code PRs + this docs PR

### Merged
- #116 — security events never dropped silently: 128-bit ids + retry, never raises, journal fallback with the event, `security_db` control — evidence: CI-passing
- #117 — Policy Gate refuses host changes the audit ledger cannot record (lockdown/diagnostics exempt); post-execution audit failure → error; no-follow ledger writes; token never in the failure log; `audit_ledger` control — evidence: CI-passing; pre-fix check fails 6 tests
- #118 — executor never blocks loopback/unspecified/multicast/broadcast/link-local, this host, its gateways or resolvers; refuses when undeterminable — evidence: CI-passing; fixtures are the host's /proc/net lines
- #119 — InternetDB fields allowlisted before cache/display (INV-004) — evidence: CI-passing

### Issues closed / filed
- none (no new findings outside these fixes; #101, #102, #104, #105 unchanged)

### Housekeeping
- Deleted 13 remote branches left from merged PRs.

### In progress / blocked
- #101 — host verification. **Host-verification blocker**: /opt/kaia is still at the 2026-11-01 code (checked this run); nothing from #98 onward is deployed. Deploy order unchanged: install, review, **re-baseline before the next boot**, start.
- #102 owner decision; #104 design-sized; #105 remainder (gate-side summary rows) is an audit-completeness trade-off for the owner.

### Baseline for next run
- `scripts/run_ci_tests.sh`: green (584)
- CI on main: green
- Lint/type: no gate in the repo

### Recommended first action next run
- If deployed, work through #101's checklist and fix what the host shows. Otherwise, code items: signed-package attestation (ROADMAP Tier 5 #2); alert reasons and a level filter in the HUD feed (Tier 5 #1); multi-table policy-routing gateways for the block guard. The fail-closed sweep covered db.py, the audit ledger, the executor's block path and threat intel; still unswept: `rule_engine` isolated validation, `web_bridge` command parsing, `fim_daemon` response path.

## Run 2026-11-09 (fourth)

**HEAD at start:** 87ab66b
**HEAD at end:** d0f8c78 (+ this docs PR)
**Batches landed this run:** 1

### Merged
- #121 — HUD console: `show audit`, `add rule`, `list recent blocks` implemented (help had advertised them; all returned "Unknown command"); a test runs every help entry through the dispatcher — evidence: CI-passing

### Reviewed, no change needed
- Web bridge command console: Host/Origin checked before body parsing, 2048-char cap, tokens scoped per command, the gate validates everything.
- `add_rule` gate path: extra payload keys are ignored by the schema; `add_rule` is in both permission sets.
- Threat-intel fetches: bounded (1 MiB, streaming, 10 s timeout).

### Deferred findings (not filed: low value)
- `fim_daemon._run` stops parsing a fanotify batch at a malformed `event_len` without checking `vers`; descriptors of later records in that batch would leak. It needs a malformed kernel record, and FIM is inactive on the host (no `CAP_SYS_ADMIN`). Fix it together with any FIM work.

### In progress / blocked
- #101 — **host-verification blocker**: `/opt/kaia` is still the 2026-11-01 code (checked this run). Nothing from #98 onward is deployed.
- #102 owner decision · #104 design-sized · #105 remainder is an audit-completeness trade-off for the owner.

### Baseline for next run
- `scripts/run_ci_tests.sh`: green (593) · CI on main: green · lint/type: no gate

### Recommended first action next run
- The code-only queue is now mostly feature work rather than fixes: signed-package attestation (ROADMAP Tier 5 #2), alert reasons and a level filter in the HUD feed (Tier 5 #1), backup checks (Tier 5 #4, needs the owner's backup location). The highest-value next step is deploying and running #101, which needs the owner.

## Run 2026-11-09 (fifth): first host deployment

**HEAD at start:** b2c1e68 · **HEAD at end:** f900b77 (+ this docs PR)

### Host (target-host verified, kernel 6.18.60-1-lts, systemd 262)
- Deployed `b2c1e68` with the script now in `scripts/kaia-deploy-verify.sh` (owner ran it; this session has no sudo password).
- Baseline 34 files; gate active; eBPF all probes attached; honeypot inode watches 4 of 4.
- Privilege probe: sudo → `privilege_setuid_exec`; `cap_setuid` test → CRITICAL `privilege_escalation_unexpected`.
- 14 controls publishing; checker and timer work as a DynamicUser; attestation (gate, 16:21) 155 files, 0 altered, 0 unowned, 1 unverifiable (dbus helper, mode 4110, unreadable without CAP_DAC_OVERRIDE, by design); full-root run clean; secret scan (gate, 16:26) 145 files, 0 exposures.

### Merged
- #123 — host findings: restore-blocks inactive-after-install is healthy, disabled is degraded; timer `OnActiveSec`; host evidence recorded (closes #101)
- #124 — SSH decoy moved to /var/backups and provisioned by the installer; /root/.ssh write grant removed (closes #102)
- #125 — `scripts/kaia-deploy-verify.sh` (root writes only to root-owned paths)

### Issues closed
- #101 (host-verified, evidence in the issue), #102 (#124)

### Needs a redeploy to take effect on the host
- #123 checker/timer, #124 decoy (expect `honeypot: active`, 5 of 5). Run: `sudo scripts/kaia-deploy-verify.sh`, then read `/var/log/kaia/deploy-*.log`.

### Open
- #104 (libbpf CO-RE, design-sized), #105 remainder (owner trade-off).

### Baseline for next run
- suite green (604) · CI on main green

### Recommended first action next run
- Read the newest `/var/log/kaia/deploy-*.log` if the owner redeployed; confirm honeypot 5 of 5 and `kaia-restore-blocks` healthy. Then feature work: signed-package attestation, HUD alert reasons/filter.

## Run 2026-11-09 (sixth): decisions, no open issues

**HEAD at end:** 3e5e443 (+ this docs PR)

### Host (second deploy, owner ran `sudo scripts/kaia-deploy-verify.sh`)
- Honeypot 5 of 5 (#124 verified), restore-blocks healthy (#123 verified), privilege probe both kinds again, attestation clean. Log: /var/log/kaia/deploy-20261109-163253.log.

### Decisions taken (the owner asked the agent to make them)
- #105: never summarise or drop audit rows; the gate publishes the ruleset and dashboards stop polling (#127).
- #104: no libbpf CO-RE rewrite; the kernel-upgrade case is surfaced as "reboot to load probes" (#128).
- CAP_SYS_ADMIN: not granted; fanotify FIM is `off` by design, other failures degraded (#129). Blocking FIM, when built, should be BPF LSM.

### Merged
- #127 (closes #105), #128 (closes #104), #129

### Open issues: none

### Needs a redeploy
- #127/#128/#129: `sudo scripts/kaia-deploy-verify.sh`. Expect the health check `ACTIVE` (fim off, decoy_listeners off, yara off), a `firewall` control, and no new `diagnostics`/`dashboard` ledger rows while a dashboard is open.

### Baseline for next run
- suite green (615) · CI on main green

### Recommended first action next run
- Read the newest /var/log/kaia/deploy-*.log. Then Tier 5 feature work in order: alert reasons and a level filter in the HUD feed; signed-package attestation; behavioural baselines.

## Run 2026-11-09 (seventh)

**HEAD at end:** b1d188f (+ this docs PR)

### Merged (all CI-passing)
- #131 HUD feed: level column, reasons, High+ filter, structured fields; node syntax check of HUD JS
- #132 signed-package verification (daily unprivileged timer; host run 149 files clean)
- #133 / #134 / #137 behavioural baselines: first-seen executables, outbound (process, port), per-program deletion rate
- #135 backup recency + sampled restorability (host: last backup 2026-11-01, 5/5 samples match)
- #136 Arch security advisories daily (host: 24 affected, 4 High, none fixable yet)
- #138 HUD start false AuditSurge fixed; ledger followed by offset; **test suite no longer reaches the live gate**
- #139 HUD shows the gate's published eBPF events; the HUD no longer loads BCC

### Host
- Third deploy (#127–#129) verified: health ACTIVE (14 active, 3 off), firewall control, honeypot 5/5.
- Headless Chromium against the live state: renders; found and fixed #138 and #139.

### Needs a redeploy: `sudo scripts/kaia-deploy-verify.sh`
- Expect new controls exec_baseline, conn_baseline, delete_rate_baseline (learning, 7 days), backup (active), and timers kaia-package-verify and kaia-vuln-check (steps 6b/6c).

### Open issues: none
### Baseline: suite green (691), CI green

### Next
- After deploy: read /var/log/kaia/deploy-*.log; confirm the backup control works through the automount inside the gate's namespace.
- Remaining roadmap: HUD streaming output, bundled fonts, audio cues, no-WebGL2 fallback check; NBNS alternate encodings; BPF-LSM execution blocking (design first).

## Run 2026-11-09 (eighth)

**HEAD at end:** a6a9ed5 (+ this docs PR)

### Host: fourth deploy verified
- Control health ACTIVE (20 active, 3 off by design); backup control reads the automounted mirror from inside the gate; exec/conn/deletion baselines learning (7 days, until ~2026-11-16); signed-package timer (149 files) and advisory timer run as dynamic users; HUD shows the gate's eBPF events.

### Merged (CI-passing)
- #141 feed local time + ordered history; host evidence + decision: no exec blocking on this host
- #142 opt-in HUD audio cues (tested in node)
- #143 HUD head-of-line blocking: one stalled client cut others to 0.47 updates/s; now concurrent sends with a 2 s timeout (live: 4.2/s, no gaps); 3-minute soak memory stable

### Live checks done this run (headless Chromium + WebSocket clients against the live state)
- Rendering, no-WebGL2 fallback, slow client, 3-minute soak.

### Needs a redeploy for #141–#143 (HUD runs from the checkout, so it already uses them)
- `sudo scripts/kaia-deploy-verify.sh` when convenient; nothing in the gate changed since the fourth deploy except docs.

### Remaining roadmap (all low value or need assets)
- HUD: bundled fonts (system fonts work), streaming command output (no long-running commands exist yet), full-body avatar (needs art).
- Discovery: alternate NBNS encodings.
- After 2026-11-16 (learning ends): review the first baseline alerts for noise and tune.

### Baseline: suite green (695) · CI green · open issues: none
