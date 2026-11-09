# Progress log

Handoffs between autonomous maintenance runs. Append a section per run; never rewrite one.

## Run 2026-11-09 21:40 UTC

**HEAD at start:** 2a65e5a
**HEAD at end:** 4de7cd7 (`main`; four PRs stacked and open for review, see below)
**Batches landed this run:** 1 merged, 4 open (green CI, awaiting owner review)

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
