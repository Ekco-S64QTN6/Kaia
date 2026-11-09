# Roadmap

What is still to build or verify. Updated 9 November 2026. What is implemented, and at what
evidence level, is in [master_plan.md](master_plan.md) §10.5; how to verify a deployed host is in
[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md). Open work is tracked as GitHub issues; the
numbers below link the detail.

## First: verify the deployed host

Nothing below is worth more than this. The repository can prove what its code does; only the
installed host can prove what is running.

1. **Done 2026-11-09** ([#101](https://github.com/Ekco-S64QTN6/Kaia/issues/101)): `main` deployed, baseline
   recorded, privilege probe (both kinds), control health, timer and attestation verified on the host.
2. Verify the installed systemd unit, `ExecStart`, `WorkingDirectory`, state directory and the
   trusted integrity baseline (deployment below `/opt`, state below `/var/lib`, `/srv` or `/mnt`,
   neither overlapping the checkout).
3. Live-test **block → UFW reload → reboot restore → unblock → second reboot** (the rule must stay
   absent).
4. Confirm fanotify marks on the deployed kernel and the FIM database migration and permissions.
   eBPF loading was confirmed on 2026-11-03 (journal: probes loaded, honeypot inode watches 4 of 4).

## Security

| Item | State | What is left |
|:--|:--|:--|
| **Control health** | Every control reports its state with heartbeats (including the event database and the audit ledger); an unprivileged timer outside the gate checks freshness and `kaia-restore-blocks` | Host verification (#101); database and threat-intel freshness |
| **Kernel telemetry** | execve, connect/retransmit, `commit_creds` root gains, openat2, unlinkat attempts, honeypot inodes; ring buffers with drop counters; kprobes attached individually (a partial load is `degraded`); a local compile check | After a kernel upgrade, control health says the probes need a reboot (#104; CO-RE not taken: decision in the changelog); not seen: in-place cred overwrites and capability-only gains; correlation of syscall attempts with outcomes |
| **Execution-blocking FIM** | Notification-mode fanotify observes, does not block | A design for blocking. Two candidates, both available on the host kernel (`CONFIG_FANOTIFY_ACCESS_PERMISSIONS=y`; BPF LSM active, BTF present): fanotify `FAN_OPEN_EXEC_PERM` (needs `CAP_SYS_ADMIN`, userspace verdict per exec, deadlock and timeout semantics) or a BPF LSM `bprm_check_security` program (in-kernel verdict, fails open if unloaded). Either needs explicit fail-open/fail-closed rules and target-kernel tests |
| **End-to-end FIM / YARA response** | Rule generation, isolated validation and same-FD scanning exist | The full event → verdict → response path, exercised on a controlled host |
| **Network discovery** | NB and NBSTAT answer records, first-level encoded names, scoped names, bounded DNS compression parsing, and rejection of fragmented/reserved-flag IPv4 frames before UDP parsing | Alternate NBNS name encodings and target-network validation |
| **Supply chain** | Completed pacman transactions are recorded; critical installed files (setuid/setgid, PAM, loader, libc, executor binaries) are attested against pacman's `mtree` every 6 h, and unowned setuid files under `/usr` are found | Attestation against signed packages in the cache (the local database is unsigned); wider scope on demand; AUR build tracking, repository and metadata changes, dependency-graph drift |
| **Network decoys** | Filesystem decoys are non-destructive and self-cleaning | The `/root/.ssh` decoy cannot be created under `ProtectHome=tmpfs` (#102); decoy listeners and their namespace (the installer does not provision them) |
| **Audit ledger growth** | Dashboards read the ruleset the gate publishes (no requests, so no audit rows); every request is still audited | Old dashboard-polling rows stay in the ledger until the operator rotates it (`kaia-rotate-ledger.sh`) |
| **Capability-token replay protection** | HMAC signature, scope, bounded UTC validity and durable single-use consumption; the database fails closed | Include the persistent registry in deployed-host backup, permissions and capacity verification |
| **Policy Gate IPC framing** | Length-prefixed JSON, size caps, a single read deadline and protocol-error closure, with synthetic regressions | Keep fragmented-read and truncated-frame regressions with protocol changes |
| **Host Executor executable boundary** | Only exact registered `SYSTEM_BINARIES` paths may launch; malformed argv is rejected before `Popen` | Add or change registered host executables only with security review and negative launch tests. Landlock (active on the host) could further confine sandboxed scripts underneath Bubblewrap |
| **Trusted integrity baseline** | No-follow loading, regular single-link files, root ownership and private permissions in system mode | Keep loader checks aligned with the baseline writer and deployed state-directory policy |

## Operator HUD

The loopback web HUD is built: telemetry waterfall and sparklines, passive-discovery radar,
capability-token widget, sentinel states, control-health strip, the Kaia codec avatar (WebGL2,
face rig) and the command console, behind a strict self-hosted CSP and loopback Host/Origin checks.

| Item | What is left |
|:--|:--|
| **Streaming command output** | Long-running diagnostics and sandboxed operations return one completed response. Needs a bounded, framed streaming protocol with cancellation, timeouts and UI back-pressure tests. |
| **Full-body avatar** | The avatar is a face-rig codec. The full-body figure in the design references ([`design/`](design/)) is not built. |
| **Bundled fonts** | The HUD uses system fonts; bundling the display and mono faces keeps the look identical across hosts without loosening the CSP. |
| **Audio cues** | Optional Web Audio cues for alerts and state changes. |
| **Browser checks** | Stress and fallback tests in a real browser: no WebGL2, slow sockets, a long-running HUD; the control strip and token cards against live data. |

## Next (Tier 5, ordered)

1. **Done 2026-11-09:** alert levels and reasons in the HUD feed, with a High+ filter.
2. **Attestation against signed packages** (Supply chain, above): verify the critical files
   against the signed `.pkg.tar.zst` in the pacman cache, closing the unsigned-database gap.
3. **Wider secrets scanning.** Kaia's own state, logs and tree are scanned (2026-11-09); extend to
   operator-configured scopes (home dotfiles, shell history) outside the gate's `ProtectHome` sandbox.
4. **Backup checks**: existence, recency and a test restore, never inferred from file presence.
5. **Behavioural baselines** for processes, connections and file activity, and only after alert
   confidence, so their output has somewhere honest to go.
6. **Threat-intel datasets**: local GeoLite2, CVE and passive-DNS stores, historical snapshots,
   optional DuckDB analysis.

## Housekeeping

- **Credential history.** The old `KAIA_CAPABILITY_TOKEN_SECRET` value remains in public Git
  history. It was rotated on 2026-09-08 and the installer rejects it; rotation is the mitigation.
- **Backup drive.** An old note mentioned pending sectors on `sdb`; re-check the device directly.
- **Root-owned `logs/kaia.log` in the checkout** makes an unprivileged development run of the gate
  fail at logging setup; `sudo chown` it back or delete it.
