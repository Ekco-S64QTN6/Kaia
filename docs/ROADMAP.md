# Roadmap

What is still to build or verify. Updated 14 November 2026. What is implemented, and at what
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

5. **Deploy and verify the 2026-11-14 review fixes** (none is on the host yet). After
   `sudo scripts/kaia-deploy-verify.sh`: `discovery` should be `active` again (it has been `failed`
   since 2026-11-10); `add rule` should succeed from the HUD console (never confirmed under the
   deployed unit); the HUD console should run commands from `./kaia-hud` and refuse them from
   `curl` without the key. Replace the poisoned restore point: `/var/lib/kaia/pre-lockdown.nft`
   currently holds the lockdown table itself (256 bytes); delete it, or copy the 1109-byte
   `pre-lockdown-20261030-225528-364775073.nft` over it, as root.
6. **Reboot.** The kernel was upgraded on 2026-11-11 (6.18.60 running, 6.18.62 installed), so
   `ebpf` is `degraded`: a gate restart before the reboot loads no probes at all, and the deploy
   in step 5 restarts the gate. Reboot first, then deploy.
7. **First live auto-block.** Honeypot auto-blocking starts working with step 5. Exercise it
   deliberately once (a decoy read from an SSH session from another machine), confirm the
   `honeypot_auto_block` event and the rule in `inet kaia_block`, then lift it with `unblock`.

## Found by the 2026-11-14 review, not yet built

Ordered by how much each would have prevented or shortened a failure already seen on this host.

| Item | Why | What it takes |
|:--|:--|:--|
| **Alerts that reach the owner** | Discovery was `failed` for four days and the health-check unit reported it about 1,000 times (every 5 minutes); nobody saw it because the only outputs are the journal and a HUD that has to be open. A critical event (canary read, unexpected root gain, tamper) has the same problem | Out-of-band delivery for critical/high events and for a health check that turns `failed`: a desktop notification through the owner's session bus first (no network, no secret), then an optional operator-configured webhook or mail relay sent from an unprivileged unit, never from the gate. Rate-limited, with a daily "all quiet" heartbeat so silence means something |
| **Response drills** | The auto-block path had never worked since it was written, because nothing exercised it end to end; the same was true of the lockdown display and `kaia-restore-blocks` before it | `kaia drill`: a root-run, logged self-test that sends a synthetic honeypot hit for a TEST-NET address (192.0.2.0/24) through the real gate and executor, checks the rule, the ledger row and the event, then unblocks; and a `--dry-run` lockdown that validates the ruleset and the restore point without applying. Run it from `kaia-deploy-verify.sh` |
| **Expiring and reviewable auto-blocks** | With auto-blocking live, every decoy read adds a permanent rule, replayed on every boot, until someone unblocks it by hand. A mis-attributed block (a CDN address, an agent's API) stays forever | A `timeout` on automatic blocks (nftables set with per-element timeouts, default 24 h, operator blocks stay permanent), the expiry recorded in the ledger so boot replay honours it, and an "automatic blocks" list in the HUD with one-click lift |
| **Blocks cut outbound traffic too** | A block is `saddr X drop` on input only. A reverse shell or an exfiltration to the same address keeps sending; only replies are dropped | A matching `daddr X drop` in an output-hook chain of `inet kaia_block`, added and removed together with the input rule, with the same protected-address refusals |
| **Baseline covers everything root runs** | The tamper baseline lists `security/*.py`, `core/config.py` and five scripts. Not listed, and run as root from `/opt/kaia`: `scripts/kaia_restore_blocks.py` (every boot), `install_decoys.py`, `install_canaries.py`, `kaia_reset_history.py`, `kaia-deploy-verify.sh`; plus the scripts the timers run | Derive the list from the unit files' `ExecStart=` lines and the installer, add them to `TamperDetector.immutable_files` (alert-only for the unprivileged ones), and a test that fails when a unit executes an unlisted file. Needs a re-baseline on deploy |
| **IPv6 and UDP in connection telemetry** | Only `tcp_v4_connect` is probed. An outbound IPv6 connection, or DNS/QUIC over UDP, is invisible to the connection baseline and to `net_shell_first_seen` | `tcp_v6_connect` with a 16-byte address field, and `udp_sendmsg` sampled per (process, port); extend the baseline key unchanged |
| **`run_script` on the deployed host** | Allowlisted scripts are looked up in the gate's home directory, which under `ProtectHome=tmpfs` is an empty tmpfs: the action cannot find any script once deployed. The Bubblewrap profile also binds the workspace (`/opt/kaia`) read-write | Decide: remove the action from the deployed lattice, or provision a root-owned `/etc/kaia/scripts/` bound read-only, with the workspace bound read-only too |
| **Integrity check tolerates a transient read error** | `_compute_sha256` returns `""` on any error (EMFILE, EIO), which differs from the baseline, which is a tamper alert and, for `security/*.py`, a lockdown | Distinguish "unreadable" from "different": retry once, then alert `tamper_unverifiable` without locking down unless it persists for N passes |
| **Daily digest** | Events are only visible live. There is no record a person reads of "what happened yesterday" | An unprivileged daily timer writing a short report (events by level, new first-seen executables and ports, control health changes, advisories, backup age) to the state directory and the HUD |
| **Lint and type gate** | CI has no static checks; the missing-field bug is the kind a typed request constructor catches | `ruff` and a type checker on `security/` and `scripts/` in CI; build gate requests through the Pydantic models on the sending side too, so a request that the gate would refuse fails where it is written |

## Security

| Item | State | What is left |
|:--|:--|:--|
| **Control health** | Every control reports its state with heartbeats (including the event database and the audit ledger); an unprivileged timer outside the gate checks freshness and `kaia-restore-blocks` | Host verification (#101); database and threat-intel freshness |
| **Kernel telemetry** | execve, connect/retransmit, `commit_creds` root gains, openat2, unlinkat attempts, honeypot inodes; ring buffers with drop counters; kprobes attached individually (a partial load is `degraded`); a local compile check | After a kernel upgrade, control health says the probes need a reboot (#104; CO-RE not taken: decision in the changelog); not seen: in-place cred overwrites and capability-only gains; correlation of syscall attempts with outcomes |
| **Execution-blocking FIM** | Decided 2026-11-09: not built on this host | Builds legitimately execute from /tmp and cache directories, so a blocking policy (BPF LSM `bprm_check_security`) risks making the machine unusable on any bug; `exec_unusual_location` alerts cover detection. Revisit for a server profile with a fixed workload |
| **End-to-end FIM / YARA response** | Rule generation, isolated validation and same-FD scanning exist; validation files moved out of the gate's private `/tmp` (2026-11-14) | `add rule` confirmed on the deployed host; the full event → verdict → response path, exercised on a controlled host |
| **Network discovery** | NB and NBSTAT answer records, first-level encoded names, scoped names, bounded DNS compression parsing, and rejection of fragmented/reserved-flag IPv4 frames before UDP parsing | Alternate NBNS name encodings and target-network validation |
| **Supply chain** | Completed pacman transactions are recorded; critical installed files (setuid/setgid, PAM, loader, libc, executor binaries) are attested against pacman's `mtree` every 6 h, and unowned setuid files under `/usr` are found | Attestation against signed packages in the cache (the local database is unsigned); wider scope on demand; AUR build tracking, repository and metadata changes, dependency-graph drift |
| **Network decoys** | Filesystem decoys are non-destructive and self-cleaning; the SSH decoy is at `/var/backups` (#124); automatic blocking of a decoy reader's peer works since 2026-11-14 (it never had) | Decoy listeners and their namespace (the installer does not provision them); first live auto-block (step 7 above) |
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
| **Full-body avatar** | The avatar uses KaiaRX/KaiaBooth's portrait codec (six rotating portraits, head and shoulders). A full-body figure needs new art and a body rig; none exists yet. |
| **Bundled fonts** | The HUD uses system fonts; bundling the display and mono faces keeps the look identical across hosts without loosening the CSP. |
| **Browser checks** | Headless Chromium against the live state on 2026-11-09: control strip, level column, filters and the gate's eBPF waterfall render; the no-WebGL2 fallback shows the static portrait. Slow-client and 3-minute soak checks done (found and fixed head-of-line blocking; memory stable). |

## Next (Tier 5, ordered)

1. **Done 2026-11-09:** alert levels and reasons in the HUD feed, with a High+ filter.
2. **Done 2026-11-09:** daily verification against the signed package archives.
3. **Wider secrets scanning.** Kaia's own state, logs and tree are scanned (2026-11-09); extend to
   operator-configured scopes (home dotfiles, shell history) outside the gate's `ProtectHome` sandbox.
4. **Done 2026-11-09:** backup recency and sampled restorability (`backup` control).
5. **Behavioural baselines:** executables done (2026-11-09: 7-day learning, first-seen alerts).
   Outbound (process, port) pairs and per-program deletion rates done too.
6. **Threat-intel datasets:** installed-package advisories done (2026-11-09, daily). Remaining: local GeoLite2, CVE and passive-DNS stores, historical snapshots,
   optional DuckDB analysis.

## Housekeeping

- **Credential history.** The old `KAIA_CAPABILITY_TOKEN_SECRET` value remains in public Git
  history. It was rotated on 2026-09-08 and the installer rejects it; rotation is the mitigation.
- **Backup drive.** An old note mentioned pending sectors on `sdb`; re-check the device directly.
- **Lockdown snapshot history.** `pre-lockdown-<time>.nft` copies in the state directory are never
  pruned (nine from 2026-10-30, eight of them the lockdown itself). Keep the newest few.
- **Unreviewed in depth on 2026-11-14:** `kaia_dashboard.py` (2,600 lines) was read for structure
  only. The monitors, the installer, `kaia_cli.py`, `database_utils.py` and the HUD JavaScript were
  read in a second pass the same day (HTML escaping in the HUD is consistent).
- **A pacman transaction killed part-way is never recorded.** `supply_chain_monitor.py` emits events
  only at `transaction completed`; packages already installed when pacman died are dropped at the
  next `transaction started`. Emit them as `pacman_package_*` with an "incomplete" marker.
- **Backup age is inferred from directory ctime.** `backup_check.py` dates the last run from the
  ctime of the mirror's `etc`, `usr`, `var`; a run that changes nothing directly inside them can
  look older than it is. Have the backup script write a timestamp file and prefer it.
- **Root-owned `logs/kaia.log` in the checkout** makes an unprivileged development run of the gate
  fail at logging setup; `sudo chown` it back or delete it.
