# Roadmap

What is still to build or verify. Updated 31 October 2026. What is implemented, and at what
evidence level, is in [master_plan.md](master_plan.md) §10.5; how to verify a deployed host is in
[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md).

## First: verify the deployed host

Nothing below is worth more than this. The repository can prove what its code does; only the
installed host can prove what is running.

1. Verify the installed systemd unit, `ExecStart`, `WorkingDirectory`, state directory and the
   trusted integrity baseline (deployment below `/opt`, state below `/var/lib`, `/srv` or `/mnt`,
   neither overlapping the checkout).
2. Live-test **block → UFW reload → reboot restore → unblock → second reboot** (the rule must stay
   absent).
3. Confirm fanotify marks and BCC probe compilation on the deployed kernel, and the FIM database
   migration and permissions.

## Security

| Item | State | What is left |
|:--|:--|:--|
| **Execution-blocking FIM** | Notification-mode fanotify observes, does not block | A separate permission-event (`FAN_CLASS_CONTENT`) design, with target-kernel tests |
| **End-to-end FIM / YARA response** | Rule generation, isolated validation and same-FD scanning exist | The full event → verdict → response path, exercised on a controlled host |
| **Kernel telemetry** | `sys_enter_unlinkat` records deletion *attempts*; probes depend on BCC and the kernel | More hook coverage, correlation with successful operations, target-kernel verification |
| **Network discovery** | NB and NBSTAT answer records, first-level encoded names, scoped names, bounded DNS compression parsing, and rejection of fragmented/reserved-flag IPv4 frames before UDP parsing | Alternate NBNS name encodings and target-network validation |
| **Supply chain** | Completed pacman transactions are recorded | AUR build tracking, repository and metadata changes, dependency-graph drift, package signature and installed-content attestation |
| **Network decoys** | Filesystem decoys are non-destructive and self-cleaning | Decoy listeners and their namespace, verified on the host (the installer does not provision them) |
| **Capability-token replay protection** | HMAC signature, scope, bounded UTC validity and durable single-use consumption are enforced; SQLite fingerprints survive concurrency and service restarts, and the database fails closed | Include the persistent registry in deployed-host backup, permissions and capacity verification |
| **Policy Gate IPC framing** | Length-prefixed JSON, request/response size caps, a single read deadline and protocol-error connection closure are covered by synthetic regressions | Keep fragmented-read and truncated-frame regressions with protocol changes |
| **Host Executor executable boundary** | Only exact registered `SYSTEM_BINARIES` paths may launch; malformed, non-string and NUL-containing argv are rejected before `Popen` | Add or change registered host executables only with security review and negative launch tests |

## Operator HUD

The loopback web HUD is built: telemetry waterfall and sparklines, passive-discovery radar,
capability-token widget, sentinel states, the Kaia codec avatar (WebGL2, face rig) and the
command console, behind a strict self-hosted CSP and loopback Host/Origin checks.

| Item | What is left |
|:--|:--|
| **Streaming command output** | Long-running diagnostics and sandboxed operations return one completed response. Needs a bounded, framed streaming protocol with cancellation, timeouts and UI back-pressure tests. |
| **Full-body avatar** | The avatar is a face-rig codec. The full-body figure in the design references ([`design/`](design/)) is not built. |
| **Bundled fonts** | The HUD uses system fonts; bundling the display and mono faces keeps the look identical across hosts without loosening the CSP. |
| **Audio cues** | Optional Web Audio cues for alerts and state changes. |
| **Browser checks** | Stress and fallback tests in a real browser: no WebGL2, slow sockets, a long-running HUD. |

## Later (Tier 5)

Process, network and file behavioural baselines · alert-confidence scoring · secrets-exposure
scanning · backup existence, recency and restorability checks · independent service and
data-freshness supervision · historical threat-intelligence snapshots and local GeoLite2, CVE and
passive-DNS datasets with DuckDB analysis.

## Housekeeping

- **Credential history.** The old `KAIA_CAPABILITY_TOKEN_SECRET` value remains in public Git
  history. It was rotated on 2026-09-08 and the installer rejects it; rotation is the mitigation.
- **Backup drive.** An old note mentioned pending sectors on `sdb`; re-check the device directly.
