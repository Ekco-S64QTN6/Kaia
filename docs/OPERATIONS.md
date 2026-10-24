# Operations

Running, deploying and recovering Kaia on a host. What remains to verify on the target host is in [ROADMAP.md](ROADMAP.md); the step-by-step acceptance tests are in [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md).

## Before editing Kaia's own source

**Stop the Policy Gate before editing Kaia's own source.**

`security/tamper_detection.py` hashes the core files at startup and re-checks them
every 30 seconds. Modifying any of them while the daemon is running is — correctly —
indistinguishable from an intruder patching the security layer, and triggers
`kaia-lockdown.service`, which flushes nftables and drops **all** traffic on the host.

```bash
sudo systemctl stop kaia-policy-gate     # 1. disarm
#    ... edit, test, commit ...
sudo systemctl start kaia-policy-gate    # 2. re-baseline against the new state
```

## Tamper-protected files

Modifying any of these trips the detector:

| File | Consequence |
|---|---|
| `.env` | **Lockdown** — holds `KAIA_CAPABILITY_TOKEN_SECRET` |
| `core/config.py` | **Lockdown** |
| **every** `security/*.py` | **Lockdown** — discovered by glob, so new modules are covered automatically |
| `scripts/kaia-lockdown.sh` | **Lockdown** |
| `/etc/systemd/system/kaia-policy-gate.service` | **Lockdown** |
| `kaia_dashboard.py`, `main.py` | Critical alert + audit event, no lockdown |

The detector watches **itself**, `security/db.py`, and `kaia-lockdown.sh` deliberately:
without that, the cheapest bypass in the system is to neutralise the watchdog first.

## Deployment model

**Develop in a checkout; run from `/opt/kaia`.**

The daemon runs as root — it manipulates nftables, restarts units and binds raw
sockets. Running it directly out of a checkout under `$HOME` therefore means root
executing code any unprivileged process can rewrite, which is a privilege-escalation
path that tamper detection can *report* but not *prevent*.

```bash
sudo scripts/install_services.sh
```

That deploys a root-owned copy below `/opt` (default `/opt/kaia`; 0644 files, 0755
dirs, root:root), provisions the configured state directory (default `/var/lib/kaia`;
custom paths must be below `/var/lib`, `/srv`, or `/mnt`), migrates existing runtime
data, and installs the units. The installer rejects destinations that overlap the
source checkout or each other. It validates the signing secret before
changing files and stops an active Policy Gate before syncing monitored code. The
installer enables but does **not start** the gate automatically. It rejects
unexpected symlinks before root ownership normalization, allowing only the tracked
`static/dashboard -> ../core/web/static` link while it resolves within the tree.
Review the deployed
diff, write the trusted baseline, then start it:

    sudo /opt/kaia/scripts/kaia-baseline.sh
    sudo systemctl start kaia-policy-gate.service

The checkout is the source; `/opt/kaia` is what runs.

## Privileges

| Capability | Status | Why |
|---|---|---|
| `CAP_NET_ADMIN` | granted | nftables rules for IP blocking |
| `CAP_NET_RAW` | granted | `AF_PACKET` sockets for passive discovery |
| `CAP_DAC_OVERRIDE` | **dropped** | Bypasses *all* file permission checks. Only ever needed to read a `0600 .env` owned by an unprivileged user; the secret now comes from root-owned `/etc/kaia/secret.env`. |
| `CAP_SYS_ADMIN` | Dropped in the last recorded host configuration | The 2026-09-08 host review recorded fanotify `EINVAL` and absent BCC. This is a host-specific observation, not proof that the current source lacks those features; rerun the probe on the deployed kernel before changing service capabilities. |

To re-enable mount-wide FIM, check whether this kernel will accept a mount mark:

```bash
sudo python3 scripts/fanotify_probe.py /opt/kaia
```

If any mask tier reports `OK`, add `CAP_SYS_ADMIN` back to **both** the
`CapabilityBoundingSet` and `AmbientCapabilities` lines in
`scripts/kaia-policy-gate.service` and reinstall.

## State and the audit ledger

INV-008 requires the ledger to live outside the agent's working tree. It now sits in
`/var/lib/kaia/` (root:kaiacord, 0750), not `storage/security/` inside the repo where
anything running as the invoking user could rewrite the administration trail.

Running unprivileged — development, CI, the test suite — falls back to the in-repo
path automatically, so tests need no root.

Rotation is deliberately **operator-driven**, never automatic: INV-003 gives the agent
zero write/delete capability over the ledger, and tamper detection prefix-hashes it, so
a daemon rotating its own log would be indistinguishable from an intruder truncating it.

```bash
sudo systemctl stop kaia-policy-gate
sudo ./scripts/kaia-rotate-ledger.sh     # archives + gzips, keeps 6
sudo ./scripts/kaia-baseline.sh          # ledger hash changed
sudo systemctl start kaia-policy-gate
```

## Block persistence

nftables rules are runtime-only. `kaia-restore-blocks.service` reconstructs the
currently active Kaia-managed blocks from the audit ledger before the Policy Gate
accepts new requests. Replay folds approved block_ip and unblock_ip events in
chronological order: a successful unblock is a tombstone for the earlier rule, while
a later approved re-block restores it. Active-rule memory is bounded to the newest
configured entries (default 500, maximum 10,000), reads are capped at 16 KiB per
record, and symlinked ledgers are rejected. Malformed or oversized records are skipped
and logged; the ledger remains append-only and its total growth needs separate monitoring.
This behavior is covered by ledger tests, but boot-time replay and the live nftables
path still need verification on the target host.

Use `unblock <ip>` through the dashboard for a normal removal. Do not manually
delete the table during a replay test and then assume the ledger agrees: without an
approved unblock event, boot replay is expected to restore an active ledger block.

## If the network goes down unexpectedly

Run this. It is deliberately standalone — no Kaia imports, no repo dependency, no
network access — because if Kaia is what broke, nothing of Kaia's can be trusted to
fix it:

```bash
sudo ~/kaia-panic-unlock.sh
```

It stops the Policy Gate first (otherwise the 30s tamper loop just re-locks you out
on the next tick), deletes **only** Kaia's own `inet filter` lockdown table, reloads
ufw, restarts NetworkManager, and verifies connectivity.

A reboot also clears a lockdown — nftables rules are runtime-only and ufw re-applies
at boot — but that loses your session and tells you nothing about why it fired.

## Recovering from a lockdown

`kaia-lockdown.sh` snapshots the live ruleset to `/var/lib/kaia/pre-lockdown.nft`
before flushing, plus a timestamped copy for forensics. The snapshot files are root-only; the state
directory remains `root:kaiacord` mode `0750` so lockdown does not break group-readable
operator ledgers and databases. To lift it:

```bash
sudo ./scripts/kaia-unlock.sh          # regenerates via `ufw reload` where ufw is active
sudo ./scripts/kaia-unlock.sh -s       # or replay the raw pre-lockdown snapshot
```

Release requires typing `RESTORE` at a prompt — it re-exposes a host that something
asserted was compromised, so it is never automatic. Afterwards, restart the Policy
Gate so the tamper baseline is re-established.

## Trusted integrity baseline

Tamper detection compares against an operator-recorded baseline, not against
whatever happens to be on disk when it starts. Without one, a file edited while the
daemon was stopped simply *becomes* the new baseline — and anyone who can write a
file can also restart a service, so that was a complete bypass.

```bash
sudo ./scripts/kaia-baseline.sh          # record the state you trust
sudo systemctl restart kaia-policy-gate  # adopt it
```

Run it **only from a state you have reviewed.** Baselining a compromise makes the
compromise the reference. The script shows a diff against the previous baseline
before replacing it, and enumerates files via the detector itself so the two cannot
drift apart. It hashes into a temporary file and atomically replaces the trusted
baseline only after every protected file is successfully hashed; failure preserves
the previous baseline. The baseline is root-only, while the state directory remains
`root:kaiacord` mode `0750` for group-readable operator data.

The baseline writer fails closed when watchlist enumeration fails or a protected
path is missing, symlinked or non-regular (the intentionally absent deployed `.env`
is the only exception). System mode refuses to start without a valid trusted baseline.
Only unprivileged development mode may bootstrap from the current disk state, where
changes made while the daemon was stopped cannot be detected. Record a baseline only after reviewing
the deployed source and deployment state; never treat bootstrap hashing as equivalent
to verification against a previously trusted baseline.

## Monitor-only mode

For tuning, or for working on the project without arming the tripwire:

```bash
KAIA_TAMPER_ENFORCE=0 python security/policy_gate.py
```

Detection and audit logging continue; lockdown is suppressed. This disables an active
security control, so it is opt-in and logged as CRITICAL at startup.

## Lockdown circuit breaker

At most **3 lockdowns per hour** (`LOCKDOWN_MAX_PER_WINDOW` / `LOCKDOWN_WINDOW_SECONDS`).
Beyond that, tampering is still detected and recorded but lockdown is suppressed, so
the host stays reachable to investigate through. Every lockdown flushes the ruleset;
an unbounded burst leaves you with no network and no firewall at the worst moment.

**Alert latching:** one modification produces one CRITICAL alert and one lockdown.
Subsequent checks log at WARNING without re-firing, and the detector re-arms when the
file returns to baseline. Without this a single event re-flushed the firewall every
30 seconds, which is precisely when an operator needs the network to investigate.
