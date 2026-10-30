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


## Capability-token trust and rotation

Capability tokens are bearer credentials used by the Policy Gate to prove that a request
was signed by a process holding Kaia's shared signing secret. A token is not a substitute
for the Policy Gate's other checks: the request still has to pass the action's Pydantic
schema, the global/workspace permission intersection where applicable, and the executor's
own scope and allowlists.

### Who issues and verifies tokens

The current Web Bridge command path in `core/web_bridge.py` imports
`generate_capability_token` from `security/policy_gate.py`. Its operator-command
dispatcher creates short-lived tokens for firewall block/unblock, service restart and
emergency lockdown commands, then sends the token in the length-prefixed Unix-socket
request to the Policy Gate. Those first-party commands currently request a 15-second
lifetime. The helper's default lifetime is 3,600 seconds; callers that use the default
must not assume the 15-second HUD lifetime applies.

The token is a JSON string containing:

| Field | Meaning in the current implementation |
|:--|:--|
| `capability` | The capability being requested (for example, `block_ip`) |
| `target` | The exact target, or `*` where a wildcard scope is deliberately used |
| `issued_at` | UTC timestamp written by the generator |
| `expires` | UTC expiration timestamp |
| `issued_by` | The literal label `operator`; it is not independently authenticated |
| `signature` | Hex HMAC-SHA256 over the other token fields serialized as sorted-key JSON |

The Policy Gate recomputes the HMAC with the same secret and compares it in constant time.
It then checks the required capability, the exact target or wildcard target, and the
expiration. Successful token verification does not bypass the request schema, policy
lattice or executor validation.

### Trust boundary and known limitations

`KAIA_CAPABILITY_TOKEN_SECRET` is the authorization signing root. Any process or person
that can read it can create signatures for token-gated capabilities and targets; the
code does not provide a separate signing service or per-issuer key. The Gate's remaining
policy and executor checks still apply, but exposure of this key is an authorization
root compromise.

The Web Bridge binds to loopback and checks the Host authority and, when supplied, the
Origin. These are locality and browser-origin protections, not user authentication:
the command route does not establish a user identity, and Origin is optional. Treat local
processes that can reach the bridge as inside this command boundary. Do not expose the
HUD port through a LAN bind, reverse proxy or tunnel without first adding and reviewing
an explicit authentication boundary.

Tokens are bearer values. The verifier now requires both timestamps to be
timezone-aware ISO-8601 values, rejects an `issued_at` later than the verifier clock,
requires `expires` to be later than `issued_at`, rejects expired tokens, and enforces a
maximum signed lifetime of 3,600 seconds. The generator rejects negative, non-integer,
and over-one-hour durations; a zero-second token can still be generated for negative
tests, but verification rejects its non-positive lifetime. The `issued_by` field remains
an informational literal and is not authenticated.

The Policy Gate now consumes every token-bearing command exactly once. It records a
SHA-256 fingerprint of the verified HMAC signature, the signed expiry and the consumption
time in the `consumed_capability_tokens` table in the configured security SQLite database.
The raw token and raw signature are never stored in this registry. A SQLite
`BEGIN IMMEDIATE` transaction plus the fingerprint primary key serializes concurrent
consumers across worker threads and processes; the record survives Policy Gate restarts.
Database errors fail closed and deny the action.

Consumed fingerprints are intentionally retained even after token expiry. Removing them
based only on wall-clock time could make a used token usable again after a clock rollback.
The table therefore grows with successful token-bearing commands for the lifetime of the
signing key; operators should include it in the existing security-database backup and
capacity monitoring. This is one-time use of a signed bearer credential, not user
authentication: `issued_by` remains informational and an authorized signer can still mint
a new token.

### Secret custody

For a system install, the Policy Gate systemd unit reads
`EnvironmentFile=/etc/kaia/secret.env`. `scripts/install_services.sh` reads only the
`KAIA_CAPABILITY_TOKEN_SECRET` assignment from the private checkout `.env`, requires
exactly 64 hexadecimal characters (32 random bytes), rejects the historically leaked
value, then writes `/etc/kaia/secret.env` as a root-owned mode-`0600` file under the
mode-`0700` `/etc/kaia` directory. The deployed code tree does not contain the checkout
`.env`. In development, `core/config.py` loads the checkout `.env`; the launcher
`scripts/activate_kaia_env.sh` sources it before starting the dashboard.

Keep `.env` out of Git (it is ignored), restrict it to the operator account (for example,
`chmod 600 .env`), and never place the secret or complete token strings in prompts,
tickets, screenshots, logs or source control. Do not copy the signing secret into the LLM
planner: the planner proposes structured intents and is not an authorization issuer.

### Rotating the signing secret

There is no dedicated key-rotation command. For a system install, use the installer so
the Gate's EnvironmentFile and the checkout configuration are updated from the same key:

1. Stop the running dashboard/Web Bridge process and the Policy Gate:
   `sudo systemctl stop kaia-policy-gate.service`.
2. Generate a new key with `openssl rand -hex 32`. Replace the
   `KAIA_CAPABILITY_TOKEN_SECRET` assignment in the private checkout `.env` with that
   value, and set `.env` to mode `0600`.
3. From the checkout, run `sudo scripts/install_services.sh`. It validates the new key,
   redeploys the reviewed source and rewrites `/etc/kaia/secret.env`. The installer
   deliberately leaves the Policy Gate stopped.
4. Review the installed tree, then record the trusted baseline and start the Gate:
   `sudo /opt/kaia/scripts/kaia-baseline.sh`, followed by
   `sudo systemctl start kaia-policy-gate.service`.
5. Restart the Web Bridge/dashboard process so it reloads the same new secret. The
   checkout launcher sources `.env` at startup; an already-running Python process does
   not hot-reload its imported configuration.

A secret change makes tokens signed with the old key fail verification immediately;
in-flight requests using an old token may be denied. If compromise is suspected, stop
the Gate, preserve and review relevant audit/journal evidence, rotate the key, and
investigate the issuer and host before trusting the restarted deployment. Rotation alone
does not remove a persistent compromise from a process that can still read the key.

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
