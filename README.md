# Kaia: Hardened AI Admin Agent & Security Subsystem

[![Status](https://img.shields.io/badge/status-active%20development-blue)](#)
[![Platform](https://img.shields.io/badge/platform-Arch%20Linux-informational)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Policy Gate](https://img.shields.io/badge/Policy_Gate-Implemented-success)](#)
[![Pydantic Validation](https://img.shields.io/badge/Pydantic_Validation-Implemented-success)](#)
[![Capability Tokens](https://img.shields.io/badge/Capability_Tokens-Implemented-success)](#)
[![Audit Logging](https://img.shields.io/badge/Audit_Logging-Implemented-success)](#)
[![Fail Closed](https://img.shields.io/badge/Fail_Closed_Design-Implemented-success)](#)

---

## 👁️ Vision

Kaia is a local-first, zero-trust AI security agent designed to monitor, analyze, and defend Linux systems while treating all LLM output as untrusted input. 

Unlike traditional conversational assistants, Kaia is structured as a hardened system administration and security operations layer with strict security invariants, sandboxing containment, and a deterministic policy gate. The objective is to create a guardian-class security platform combining advanced AI reasoning with deterministic authorization, real-time host telemetry, threat intelligence, and human oversight without granting direct, unchecked shell authority to the language model.

---

## 🛡️ Mission Statement

Kaia exists to answer a simple question:
> *Can an AI help defend a Linux system without becoming a security risk itself?*

Every structural decision in Kaia's architecture derives from five non-negotiable axioms:
1. **LLM outputs are completely untrusted.** All proposed actions, privileges, and scripts are treated as raw, unverified data streams.
2. **All actions are schema-validated.** A deterministic host process validates structured intents using rigorous Pydantic definitions before evaluation.
3. **The host—not the model—makes execution decisions.** No LLM is involved in direct execution authorization or environment path matching.
4. **Security policy and personality are completely decoupled.** Permissions and capability boundaries never depend on relationship standing, affinity, or the affective state of the agent.
5. **Fail-closed enforcement is mandatory.** Each boundary needs explicit failure behavior and tests; claims about effective containment require evidence from the relevant tests and target-host validation.

---

## Implementation Status — repository review snapshot (2026-10-22)

This describes capabilities present in the repository. It is not proof that privileged services are installed on a particular host or that every kernel-dependent control works there.

### Implemented in the codebase

- **Policy Gate and host execution:** typed request schemas, expiring capability tokens, permission-lattice checks, Unix-socket IPC, audit logging, allowlisted direct-argv execution, path containment and sandbox/resource controls.
- **Firewall lifecycle:** the block and unblock commands are routed through the Policy Gate. Unblock removes exact matching Kaia-owned nftables rules by handle; boot replay folds approved block/unblock events so a successful removal does not resurrect an older block.
- **Integrity monitoring:** trusted-baseline validation, watchlist discovery, tamper alerts and lockdown circuit breaking are implemented. System mode refuses to start without a valid trusted baseline; only unprivileged development mode may bootstrap from current disk state.
- **File event handling:** the fanotify handler keeps the original event descriptor open, scans and hashes the same captured bytes, and caps the snapshot at 64 MiB. Its audit database uses the configured `KAIA_STATE_DIR` (`fim_audit.db`), and the installer migrates legacy history from `/var/lib/secdaemon/fim_audit.db`. YARA validation fails closed when the isolated validator is unavailable, and generated IOC strings are escaped as literals.
- **eBPF telemetry:** the probe source includes execve, openat2, network-connect/retransmit, privilege-transition and unlinkat events. Unlinkat records are surfaced in the waterfall as syscall-entry attempts, not proof a deletion succeeded.
- **Passive discovery:** ARP, IPv4/UDP mDNS, LLMNR and SSDP observations are recorded from traffic on the selected interface; basic NetBIOS Name Service response QNAME decoding is included. It is not a complete inventory and query names are not attributed to queriers.
- **Threat intelligence:** InternetDB input validation, per-process serialized requests and negative caching for authoritative 404 responses are implemented alongside the existing reputation/CVE lookup paths.
- **Package transactions:** a passive pacman-log monitor records install/upgrade/downgrade/reinstall/remove events after successful ALPM transaction completion. It is a first increment, not full supply-chain attestation.
- **Operator UI:** the loopback-only web HUD has interactive command entry, bounded command dispatch through the Policy Gate, telemetry WebSocket updates, strict self-hosted CSP, and Host/Origin validation for loopback host, scheme and effective port on command and telemetry endpoints. Command results are returned after the request completes; incremental response streaming for long-running operations remains unfinished.
- **Regression automation:** GitHub Actions compiles Python modules and runs the focused security regression suite. The latest recorded main run on 2026-10-22 passed 176 tests. This does not replace privileged host validation.

### Still incomplete or host-dependent

- **Execution-blocking FIM:** the current implementation uses FAN_CLASS_NOTIF. It observes file events but does not hold the originating process for a scan verdict.
- **Effective kernel telemetry:** full eBPF visibility depends on BCC, kernel support, available capabilities and successful probe registration. Empty telemetry is not a benign verdict.
- **End-to-end detection enforcement:** YARA rule validation and event scanning code exist, but the whole event-to-verdict-to-response path still requires target-host verification.
- **Broader roadmap:** expanded eBPF/FIM coverage; broader NBNS/NBSTAT parsing and edge-case coverage; AUR/repository/dependency graph monitoring; behavioral baselines; alert-confidence scoring; secrets-exposure scanning; backup restorability checks; independent service/data-freshness supervision; and expanded offline GeoLite2/CVE/passive-DNS threat intelligence remain unfinished.
- **Production proof:** the installed systemd units, trusted baseline, firewall block/unblock behavior, UFW reload survival and reboot replay must be verified on the actual Arch Linux host.

For the full implementation/verification matrix and remaining priorities, see docs/master_plan.md. The deployment checklist is a dated historical checkpoint and must be reconciled with the current host before running privileged commands.

## 🏗️ Architecture and Isolation Layout

The execution security boundary strictly prevents the model from interacting with the operating system shell directly:

```
                    [ User Input / Shell Query ]
                                 │
                                 ▼
                    [ Heuristic Intent Classifier ]
                                 │
                                 ▼
                    [ LLM Action Plan Planner ]
                                 │
                                 ▼
                    [ Structured Intent Payload ]
                     (JSON: action, args, justification)
                                 │
                                 ▼
                    [ Deterministic Policy Gate ] ◄── [ Capability Tokens ]
                                 │
                                 ▼
                    [ Hardened Host Executor ]
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         [ Sandboxed Execution ]    [ State Modification ]
          (Bubblewrap Sandbox)       (Strict Path Validation)
```

### The Definitive Core Operational Pipeline:
1. The user interfaces with the orchestration layer.
2. The AI agent proposes an action plan as a structured intent payload.
3. The out-of-process, deterministic Policy Gate evaluates the payload against capability tokens and static system constraints.
4. The privileged Host Executor consumes the approved schema.
5. Target binaries run inside unprivileged Bubblewrap namespaces with isolated filesystems, or apply strict path filters for state changes.

---

## Repository Topology

```text
Kaia/
├── core/
│   ├── config.py              # Runtime configuration and state paths
│   ├── kaia_cli.py            # CLI / structured-intent orchestration
│   ├── web_bridge.py          # Loopback HTTP, WebSocket and command bridge
│   └── web/static/            # Local dashboard HTML, CSS, JavaScript and Kaia codecs
├── security/
│   ├── supply_chain_monitor.py # Passive pacman transaction monitor
│   ├── policy_gate.py         # Token/schema checks and action authorization
│   ├── host_executor.py       # Privileged command and state-operation boundary
│   ├── tamper_detection.py    # Trusted baselines and runtime integrity checks
│   ├── ebpf_telemetry.py      # Optional kernel event collection
│   ├── fim_daemon.py          # Fanotify-based file monitoring
│   └── threat_intel.py        # Local enrichment cache and InternetDB client
├── scripts/                   # Deployment, baseline, recovery and maintenance
├── tests/                     # Security, IPC, dashboard and regression tests
├── docs/                      # Master plan, next steps and historical reviews
├── kaia_dashboard.py          # Snapshot collectors and curses dashboard
└── main.py                    # Entry point / fallback tooling
```

Runtime databases, caches, audit ledgers and secrets are configuration-dependent and may live outside the checkout in system mode. They should not be assumed to be tracked repository files.

## Project source of truth

`docs/README.md` explains the documentation hierarchy. `docs/master_plan.md` is the authoritative architecture and current repository-status record; `docs/NEXT_STEPS.md` is the target-host acceptance checklist. This README is the user-facing overview. Files under `docs/archived/` and dated execution reports are historical evidence, not current requirements or proof of present host state.
---

> **Deployment state requires local verification.** The last recorded setup checklist is dated 2026-09-08. This repository cannot determine whether the services have since been installed or tested on the target host. Verify the active unit, deployment path, state directory, trusted baseline and firewall recovery before relying on them. See **[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)**.

## 🚀 Getting Started

### 1. Prerequisites (Arch Linux)
```bash
sudo pacman -S python python-pip postgresql bubblewrap
sudo systemctl enable --now postgresql
```

### 2. Database Configuration
Create the PostgreSQL database and user:
```bash
sudo -u postgres createuser --pwprompt kaiauser
sudo -u postgres createdb -O kaiauser kaiadb
```
Add credentials to your `.env` file in the root directory:
```bash
export KAIA_DB_USER="kaiauser"
export KAIA_DB_PASS="your_secure_password"
export KAIA_DB_HOST="localhost"
export KAIA_DB_NAME="kaiadb"
# Generate 32 random bytes (64 hex characters): openssl rand -hex 32
# Paste that command's output verbatim in place of the placeholder below.
export KAIA_CAPABILITY_TOKEN_SECRET="REPLACE_WITH_64_HEX_CHARACTERS_FROM_openssl_rand_-hex_32"
```

The installer rejects a missing key, the historically leaked key, and anything other
than exactly 64 hexadecimal characters. Do not use the illustrative placeholder as a
real signing secret.

### 3. Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Running Verification Tests

Install the focused test dependencies and run the same deterministic regression suite used by GitHub Actions:

    python -m pip install -r requirements-test.txt
    bash scripts/run_ci_tests.sh

The runner explicitly selects deterministic, non-privileged tests. Legacy and live-host integration scripts are not part of this suite because some require installed systemd services, populated runtime databases, or heavyweight optional application dependencies. The regression suite does not replace target-host checks for BCC/eBPF, fanotify, live nftables, systemd transient services or privileged recovery.



### 5. (Optional) GeoIP Database
To enable country annotations on blocked IPs, obtain a free MaxMind license key
from https://www.maxmind.com and run:
```bash
export MAXMIND_LICENSE_KEY="your_key_here"
./scripts/update_geoip.sh
```
Without this the geo field displays "Unknown" — all other functionality is unaffected.

### 6. Launching Kaia
Initialize all local dependencies (Ollama, Postgres) and launch the CLI session:
```bash
./scripts/activate_kaia_env.sh
```

---

## 🔧 Maintenance & Development

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

### Tamper-protected files

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

### Deployment model

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

### Privileges

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

### State and the audit ledger

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

### Block persistence

nftables rules are runtime-only. `kaia-restore-blocks.service` reconstructs the
currently active Kaia-managed blocks from the audit ledger before the Policy Gate
accepts new requests. Replay folds approved block_ip and unblock_ip events in
chronological order: a successful unblock is a tombstone for the earlier rule, while
a later approved re-block restores it. Malformed records are skipped and logged.
This behavior is covered by ledger tests, but boot-time replay and the live nftables
path still need verification on the target host.

Use `unblock <ip>` through the dashboard for a normal removal. Do not manually
delete the table during a replay test and then assume the ledger agrees: without an
approved unblock event, boot replay is expected to restore an active ledger block.

### If the network goes down unexpectedly

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

### Recovering from a lockdown

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

### Trusted integrity baseline

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

System mode refuses to start without a valid trusted baseline. Only unprivileged
development mode may bootstrap from the current disk state, where changes made while
the daemon was stopped cannot be detected. Record a baseline only after reviewing
the deployed source and deployment state; never treat bootstrap hashing as equivalent
to verification against a previously trusted baseline.

### Monitor-only mode

For tuning, or for working on the project without arming the tripwire:

```bash
KAIA_TAMPER_ENFORCE=0 python security/policy_gate.py
```

Detection and audit logging continue; lockdown is suppressed. This disables an active
security control, so it is opt-in and logged as CRITICAL at startup.

### Lockdown circuit breaker

At most **3 lockdowns per hour** (`LOCKDOWN_MAX_PER_WINDOW` / `LOCKDOWN_WINDOW_SECONDS`).
Beyond that, tampering is still detected and recorded but lockdown is suppressed, so
the host stays reachable to investigate through. Every lockdown flushes the ruleset;
an unbounded burst leaves you with no network and no firewall at the worst moment.

**Alert latching:** one modification produces one CRITICAL alert and one lockdown.
Subsequent checks log at WARNING without re-firing, and the detector re-arms when the
file returns to baseline. Without this a single event re-flushed the firewall every
30 seconds, which is precisely when an operator needs the network to investigate.

---

## ⚖️ License
Licensed under the [MIT License](LICENSE.md). Third-party dependencies are detailed in [NOTICE.md](NOTICE.md).
