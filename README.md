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

This snapshot describes code present in the repository, not proof that privileged services are installed, optional kernel integrations work on the target host, or every security boundary has passed end-to-end testing.

### Foundations present in the repository

- **Policy Gate and intent validation:** the out-of-process Unix-socket gate, Pydantic schemas, capability tokens, permission-lattice checks and host-executor dispatch are implemented in the codebase.
- **Host execution controls:** path containment, direct argv execution, allowlists, sandbox wrappers and cgroup limits exist. Their effective behavior still depends on system configuration, executable paths, privileges and the installed deployment.
- **Integrity monitoring:** trusted file baselines, discovered security-module watchlists, alert latching, ledger integrity checks and a lockdown-rate circuit breaker exist. Baseline creation and deployment coverage remain operator responsibilities.
- **Operator surfaces:** the curses dashboard and a loopback web telemetry/command bridge both exist. Mutating commands must remain behind the Policy Gate; the web endpoint is not a general shell interface.
- **Passive discovery and enrichment:** AF_PACKET parsing, local asset storage and optional InternetDB/GeoIP enrichment are implemented. Passive discovery is not active scanning or a complete inventory.

### Limits that must remain explicit

- **Fanotify FIM:** the current implementation uses notification-mode fanotify. It observes filesystem events; it does not block a process pending a scan verdict. Do not describe it as synchronous pre-execution enforcement.
- **eBPF:** coverage depends on installed BCC support, supported hooks, kernel version, capabilities and callback registration. An empty or unavailable event stream is not proof that the host is clean.
- **YARA:** rule generation, isolated validation, runtime scanner activation and file-event scanning are separate parts of the pipeline. A saved rule file alone does not prove activation.
- **Firewall response/recovery:** unit tests prove command construction and replay logic, not that real kernel rules install, UFW reload preserves them, or reboot recovery works.
- **Threat intelligence:** cached and external results are enrichment signals. Missing, stale or unrated records must not be presented as confirmed benign.
- **Fail-closed posture:** this is a required design property, not a blanket assertion that every component already meets it. Use regression tests and controlled host tests as evidence.

For the authoritative target architecture and dated implementation status, see `docs/master_plan.md`. The deployment checklist in `docs/NEXT_STEPS.md` reflects a historical checkpoint and must be verified against the current host.

## Target Capability Matrix

The following are goals and boundaries, not blanket claims that each control is already production-verified.

### 1. Host security and integrity

- **File integrity:** trusted baselines and fanotify event processing exist. Notification-mode fanotify does not hold an originating process for a permission decision; a blocking execution-permission design remains separate work.
- **Process observation:** eBPF telemetry is optional and hook-specific. Verify actual probes on the target kernel; missing events must be treated as missing visibility.
- **Configuration and supply-chain drift:** file-hash baselines exist. Package-manager transaction tracking and behavioral baselines remain future work.

### 2. Passive network discovery

- **Layer-2 visibility:** native AF_PACKET parsing handles supported protocols from traffic observed on the configured interface. Quiet devices may remain undiscovered.
- **Asset profiling:** the local inventory is useful but not a complete or continuously accurate network inventory.
- **Reputation enrichment:** InternetDB and optional GeoIP records enrich observed addresses; cache misses and unrated entries are not clean verdicts. Internet-wide and active scanning remain out of scope.

### 3. Threat detection and defense

- **YARA:** generated rules need isolated validation, runtime activation and live event delivery before they count as active detection.
- **Honeypots:** modules and setup helpers exist, but installation and the full event-to-response path need target-host verification.
- **Telemetry sanitization:** every new ingress path must sanitize untrusted fields before they reach rendering, command construction, logs or model context.

### 4. Platform resilience

- **Tamper detection:** trusted hashes, watchlist discovery and rate-limited response exist; bootstrap and deployment coverage still matter.
- **Firewall enforcement:** Kaia-owned nftables rule management and recovery code exist. Live block, reload-survival, reboot-replay and panic-recovery tests remain necessary.
- **Health/supply-chain monitoring:** independent service supervision, package transaction monitoring, alert-confidence scoring, secrets scanning and backup restorability remain unfinished capabilities.

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

`docs/master_plan.md` is the authoritative architectural specification and roadmap. This README is the user-facing overview. Archived reviews and secondary design documents are historical inputs; reconcile them against the master plan and actual implementation before treating them as current requirements.
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
export KAIA_CAPABILITY_TOKEN_SECRET="your_signing_secret_key"
```

### 3. Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Running Verification Tests

Install the focused test dependencies and run the regression suite from the repository root:

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

The suite does not replace target-host checks for BCC/eBPF, fanotify, live nftables, systemd transient services or privileged recovery.bash
python tests/verify_security.py
python tests/test_heuristics.py
```

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

That deploys a root-owned copy to `/opt/kaia` (0644 files, 0755 dirs, root:root),
provisions `/var/lib/kaia` for state, migrates any in-repo ledger, and installs the
units. Re-run it after changes — the checkout is the source, `/opt/kaia` is what runs.

### Privileges

| Capability | Status | Why |
|---|---|---|
| `CAP_NET_ADMIN` | granted | nftables rules for IP blocking |
| `CAP_NET_RAW` | granted | `AF_PACKET` sockets for passive discovery |
| `CAP_DAC_OVERRIDE` | **dropped** | Bypasses *all* file permission checks. Only ever needed to read a `0600 .env` owned by an unprivileged user; the secret now comes from root-owned `/etc/kaia/secret.env`. |
| `CAP_SYS_ADMIN` | **dropped** | Needed only for mount-wide `fanotify` and BCC/eBPF. Both were non-functional as shipped — fanotify failed `EINVAL` on every start, BCC is not installed — so it granted privilege for nothing. |

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

nftables rules are runtime-only, so a reboot silently clears every address Kaia has
blocked. `kaia-restore-blocks.service` replays approved `block_ip` entries from the
ledger at boot, before the Policy Gate starts accepting new requests. It re-uses
`HostExecutor.execute_mitigation`, so table, chain and address-family handling have a
single implementation. Malformed ledger lines are skipped rather than fatal — a corrupt
ledger must not stop the host booting with its blocks in place.

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
before flushing, plus a timestamped copy for forensics. To lift it:

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
drift apart.

With no baseline present the daemon still runs, but logs a warning at startup that
modifications made while it was stopped cannot be detected.

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
