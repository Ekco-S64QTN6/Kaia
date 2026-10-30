# Kaia — hardened AI security agent for Linux

[![Platform](https://img.shields.io/badge/platform-Arch%20Linux-informational)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE.md)
[![Regressions](https://github.com/Ekco-S64QTN6/Kaia/actions/workflows/security-regressions.yml/badge.svg?branch=main)](https://github.com/Ekco-S64QTN6/Kaia/actions/workflows/security-regressions.yml?query=branch%3Amain)

Kaia is a local-first, zero-trust AI agent that monitors, analyses and defends a Linux host while
treating everything the language model says as untrusted input. The model proposes; a
deterministic, out-of-process Policy Gate decides; a hardened executor acts. The model never gets a
shell.

> *Can an AI help defend a Linux system without becoming a security risk itself?*

## The five axioms

1. **LLM output is untrusted.** Every proposed action, privilege and script is unverified data.
2. **Every action is schema-validated.** A deterministic host process validates structured intents (Pydantic) before evaluating them.
3. **The host decides, not the model.** No model is involved in authorising execution or matching paths.
4. **Policy and personality are decoupled.** Permissions never depend on rapport, affinity or the agent's mood.
5. **Every boundary fails closed**, with tests for its failure behaviour.

## How an action runs

```
   [ Operator ] ──► [ Intent classifier ] ──► [ LLM planner ]
                                                     │  structured intent (action, args, justification)
                                                     ▼
                    [ Capability tokens ] ──► [ Policy Gate ]   out of process, deterministic
                                                     │  approved schema only
                                                     ▼
                                           [ Host executor ]
                                     ┌───────────────┴───────────────┐
                                     ▼                               ▼
                          Bubblewrap sandbox              state change behind strict
                        (isolated filesystem)              path validation
```

## What is built

| Area | |
|:--|:--|
| **Policy Gate** | Typed schemas, expiring HMAC-signed capability tokens scoped to a target, a permission lattice, Unix-socket IPC, an append-only audit ledger, allowlisted argv execution and symlink-resistant atomic workspace writes |
| **Firewall** | Block and unblock through the gate; unblock removes only Kaia's exact nftables handles; boot replay folds the ledger so a removal stays removed |
| **Integrity** | A trusted, operator-recorded baseline; tamper detection with alert latching and a lockdown circuit breaker; system mode refuses to start without the baseline |
| **Telemetry** | eBPF (execve, openat2, connect/retransmit, privilege transitions, unlinkat attempts) and fanotify file monitoring with same-descriptor YARA scanning |
| **Discovery and intel** | Passive ARP, mDNS, LLMNR, SSDP and NBNS from the wire; InternetDB enrichment; pacman transaction monitoring |
| **Decoys** | Non-destructive filesystem honeypots that only ever remove what they created |
| **Operator HUD** | A loopback web HUD — telemetry waterfall, discovery radar, token widget, the Kaia codec avatar and a command console — behind a strict self-hosted CSP and Host/Origin checks; and a curses dashboard |

Each claim is at the level the code and its tests support. Kernel-dependent controls (eBPF,
fanotify, nftables, systemd) are only proven on the deployed host. What is left, starting with that
host verification: **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Quick start

Arch Linux, PostgreSQL, Bubblewrap and [Ollama](https://ollama.com).

```bash
sudo pacman -S python python-pip postgresql bubblewrap
sudo systemctl enable --now postgresql
sudo -u postgres createuser --pwprompt kaiauser && sudo -u postgres createdb -O kaiauser kaiadb

python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Put the database credentials and a fresh signing secret in `.env`:

```bash
export KAIA_DB_USER="kaiauser" KAIA_DB_PASS="…" KAIA_DB_HOST="localhost" KAIA_DB_NAME="kaiadb"
export KAIA_CAPABILITY_TOKEN_SECRET="…"   # exactly 64 hex characters: openssl rand -hex 32
```

The installer rejects a missing key, a malformed one, and the historically leaked one. Then:

```bash
./scripts/activate_kaia_env.sh     # start Ollama and Postgres, launch the CLI session
```

Country annotations on blocked IPs need a free MaxMind key: `MAXMIND_LICENSE_KEY=… ./scripts/update_geoip.sh`.

**Deploying for real** — a root-owned copy under `/opt/kaia`, state under `/var/lib/kaia`, a trusted
baseline, the systemd units — is in [Operations](docs/OPERATIONS.md), with the acceptance tests in
the [deployment checklist](docs/DEPLOYMENT_CHECKLIST.md).

The capability-token signing secret is an authorization root; [Operations](docs/OPERATIONS.md#capability-token-trust-and-rotation) documents its custody, limitations and rotation.

> **Stop the Policy Gate before editing Kaia's own source.** Tamper detection re-hashes its core
> files every 30 seconds; an edit while it runs is indistinguishable from an intruder and triggers a
> lockdown that drops all traffic. `sudo systemctl stop kaia-policy-gate`, edit, then start it to
> re-baseline. If the network goes down anyway: `sudo ~/kaia-panic-unlock.sh` (see
> [Operations](docs/OPERATIONS.md#if-the-network-goes-down-unexpectedly)).

## Tests

```bash
python -m pip install -r requirements-test.txt
bash scripts/run_ci_tests.sh
```

The same deterministic suite runs in GitHub Actions. It does not replace target-host checks of
BCC/eBPF, fanotify, live nftables, systemd or privileged recovery.

## Documentation

| | |
|:--|:--|
| [Architecture specification](docs/master_plan.md) | Invariants, component design, and the implementation status in §10.5 |
| [Roadmap](docs/ROADMAP.md) | What is left, host verification first |
| [Operations](docs/OPERATIONS.md) | Deployment, privileges, capability-token trust and rotation, the ledger, block persistence, lockdown recovery and the baseline |
| [Deployment checklist](docs/DEPLOYMENT_CHECKLIST.md) | Step-by-step acceptance tests for a deployed host |
| [Security review process](docs/security_review_process.md) | What a security change needs before it merges |
| [Documentation map](docs/README.md) | Which document is authoritative for what |

## Repository layout

```text
core/        configuration, CLI orchestration, the web bridge and HUD (core/web/static/)
security/    Policy Gate, host executor, tamper detection, eBPF, FIM, threat intel, supply chain
scripts/     install, baseline, lockdown and recovery, maintenance
tests/       security, IPC, dashboard and regression tests
docs/        specification, roadmap, operations, checklists, design references
```

## License

[MIT](LICENSE.md). Third-party dependencies are listed in [NOTICE.md](NOTICE.md).
