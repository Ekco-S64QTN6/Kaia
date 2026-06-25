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
5. **Every security boundary fails closed.** Any state validation mismatch, IPC disruption, or configuration anomaly halts execution and revokes capability tokens instantly.

---

## 📈 Target Implementation Status

Kaia is modeled around its completed architectural state, maintaining stable components alongside active target tracking matrices:

* **Policy Gate:** Fully implemented. Handles deterministic out-of-process verification over Unix Domain Sockets.
* **Intent Validation:** Fully implemented. Rigidly enforces Pydantic schemas on incoming JSON payloads.
* **Capability Tokens:** Fully implemented. Restricts actions dynamically based on signed cryptographic operational permissions.
* **Audit Ledger:** Fully implemented. Maintains an append-only transaction stream to `security_events.db` and `audit_ledger.json`.
* **Host Telemetry & Observers:** Operating via live `fanotify` mount-wide monitors and deep eBPF syscall hooks (`sys_enter_execve`, `sys_enter_openat2`) to prevent polling latency gaps.
* **Network Threat Intelligence:** Shell-less Layer-2/Layer-3 discovery tracking via native `AF_PACKET` socket interceptors alongside a cache-first, live Shodan InternetDB API synchronization framework.

---

## 📊 Planned Capability Matrix

### 1. Host Security & Integrity Monitoring
* **File Integrity Monitoring (FIM):** Continuous mount-wide event processing via `fanotify` bindings with synchronous pre-execution header verification blocks.
* **Process Observers:** Direct eBPF event stream capturing for process lifecycles and real-time execution context derivation.
* **Configuration Drift Enforcement:** Structural verification of core platform states against established baseline configurations.
* **Supply Chain Isolation:** Strict namespace tracking over system update hooks and package management transactions.

### 2. Shell-less Network Discovery
* **Layer-2/Layer-3 Active Mapping:** Raw socket processing loops capturing ARP, mDNS, and LLMNR broadcast frames directly from interfaces.
* **Unmanaged Device Profiling:** Programmatic local area network asset tracking without relying on external wrapped tools or shell utilities.
* **Reputation Enrichment:** Cache-first, automated lookup cycles utilizing local SQLite stores and live fallback synchronization routines.

### 3. Threat Detection & Proactive Defense
* **Signature Enforcement:** Native compilation and live processing of YARA rules across modified execution targets.
* **Internal Honeypots:** Low-overhead network tripwires and local configuration honey-tokens designed to isolate internal traversal vectors.
* **Telemetry Sanitization:** Pre-processing and striping of high-frequency metrics before passing payloads to AI context layers.

### 4. Platform Resilience
* **Self Health Monitoring:** Independent supervisor daemons verifying the integrity and execution states of core validation processes.
* **Tamper Detection:** Defensive validation checks looking for runtime interference or alterations within privileged code spaces.
* **Fail-Closed Isolation:** Native `nftables` ruleset flushes dropping all inbound, forward, and outbound packets during an active compromise assertion.

---

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

## 📂 Repository Topology

```
Kaia/
├── core/
│   ├── config.py               # Central configuration & base directory logic
│   ├── data/                   # Data directory for RAG indexing
│   └── personal_context/       # Personal context directory
├── security/
│   ├── policy_gate.py          # Deterministic out-of-process validation service
│   ├── host_executor.py        # Privileged runtime sandboxing and command handler
│   ├── telemetry_sanitizer.py  # Filters metrics before feeding to the LLM
│   └── db.py                   # Secure ledger handling
├── scripts/
│   └── activate_kaia_env.sh    # Main setup and daemon activation sequence
├── tests/
│   ├── verify_security.py      # Multi-tiered sandbox and lattice constraint tests
│   └── test_heuristics.py      # Classifier accuracy validation suite
├── toolbox/                    # System diagnostics utilities
└── storage/
├── security_events.db      # Append-only security audit ledger
├── audit_ledger.json       # Policy Gate audit ledger
└── threat_intel/           # Offline threat intelligence databases (SQLite, MMDB, Parquet)
```
---

## ⚖️ Source of Truth Constitutional Clause

This `README.md` serves as the absolute, unalterable constitution of the Kaia project. It defines the definitive project goals, non-negotiable security axioms, architectural boundaries, and target capability frameworks. 

All detailed design reviews, feature proposals, temporary scratchpads, and execution reports are subordinate to this document. If any conflict arises between implementation artifacts or secondary planning documentation and this constitution, **this file takes immediate precedence**. Automated coding agents and systems engineers are directed to baseline all development tracking against the criteria defined herein.

---

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

### 4. Running Verification Suite
Confirm the security boundaries and IPC communications are functioning correctly:
```bash
python tests/verify_security.py
python tests/test_heuristics.py
```

### 5. Launching Kaia
Initialize all local dependencies (Ollama, Postgres) and launch the CLI session:
```bash
./scripts/activate_kaia_env.sh
```

---

## ⚖️ License
Licensed under the [MIT License](LICENSE.md). Third-party dependencies are detailed in [NOTICE.md](NOTICE.md).
