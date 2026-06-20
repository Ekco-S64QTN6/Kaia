# Kaia: Hardened AI Admin Agent & Security Subsystem

Kaia is a strategic, local AI administrative agent designed for Linux environments. Departing from standard conversational assistants, Kaia is structured as a **hardened system administration and security operations layer** with strict security invariants, sandboxing containment, and a deterministic policy gate.

## 🛡️ Core Design Principles

Every decision in Kaia's architecture derives from four non-negotiable axioms:
1. **LLM outputs are untrusted.** All proposed privileges and commands are treated as raw inputs.
2. **All actions are schema-validated.** A deterministic host process validates structured intents.
3. **The host — not the model — makes security decisions.** No LLM is involved in direct execution authorization.
4. **Security policy and personality are decoupled.** Permissions never depend on relationship standing, affinity, or the emotional state of the agent.

---

## 🏗️ Architecture and Isolation Layout

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
                       [ EXPIRING CAPABILITY TOKEN ]
                     (HMAC-SHA256 Signed by Operator)
                                     │
                                     ▼
             /================================================\
            ||      UNIX DOMAIN SOCKET IPC POLICY GATE        ||
            ||         (Fails Closed / Root Owned)            ||
            ||                                                ||
            ||  1. Pydantic Schema Validator                  ||
            ||  2. Token Signature & Expiry Verifier          ||
            ||  3. Deterministic Target Allowlist             ||
             \================================================/
                                     │
                                     ▼
                       [ DETERMINISTIC HOST EXECUTOR ]
                       (No Shell Interpolation / Args Only)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [ Mitigations & Services ]              [ Script Execution ]
       (nftables, systemctl, ss)            (Bubblewrap Sandboxing Tier)
```

### 1. Sandboxing & Isolation (Bubblewrap / systemd-nspawn)
Nested processes are confined via `bwrap` with `--unshare-all --new-session --die-with-parent`. Key directories (SSH credentials, database paths, user histories) are completely excluded from bind-mount targets. Resources are constrained using cgroup ceilings (CPU quota, memory limits, task quotas).

### 2. Typed Policy Gate (Unix Domain Socket)
The agent never runs raw shell commands. Privileged actions flow through the policy gate daemon listening on `/run/kaiacord/policy_gate.sock`.
* **Diagnostics (Green Tier):** Reads `ss`, `ip route`, `nftables list` via direct subprocess calls.
* **Mitigation (Yellow Tier):** Configures direct firewall blocks (`nftables drop`).
* **Service Control (Yellow Tier):** Restarts system units within a static allowlist (e.g. `nginx`, `postgresql`, `ollama`).
* **State Modification (Red Tier):** File modifications restricted to the workspace directory.
* **Script Execution (Red Tier):** Runs scripts confined within Bubblewrap.

### 3. Telemetry & Host Integrity Monitoring (FIM / Guardian Matrix)
Observations are separated from execution. The system runs an active telemetry daemon implementing:
* **Integrity Auditing (FIM)**: Intercepts kernel file transactions via `fanotify` and deep eBPF syscall hooks (`sys_enter_openat2`, `sys_enter_unlinkat`, `sys_enter_write`). If workspace binaries or critical system paths are altered, the daemon executes YARA header checks and halts execution (Fail-Closed) before process context initialization.
* **Network & Device Tracking**: Passive raw socket listeners capture local broadcasts (ARP, mDNS, LLMNR, NetBIOS) to profile LAN assets and detect unmanaged network interfaces.
* **Proactive Honeypots**: Leaves decoy file tokens and spawns decoy network ports. Scanner connections to these traps trigger instant, automated Policy Gate rules to drop the source IP connection using `nftables`.
* **Input Sanitization**: A **Telemetry Sanitizer** filters incoming metrics using character allowlists to block prompt injection attacks.

### 4. Offline Threat Intelligence & Rule Compiler
To evaluate network alerts privately, the threat intel module utilizes MaxMind GeoLite2 databases and versioned Shodan SQLite caches (InternetDB, DNSDB, CVEDB) stored locally under `storage/threat_intel/`. A dynamic DuckDB Parquet pipeline calculates scan-over-scan changes ($\Delta S = S_t \setminus S_{t-1}$) to isolate emerging host exposures, which are programmatically compiled into valid YARA/SIGMA rules via an AST generator.

### 5. Memory Separation & State Modulation
* **`beliefs.json` (Cognitive working memory):** Capped at 50 slots under `storage/beliefs.json`. Real-time event triggers update Kaia's assertions on alert capture, dynamically updating her `AffectiveState` (valence, arousal, energy) and chat prompts.
* **`security_events.db` (Security Ledger):** Uncapped, indexed, append-only SQLite ledger tracking security audits, violations, and fail-closed incidents.

---

## 📂 Project Structure

```
Kaia/
├── main.py                     # Entry point (starts CLI conversational session)
├── README.md                   # Project documentation
├── LICENSE.md                  # MIT License
├── NOTICE.md                   # Third-party licensing notices
├── requirements.txt            # Python dependencies
├── .env                        # Environment configurations (database login credentials)
│
├── core/                       # Core system files
│   ├── config.py               # Shared settings, prompt configurations, & allowlists
│   ├── database_utils.py       # PostgreSQL structured memory utilities
│   ├── kaia_cli.py             # System status gatherer & command generator
│   ├── utils.py                # ANSI coloring & Ollama model helper functions
│   ├── data/                   # General knowledge and persona markdown configuration
│   └── personal_context/       # Personal memory/context files
│
├── security/                   # Hardened security subsystem
│   ├── cognitive_wiring.py     # Affective state vectors (arousal, valence, energy)
│   ├── db.py                   # Security audit database controller
│   ├── host_executor.py        # Safe subprocess execution layer
│   ├── policy_gate.py          # Unix domain socket validation daemon
│   ├── schemas.py              # Pydantic validation schemas
│   ├── telemetry_daemon.py     # eBPF monitoring hooks
│   ├── telemetry_sanitizer.py  # Input sanitization against injection attacks
│   └── threat_intel.py         # GeoLite2 & local reputation intelligence
│
├── tests/                      # Suite of verification tests
│   ├── test_database_utils.py  # DB logic unit tests
│   ├── test_heuristics.py      # Prompt classification tests
│   ├── test_kaia_cli.py        # System status mock tests
│   ├── verify_changes.py       # General imports & status verification
│   ├── verify_security.py      # Socket, signing, and fail-closed logic validation
│   ├── verify_status.py        # Console diagnostics validation
│   └── verify_wrap.py          # Word wrap formatting tests
│
├── scripts/                    # Activation scripts
│   └── activate_kaia_env.sh    # Main service launcher & virtual env activator
│
└── storage/                    # Databases, logs, and threat intelligence cache directory
    ├── beliefs.json            # 50-cap revisable cognitive belief store
    ├── security_events.db      # Append-only security audit ledger
    ├── audit_ledger.json       # Policy Gate audit ledger
    └── threat_intel/           # Offline threat intelligence databases (SQLite, MMDB, Parquet)
```

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
Initialize all local dependencies (Ollama, ChromaDB, Postgres) and launch the CLI session:
```bash
./scripts/activate_kaia_env.sh
```

---

## ⚖️ License
Licensed under the [MIT License](LICENSE.md). Third-party dependencies are detailed in [NOTICE.md](NOTICE.md).
