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
* **Service Control (Yellow Tier):** Restarts system units within a static allowlist (e.g. `nginx`, `postgresql`, `ollama`, `chroma`).
* **State Modification (Red Tier):** File modifications restricted to the workspace directory.
* **Script Execution (Red Tier):** Runs scripts confined within Bubblewrap.

### 3. High-Fidelity Local Telemetry Pipeline
Observations are separated from execution. A telemetry daemon streams system metrics, process lifecycles via eBPF probes (`execve`, `tcp_connect`), and systemd D-Bus states. A **Telemetry Sanitizer** filters fields using static character allowlists to prevent prompt injections via attacker-controlled network strings.

### 4. Separated Memory
* **`beliefs.json` (Cognitive Store):** Manages social and personality memory (social preferences, conversation history). Subject to decay.
* **`security_events.db` (Security Ledger):** Append-only database tracking security incidents (violations, failed validation, fail-closed events). Immutable and protected from agent modifications.

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
│   └── utils.py                # ANSI coloring & Ollama model helper functions
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
├── toolbox/                    # Utility tools
│   └── video_converter.py      # Video-to-GIF converter utility
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
├── data/                       # general knowledge and persona markdown configuration
└── storage/                    # ChromaDB, LlamaIndex, & PostgreSQL cache directory
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
