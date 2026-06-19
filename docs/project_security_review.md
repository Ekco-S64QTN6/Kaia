# Kaia Project Security Review & Mitigation Check

This document provides a comprehensive security review and gap analysis of the Kaia repository against the **Hardened AI Admin Agent v2** specification.

---

## 🔍 Executive Summary

While the project has successfully reorganized into a cleaner repository structure, several critical architectural and implementation gaps exist that violate the security invariants specified in [Hardened AI Admin Agents_v2.md](file:///home/ekco/github/Kaia/Hardened%20AI%20Admin%20Agents_v2.md). 

Most notably, the current containment and execution models present avenues for **Remote Code Execution (RCE)** and **Privilege Escalation** in the event of a prompt injection attack.

---

## 🛑 Critical Gaps & Vulnerabilities

### 1. Insecure State Modification (`write_file`) — Self-Modification & RCE
* **Spec Alignment:** Section 2.2 defines State Modification as a "Red" tier action restricted by "sandbox containment".
* **Vulnerability:** `HostExecutor.execute_state_modification` validates only that the target path starts with the workspace root directory. It allows writing to **any file** inside the workspace.
* **Risk:** A compromised agent can overwrite `main.py`, `core/config.py`, or any file in `security/` with arbitrary Python code. Since these run in the main application loop, this results in immediate arbitrary code execution (RCE) on the host. It also allows overwriting `.env` to alter security configurations.
* **Severity:** **CRITICAL**

### 2. Sandbox Leakage in Bubblewrap (`run_script`) — Credential & Data Exposure
* **Spec Alignment:** Section 1.3 states that "High-value assets — SSH credentials, database files, shell histories, local keys — are completely excluded from bind-mount targets."
* **Vulnerability:** The `bwrap` command in `HostExecutor.execute_script` bind-mounts the entire `config.WORKSPACE_DIR` read-write.
* **Risk:** The workspace directory contains `.env` (which holds PostgreSQL passwords) and the `storage/` directory (which holds the SQLite security events database and LlamaIndex/ChromaDB files). Any sandboxed script can read the database credentials from `.env` and read/write raw database files, completely bypassing database access control boundaries.
* **Severity:** **HIGH**

### 3. In-Process Policy Gate Threading — Memory Boundary Defeat
* **Spec Alignment:** Section 2.3 and Section 9 describe the policy gate as a "deterministic host process" listening on a Unix socket, separate from the agent.
* **Vulnerability:** Currently, `main.py` starts `PolicyGate` as a background `threading.Thread` within the *same* process as the LLM-driven agent application.
* **Risk:** If the Python agent process is compromised (e.g., via a library exploit or python code execution), the policy gate is in the same memory address space. An attacker can manipulate python runtime globals, disable socket validations, sign fake capability tokens, or hook `HostExecutor` methods directly in memory, bypassing the gate.
* **Severity:** **HIGH**

### 4. Telemetry Daemon Fallback — Polling vs eBPF Probes
* **Spec Alignment:** Section 3.3 specifies a table of tracepoints and kprobes (`sys_enter_execve`, `tcp_connect`, `tcp_retransmit_skb`, `sys_enter_openat`, `sys_enter_setuid`).
* **Vulnerability:** The current `telemetry_daemon.py` relies entirely on polling using `psutil`. It does not install eBPF hooks or handle bpftrace streaming.
* **Risk:** Attacker actions that happen between polling intervals (e.g., quick script execution, fast socket connections) are invisible to the agent.
* **Severity:** **MEDIUM**

### 5. Threat Intelligence Databases & Vector Check
* **Spec Alignment:** Section 6.2 specifies local SQLite databases for Shodan InternetDB (~15-35 GB), DNSDB (~8-20 GB), CVEDB (~4-10 GB), and GeoIP.
* **Vulnerability:** While `threat_intel.py` implements the query interface, the database files are missing or default to fallbacks. Vectorized DuckDB calculations are not implemented.
* **Severity:** **LOW / FEATURE DEFERRED**

---

## 🛠️ Proposed Mitigations & Patches

We will implement the following mitigations immediately:

| Mitigation Target | Implementation Strategy | Status |
|---|---|---|
| **State Modification Protection** | Block `write_file` on `*.py`, `*.sh`, hidden files (`.env`, `.git`), and system folders (`core/`, `security/`, `tests/`, `scripts/`, `storage/`). | **Planned** |
| **Bubblewrap Isolation Masking** | Mask `.env` and the `storage/` directory inside Bubblewrap using `/dev/null` masking and `tmpfs` mounts. | **Planned** |
| **Fail-Closed Verification** | Enhance test suites to verify path containment and file write rejections. | **Planned** |
