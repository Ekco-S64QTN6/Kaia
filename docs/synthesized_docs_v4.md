# Kaia Project Status & Phase 3/4 Implementation Masterplan (v4.0)

## Executive Summary & Refactor Review

The Kaia project has successfully completed its core security refactoring phase. All previous high-priority gaps from version 3.0 of the status roadmap have been resolved and validated through automated testing.

### Recently Completed Work
1. **Capability Token Externalization**: The cryptographic key `CAPABILITY_TOKEN_SECRET` has been completely removed from hardcoded configurations in `core/config.py` and is now strictly loaded from the environment variable `KAIA_CAPABILITY_TOKEN_SECRET`. The system exits immediately with a fatal error if this variable is missing.
2. **Socket Permission Hardening**: The Unix Domain Socket in `security/policy_gate.py` has been secured. Its file permissions are restricted to `0o660`, and its group ownership is explicitly set to the `kaiacord` group via `os.chown` to prevent unauthorized local processes from interacting with the gate.
3. **Pydantic Intent Validation**: In `tests/test_kaia_cli.py`, the CLI generation model has been integrated with the `DiagnosticsRequest` schema. Generated JSON command intents are strictly validated before execution.
4. **Test Suite Stabilization**: All active test suites, including advanced security tests (`tests/test_advanced_security.py`), run cleanly without leaking mock log handlers, achieving a 100% pass rate.
5. **Sensitive File Exclusion**: `/core/personal_context` has been untracked from Git and pinned to the `.gitignore` index.

### Remaining Infrastructure Gaps
- **Separate Daemon Process**: The Policy Gate is currently run via a subprocess manager in `scripts/activate_kaia_env.sh`, but it needs systemd service isolation to prevent process namespace leakage.
- **eBPF-based Telemetry**: `security/telemetry_daemon.py` currently relies on `psutil` polling. To capture high-fidelity events (e.g., transient network connections or process lifecycles), the telemetry engine must be rewritten to interface with Arch Linux's kernel via `bpftrace` or direct `kprobes`.
- **Systemd D-Bus Gio/GLib Integration**: Subprocess calls (like `systemctl show`) should be fully refactored to use D-Bus property lookups via Gio/GLib for efficiency and security.

---

## Phase 3: Local Threat Intelligence Ingestion Architecture

To make Kaia an autonomous system monitor, she requires offline threat analysis capabilities. Performing live, internet-wide scanning (like ZMap) is too resource-intensive and noisy for a workstation. Instead, the threat intelligence layer implements a local passive enrichment system using Shodan snapshots and geo-databases.

### 1. Versioned Data Layout
The local threat intelligence files are organized within the repository storage tree under `storage/threat_intel/` (which resolves relative to `config.SECURITY_DB_PATH`):

```
storage/threat_intel/
├── geoip/
│   └── GeoLite2-City.mmdb          # MaxMind GeoIP binary database (~30MB)
├── cvedb/
│   └── cve.db                      # SQLite CVEDB with NVD repository (~4-10GB)
├── dnsdb/
│   └── dns.db                      # SQLite DNSDB mapping domains/hostnames (~8-20GB)
└── internetdb/
    ├── YYYY-MM-DD/
    │   ├── internetdb.sqlite       # Shodan global IPv4 InternetDB snapshot (~15-35GB)
    │   └── diff.parquet            # DuckDB-computed daily delta file
    └── latest/
        └── internetdb.sqlite -> ../YYYY-MM-DD/internetdb.sqlite
```

### 2. SQLite Database Schemas
*   **Shodan InternetDB (`internetdb.sqlite`)**:
    *   *Table*: `data`
    *   *Schema*: `ip (INTEGER PRIMARY KEY), ports (TEXT), hostnames (TEXT), tags (TEXT), vulns (TEXT), cpes (TEXT)`
    *   *Indexes*: `ip_index ON data(ip)`
*   **Shodan DNSDB (`dns.db`)**:
    *   *Table*: `hostnames`
    *   *Schema*: `hostname (TEXT), domain (TEXT), type (TEXT), value (TEXT)`
    *   *Indexes*: `domain_index ON hostnames(domain)`
*   **Shodan CVEDB (`cve.db`)**:
    *   *Table*: `vulns`
    *   *Schema*: `cve_id (TEXT PRIMARY KEY), cvss (REAL), cvss_version (TEXT), compressed_cve_data (TEXT)`
    *   *Indexes*: `cve_index ON vulns(cve_id)`

### 3. DuckDB Ingestion & Vectorized Delta Pipeline
A nightly ingestion pipeline will download the global Shodan InternetDB SQLite snapshot (~15-35 GB). To avoid CPU-heavy database-wide scans, we use DuckDB to run vectorized OLAP queries directly on Parquet-converted snapshots.

The pipeline calculates scan-over-scan changes to identify emerging infrastructure changes ($\Delta S = S_t \setminus S_{t-1}$):
```sql
-- Compute newly opened ports or added CVEs since the previous day
SELECT ip, ports, vulns
FROM read_parquet('storage/threat_intel/internetdb/*/*.parquet')
WHERE list_contains(ports, 80) AND NOT list_contains(ports, 443);
```

### 4. Real-Time Security Event Consolidation & Maintenance Cycle
The concept of a nightly "Dream Cycle" is a legacy holdover from chatbot architectures. For an autonomous systems administrator, waiting until 3:00 AM to process security incidents is unacceptable. Threat recognition and cognitive state adjustments must happen in real-time:

*   **Real-Time Cognitive Updates**: 
    When a security event is logged in `security_events.db` (e.g., blocked connection, script modification, policy violation), the system immediately triggers a callback to the cognitive core. The core queries local threat intel databases (GeoLite2, CVEDB, InternetDB), generates high-level assertions, and injects them directly into her 50-cap revisable belief store `beliefs.json` in real-time. This immediately updates her `AffectiveState` and active chat prompt context.
*   **Nightly System Maintenance**: 
    Instead of a "dream cycle," nightly scheduled tasks (e.g., at 3:00 AM) are strictly limited to standard data hygiene and automation pipelines:
    1. Retrieve daily updates to MaxMind GeoIP and Shodan InternetDB snapshots.
    2. DuckDB updates the local Parquet diff databases.
    3. Run `VACUUM` and prune security logs older than 30 days from the SQLite databases.

```json
{
  "entity": "203.0.113.42",
  "assertion": "External host scanned local port 22 with known CVE-2026-1234 details.",
  "confidence": 0.95,
  "timestamp": "2026-06-20T03:14:00Z",
  "revisable": true
}
```

---

## Phase 4: Curses Dashboard UI Integration

To provide real-time visibility into Kaia's operations, we will build a custom curses terminal interface tailored to her security agent role, refactoring the design blocks from `docs/kaiamon.py`.

### 1. UI Architecture (Snapshot-Based Rendering)
The dashboard uses a strict separation between background telemetry collectors and the foreground UI renderer:
- **Centralized Collectors**: Four background threads gather metrics and write to thread-safe buffers:
    - **AuditLogCollector**: Reads and parses Policy Gate decisions from the append-only audit ledger (`storage/audit_ledger.json`).
    - **TelemetryDaemonCollector**: Streams process lifecycle alerts, socket bindings, and Script Sentinel file warnings from `security_events.db`.
    - **CognitiveStateCollector**: Pulls the current affective state vector (valence, arousal, energy) and active assertions from `beliefs.json`.
    - **ChatStreamCollector**: Manages the conversation buffer and streams Kaia's real-time model generations.
- **Immutable Snapshot**: Once per frame (100 ms / ~10 FPS), the UI thread captures a `KaiaDashboardSnapshot` dataclass containing current audit logs, cognitive parameters, and conversation buffers, keeping rendering lock-free.
- **Main Curses Thread**: Restricts all terminal drawing calls to the main thread. Proper system signal handlers and `atexit` hooks guarantee standard terminal restoration on exit or crash.

### 2. Interface Layout (Split Action View & Chat Panel)
The layout divides an 80×24 terminal into a top 2/3rds action monitor (16 lines high) and a bottom 1/3rd interactive chat window (8 lines high) using high-aesthetic box borders and neon color pairs:

```
┌───────────────────────────────────────────────┬────────────────────────────┐
│ KAIA ACTION & AUDIT LOGS                      │ AGENT STATE & BOUNDS       │
│ 04:52:10 [GATE] DiagnosticsRequest validated  │ Mood: Vigilant (Security)  │
│ 04:52:10 [EXEC] Running 'ss -tulpn' in bwrap  │ Vector: [-0.80, 1.00, 0.70]│
│ 04:52:12 [SENT] Watchdog alert: new script    │ Status: Vigilant Mode      │
│ 04:52:15 [INTEL] IP 203.0.113.42 scored 15/100├────────────────────────────┤
│ 04:52:15 [GATE] Blocked mitigation request    │ POLICIES & BOUNDS          │
│ 04:52:16 [EXEC] Added drop rule to nftables   │ Effective Level: bwrap     │
│ 04:52:20 [COGN] Assertion added to beliefs    │ Active IP Blocks: 4        │
│ 04:52:21 [INFO] Policy socket normal          │ Token Secret: Loaded       │
├───────────────────────────────────────────────┴────────────────────────────┤
│ INTERACTIVE COGNITIVE CHAT                                                 │
│ Operator: check connection from 203.0.113.42                               │
│ Kaia: External IP 203.0.113.42 has low reputation (15). Blocked connection│
│       attempt. nftables rule drop appended.                                │
│ > _ [Cursor]                                                               │
└────────────────────────────────────────────────────────────────────────────┘
```
- **Neon Colors**: Cyan borders, Magenta titles, Green for active/good status, Yellow for warnings, Red for errors and security blocks.
- **Interactive Chat Input**: The bottom pane contains a text input prompt allowing the operator to send commands or converse with Kaia while system logs stream in real-time above.

### 3. Cognitive Coupling
Anomalous events captured on the dashboard will directly drive the agent's internal cognitive variables:
- **Affective State Vector**: Policy violations, unauthorized script creations, or blocked connections map to her emotional vector. Anomalies trigger immediate arousal spikes ($A \rightarrow 1.0$) and valence drops ($V \rightarrow -0.8$), modifying the interface tone and terminal output widgets dynamically.
- **Proactive Alerts**: If a high-severity incident is captured in the log panel, the chat pane will highlight a warning prompt, prompting the operator for mitigation confirmation.

---

## Phase 5: Enterprise-Grade Active Protection (Guardian Matrix)

To elevate Kaia from an isolated execution engine to a proactive, enterprise-grade security operations daemon (comparable to Wazuh and RunZero), we will define the technical architecture for the **Guardian Matrix** in Phase 5. For detailed kernel syscall signatures, raw packet parsing structs, and AST compilation details, see the accompanying [gemini_design_research.md](file:///home/ekco/github/Kaia/docs/gemini_design_research.md).

### 1. Host Integrity & File Integrity Monitoring (FIM)
Instead of passive log-watching, Kaia needs real-time file header protection and continuous memory auditing:
- **`fanotify` / eBPF Kernel Hooks**: Attach active monitors to kernel syscall paths (`sys_enter_openat2`, `sys_enter_unlinkat`, `sys_enter_write`). When any critical system executable or workspace binary is accessed or modified, the telemetry daemon intercepts the execution flow.
- **Pre-Launch Header Scans**: Before a modified script or tool executes, the daemon hashes the file and runs YARA signature scans on binary headers. If a threat or unauthorized modification is detected, execution is immediately aborted (Fail-Closed) before process context initialization.

### 2. Local Network & Unmanaged Device Tracking
Rather than wrapping unsafe external tools like `nmap` in shell processes (which introduces injection vectors and system noise), the network discovery module utilizes Python raw socket parsing:
- **Passive Layer-2/3 Listeners**: Binds raw sockets to capture local area network broadcasts (ARP requests, mDNS, LLMNR, and NetBIOS).
- **Rogue Host Profiling**: Reconstructs frame payloads to identify and profile unmanaged network interfaces, rogue LAN assets, or inbound gateway routing shifts. This telemetry is serialized as structured JSON and piped directly into her telemetry validator.

### 3. Proactive Tripwires and Host Honeypots
To detect stealth reconnaissance attempts, Kaia exposes decoy interfaces (tripwires) that alert her immediately:
- **File Honey-Tokens**: Leaves dummy configuration files (e.g., decoy credentials or fake environment configs) in untypical system directories. Reading or modifying these paths triggers an instant high-priority alert.
- **Decoy Containers**: Spawns lightweight container ports listening on local interfaces. Any internal network mapping or port scan hitting these decoy ports triggers a policy gate rule to automatically drop the source IP connection using `nftables`.

### 4. Automated IOC, YARA, and Rule Compilers
Kaia will programmatically compile active threat rules from new discoveries:
- **Rule Abstract Syntax Tree (AST)**: Builds an AST compiler inside her rules processor. 
- **Dynamic Rule Generation**: When the DuckDB delta pipeline identifies new malicious IP patterns, vulnerability exposures, or threat actor IOCs, the compiler translates these indicators directly into structurally valid YARA syntax or SIGMA detection rules to update local host scanning configurations dynamically.

---

## Next Phase Implementation Roadmap

### Phase 1 & 2 Cleanup
- [ ] Configure `systemd-tmpfiles` rules for `/run/kaiacord` to avoid permission fallbacks on system boot.
- [ ] Refactor the Policy Gate to run as a separate systemd daemon service rather than an ad-hoc python command.
- [ ] Implement service restart throttling and health checks (i.e. blocking restarts if nominal metrics are healthy).

### Phase 3: Threat Intelligence Ingestion
- [ ] Download and verify the MaxMind GeoLite2 database in `storage/threat_intel/geoip/`.
- [ ] Build the offline Shodan database tables (`internetdb.sqlite`, `dns.db`, `cve.db`) and configure corresponding indexes.
- [ ] Write the Python pipeline to download snapshots and process them into DuckDB Parquet tables.
- [ ] Implement the vectorized DuckDB delta-scanning queries.
- [ ] Implement Real-Time Event triggers to query threat intelligence databases and update `beliefs.json` on event capture.


### Phase 4: Curses Dashboard
- [ ] Build the central UI `layout_manager.py` for terminal coordinate calculations.
- [ ] Implement background collectors (`AuditLogCollector`, `TelemetryDaemonCollector`, `CognitiveStateCollector`, `ChatStreamCollector`).
- [ ] Integrate the UI snapshot logic and write the main curses loop.
- [ ] Wire telemetry alerts to her `AffectiveState` and inner monologue prompt templates.

### Phase 5: Enterprise Security (Guardian Matrix)
- [ ] Implement `fanotify` or eBPF observers for active binary header hashing.
- [ ] Build Python raw socket packet parsing loops (ARP/mDNS/LLMNR/NetBIOS).
- [ ] Configure decoy container honeypots and honey-token file tripwires.
- [ ] Implement the AST rule compiler for automated YARA/SIGMA generations.

