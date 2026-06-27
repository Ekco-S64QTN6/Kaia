# Kaia — Tier 5 Action Plan: Enterprise Guardian Matrix
**Source of truth:** `docs/master_plan_draft.md`  
**Prerequisite:** All Tier 1–3 blocks from the main action plan must be complete and green before starting any block here.  
**Date:** 2026-06-25

Tier 5 is the kernel-boundary and active defense layer. It replaces psutil polling with real event streams, deploys hardware decoys, adds a YARA rule pipeline, and wires an emergency lockdown. Each block is independent enough to be assigned to a separate agent pass but they have the dependency order noted.

---

## Block T5-1 — Prerequisite: system dependency install manifest

**Files:** `scripts/install_tier5_deps.sh` (new), `requirements.txt`

Before any other block can be implemented the host needs the right packages. Create a documented install script the agent and operator can both read.

**Arch Linux packages to install via pacman:**
- `bcc` — BCC/BPF compiler collection (includes Python bindings at `/usr/lib/python3.x/site-packages/bcc`)
- `bpf` — kernel BPF tools
- `libbpf` — low-level BPF library
- `linux-headers` — kernel headers needed to compile BPF programs (must match running kernel)
- `yara` — YARA binary for rule validation in ephemeral sandboxes (§7.3)
- `python-yara` — Python bindings for YARA scanning
- `maxminddb-c` / `python-maxminddb` — for GeoLite2 MMDB reads (§threat_intel.py already imports maxminddb)
- `nftables` — already present but confirm `libnftables` shared lib is available for Python bindings

**Python packages to add to `requirements.txt`:**
```
bcc                    # BPF/eBPF Python bindings (installed system-wide via pacman, add as comment)
yara-python>=4.3.0     # YARA rule compilation and scanning
maxminddb>=2.4.0       # GeoLite2 MMDB reader (already imported in threat_intel.py, add formally)
```

The install script must check that `uname -r` matches an installed `linux-headers` package before proceeding. If they don't match, print a fatal error and exit — mismatched headers cause silent BPF compilation failures.

**Acceptance:** `python -c "import bcc; import yara; import maxminddb"` exits 0.

---

## Block T5-2 — fanotify FIM subsystem (§4.1)

**Files:** `security/fim_daemon.py` (new), `security/policy_gate.py`, `requirements.txt`

Replace the `watchdog`-based `ScriptSentinelHandler` in `telemetry_daemon.py` with a proper `fanotify` implementation. The watchdog stays as a fallback for environments where fanotify isn't available (CI, containers), but the primary path is kernel-native.

**New file: `security/fim_daemon.py`**

Implement `FIMDaemon` class:

- Uses `ctypes` to open a fanotify fd: `libc.fanotify_init(FAN_CLASS_NOTIF | FAN_CLOEXEC | FAN_NONBLOCK, O_RDONLY | O_LARGEFILE)`
- Adds mount-wide marks via `fanotify_mark` with `FAN_MARK_ADD | FAN_MARK_MOUNT` on the filesystem containing `WORKSPACE_DIR`, monitoring events: `FAN_MODIFY | FAN_CREATE | FAN_ATTRIB | FAN_ONDIR`
- Implements path exclusion before processing — skip any event where the path resolves under `/var/log`, `/tmp`, `/var/tmp`, `/proc`, `/sys`, `/dev`, or `~/.cache`. Build the exclusion set at startup, check by prefix match.
- For each event: reads from the fd, extracts the `fanotify_event_metadata` struct via ctypes, derives the file path from `/proc/self/fd/<event.fd>` via `os.readlink`, then closes the event fd immediately.
- If the path is a binary (ELF header `\x7fELF`) or script (shebang `#!`): call `yara_scanner.scan_path(path)` with the active ruleset. On any YARA match, call `log_security_event` with `type="fim_yara_match"`, `disposition="blocked"`.
- All events (match or not) go to `/var/lib/secdaemon/fim_audit.db` — a separate append-only SQLite database outside the monitored workspace. Schema: `rowid, timestamp TEXT, event_type TEXT, pid INTEGER, comm TEXT, path TEXT, yara_matches TEXT, sha256 TEXT`.
- Runs in a daemon thread. On any unhandled exception: log CRITICAL to `security_events.db`, then `raise` to let the thread die. The caller (Policy Gate `__main__`) must detect thread death and trigger systemd restart via `systemctl restart kaia-policy-gate`.
- Exposes `start()`, `stop()`, and `get_recent_alerts(n=10) -> list` for dashboard consumption.

**Wire into `security/policy_gate.py` `__main__`:**  
After `TamperDetector.start()`, instantiate `FIMDaemon` and call `start()`. Wrap in try/except — if fanotify_init fails (permission denied, kernel too old), log a WARNING and fall back to the watchdog sentinel, don't crash.

**Wire into `kaia_dashboard.py` Pane 3 (Containment):**  
The `ContainmentCollector` (from main action plan Block 2) should call `fim_daemon.get_recent_alerts(5)` and display them below the Script Sentinel section.

**Acceptance:** Creating a Python file in `WORKSPACE_DIR` while the daemon runs produces a `fim_yara_match` or generic `fim_event` entry in `/var/lib/secdaemon/fim_audit.db` within 1 second. `watchdog` sentinel still functions as fallback when fanotify fd init raises `PermissionError`.

---

## Block T5-3 — eBPF telemetry pipeline: replace psutil for security-critical observation (§4.2, INV-009)

**Files:** `security/ebpf_telemetry.py` (new), `security/telemetry_daemon.py`, `kaia_dashboard.py`

This replaces `get_network_connections()` and `get_process_lifecycle_events()` in `telemetry_daemon.py` with kernel-hook equivalents. psutil stays for non-security metrics (CPU temp, fan speed in TelemetryCollector) because those aren't security-critical — the invariant at INV-009 applies to security event observation specifically.

**New file: `security/ebpf_telemetry.py`**

Implement `EBPFTelemetryEngine` using the BCC Python bindings (`from bcc import BPF`).

Six hook points per §4.2. Implement each as a BPF C program string and attach:

1. **`sys_enter_execve`** (tracepoint) — capture `pid, uid, comm, filename`. On each event push a dict to an internal `deque(maxlen=500)` protected by a `threading.Lock`.

2. **`tcp_connect`** (kprobe) — capture `pid, comm, daddr, dport`. Convert `daddr` from network-byte-order u32 to dotted-quad string inside the BPF program using `bpf_ntohl`. Push to connection deque.

3. **`tcp_retransmit_skb`** (kprobe) — capture `saddr, daddr, sport, dport, state`. Push to retransmission deque.

4. **`sys_enter_setuid` and `sys_enter_setreuid`** (tracepoints) — capture `pid, comm, uid`. Push to privilege deque.

5. **`sys_enter_openat` / `sys_enter_openat2`** (tracepoints) — capture `pid, comm, filename, flags`. Push to file-access deque. This is the honeypot detection hook (§6.1) — check filename against the honeypot path set inside the BPF program using a BPF hash map of watched paths. On match, submit to a separate perf buffer for immediate userspace alerting.

6. **`sys_enter_unlinkat`** (tracepoint) — capture `pid, comm, filename`. Push to delete deque.

All BPF programs must use perf buffers (`BPF_PERF_OUTPUT`) for submitting events to userspace, not ring buffers (BCC compatibility on older kernels). Poll the perf buffers in a background thread via `b.perf_buffer_poll(timeout=100)`.

All string fields captured from kernel memory must be truncated to safe lengths in the BPF program itself (16 bytes for `comm`, 255 bytes for paths) to avoid kernel memory reads beyond safe bounds.

Expose:
- `get_recent_connections(n=20) -> list[dict]` — most recent tcp_connect events
- `get_recent_execs(n=20) -> list[dict]` — most recent execve events  
- `get_privilege_escalations(n=10) -> list[dict]` — setuid events
- `get_honeypot_hits() -> list[dict]` — openat hits on honeypot paths, cleared on read

**Wire into `telemetry_daemon.py`:**  
Replace `get_network_connections()` and `get_process_lifecycle_events()` with wrappers that call `EBPFTelemetryEngine.get_recent_connections()` and `get_recent_execs()`. Keep the old psutil implementations behind a `try/except ImportError` fallback in case BCC isn't installed — fail open with a WARNING log, not a crash.

**Wire into `kaia_dashboard.py`:**  
The `ContainmentCollector` should pull `get_recent_connections(5)` and `get_privilege_escalations(3)` and surface them in Pane 3. Any `get_honeypot_hits()` result should flash as a CRITICAL event in Pane 1.

**Acceptance:**  
Running `curl http://example.com` on the host produces a `tcp_connect` event in `get_recent_connections()` within 2 seconds. Running any binary produces an `execve` event. `grep psutil security/telemetry_daemon.py` returns only the fallback block.

---

## Block T5-4 — Passive Layer-2/Layer-3 asset discovery (§5)

**Files:** `security/network_discovery.py` (new), `storage/threat_intel/assets.db` (created at runtime)

**New file: `security/network_discovery.py`**

Implement `PassiveDiscoveryEngine`:

- Opens a raw `AF_PACKET` socket with `ETH_P_ALL` (requires `CAP_NET_RAW`, available on the Policy Gate daemon per Appendix A). Binds to a configurable interface name (add `DISCOVERY_INTERFACE = "auto"` to `config.py` — if "auto", detect the default route interface via `/proc/net/route`).
- Reads raw Ethernet frames in a loop. Parse the 14-byte Ethernet header: `dst_mac (6) | src_mac (6) | ethertype (2)`.
- For EtherType `0x0806` (ARP): parse the ARP payload — `htype, ptype, hlen, plen, oper, sha, spa, tha, tpa`. Extract sender MAC (`sha`) and sender IP (`spa`). Log asset with vector `ARP`.
- For EtherType `0x0800` (IPv4): parse IP header to get protocol and src/dst. If protocol is UDP (17), inspect destination port:
  - `5353` → mDNS: parse DNS payload (first question record) to extract queried name. Log with vector `mDNS`.
  - `5355` → LLMNR: log with vector `LLMNR`, extract queried name from DNS-format payload.
  - `1900` → SSDP: log with vector `SSDP`, extract `ST:` or `Location:` header from HTTP-like body.
  - `137` → NetBIOS: parse NetBIOS Name Service packet to extract the encoded hostname (15-char L2-encoded name, strip trailing spaces and null bytes). Log with vector `NetBIOS`.
- All MAC-to-vendor lookups: ship a minimal OUI prefix file (`storage/threat_intel/oui.txt`, first 3 bytes of MAC → vendor name). Load into a dict at startup. If the file doesn't exist, vendor field is `"unknown"` — don't fetch it at runtime (out of scope per §12).
- Deduplication: maintain an in-memory dict keyed by `(mac, ip)`. Only write a new row to `assets.db` if the pair hasn't been seen in the last 5 minutes. This prevents flooding the DB from chatty mDNS devices.
- **Asset DB schema** (`storage/threat_intel/assets.db`, table `assets`):
  ```
  ip TEXT, mac TEXT, hostname TEXT, vendor TEXT,
  detection_vector TEXT, first_seen TEXT, last_seen TEXT,
  PRIMARY KEY (ip, mac)
  ```
  On conflict (same ip+mac seen again): update `last_seen` only.
- Runs in a daemon thread. Expose `get_recent_assets(n=20) -> list[dict]`.
- Feeds into `threat_intel.py` correlation: after logging a new asset, call `lookup_internetdb(ip)` in a separate thread (non-blocking) and update the asset row with any Shodan-sourced tags.

**Wire into `kaia_dashboard.py` Pane 2 (Threat Intel):**  
Add a "LAN Assets" section below the firewall blocks, showing the last 5 newly discovered assets with their MAC, vendor, and detection vector.

**Wire into `config.py`:** Add `DISCOVERY_INTERFACE = "auto"` and `ASSETS_DB_PATH = str(STORAGE_DIR / "threat_intel" / "assets.db")`.

**Acceptance:** On a LAN with other devices, running `python -c "from security.network_discovery import PassiveDiscoveryEngine; e = PassiveDiscoveryEngine(); e.start(); import time; time.sleep(30); print(e.get_recent_assets())"` returns at least one asset within 30 seconds (from ARP or mDNS traffic). `assets.db` is created and populated.

---

## Block T5-5 — eBPF honeypot canary and network decoy system (§6.1, §6.2)

**Files:** `security/honeypot.py` (new), `scripts/setup_decoys.sh` (new)

**New file: `security/honeypot.py`**

Implement `HoneypotCoordinator`:

**Filesystem honey-tokens (§6.1):**
- On `start()`, create the following decoy files if they don't exist. Write plausible but obviously fake content:
  - `/etc/api_keys.json` — `{"aws_key": "AKIAIOSFODNN7EXAMPLE", "note": "decoy"}`
  - `/var/backups/credentials.txt` — `db_password=example_decoy_do_not_use`
  - `~/.ssh/authorized_keys.bak` — empty file with a comment header
- Store SHA-256 of each decoy on creation. On `stop()`, delete them.
- The eBPF hook in `EBPFTelemetryEngine` (Block T5-3) already monitors `sys_enter_openat`. Wire up the honeypot path set: call `ebpf_engine.register_honeypot_paths(["/etc/api_keys.json", "/var/backups/credentials.txt", "~/.ssh/authorized_keys.bak"])` which populates the BPF hash map.
- When `ebpf_engine.get_honeypot_hits()` returns events: log each to `security_events.db` with `type="honeypot_file_access"`, `disposition="blocked"`, severity CRITICAL. Attempt to derive the source IP from the `pid` via `/proc/<pid>/net/tcp` — if found, immediately enqueue a `block_ip` request through `utils.send_to_policy_gate`.

**Network decoy listeners (§6.2):**
- Create a network namespace `ns_decoy` using `ip netns add ns_decoy` (subprocess, no shell=True, absolute path `/usr/bin/ip`). Create a veth pair: `veth_host` in the host namespace, `veth_decoy` in `ns_decoy`. Assign IPs: `10.254.0.1/30` on `veth_host`, `10.254.0.2/30` on `veth_decoy`.
- In `ns_decoy`, start lightweight TCP listeners on ports `22, 443, 3306, 5432, 8080` using `socket.socket` in Python threads. Each listener accepts a connection, logs the source IP and port, then closes immediately.
- Configure nftables to redirect traffic destined for the host on those ports to the decoy namespace: use `ip rule` and `ip route` to steer the traffic (not nftables redirect — nftables redirect requires NAT which changes the source IP we need to log).
- When any decoy listener gets a connection: log to `security_events.db` with `type="honeypot_port_trigger"`, `disposition="blocked"`. Immediately call `utils.send_to_policy_gate` with a `block_ip` payload for the source IP. This is the automated pre-approved policy exception per §1.1 axiom 5 (immutable policy: any decoy port hit = auto-block).
- Expose `get_decoy_status() -> dict` returning `{port: listener_active, last_trigger_ip, last_trigger_time, total_triggers}` for each port.

**New file: `scripts/setup_decoys.sh`:**  
Documents the one-time setup: creating the `ns_decoy` namespace, veth pair, and nftables rules that survive reboots. This is operator-run once; `honeypot.py` assumes the namespace already exists and reconnects to it.

**Wire into `kaia_dashboard.py` Pane 4 (System Security):**  
The `SystemSecurityCollector` (from main plan Block 2) should call `honeypot_coordinator.get_decoy_status()` to populate the "Honeypot Status" section showing last trigger time and active decoy ports.

**Wire into `security/policy_gate.py` `__main__`:**  
Instantiate `HoneypotCoordinator` and call `start()` after FIM and eBPF startup.

**Acceptance:** `nc -zv localhost 3306` produces a `honeypot_port_trigger` entry in `security_events.db` within 2 seconds and a corresponding `block_ip` approved/denied entry in `audit_ledger.json`.

---

## Block T5-6 — YARA rule pipeline (§7)

**Files:** `security/rule_engine.py` (new), `security/schemas.py`, `storage/threat_intel/rules/` (created at runtime)

**New file: `security/rule_engine.py`**

Implement `RuleEngine`:

**Rule ingestion schema (§7.1):**  
Add `IocRuleRequest` Pydantic model to `security/schemas.py`:
```
rule_name: str — regex `^[a-zA-Z0-9_]+$`, 3–64 chars
author: str — default "Kaia Automated Rule Engine"
threat_description: str — 10–256 chars
target_ioc_indicator: str — 4–128 chars
mitre_framework_id: Optional[str] — pattern `^T[0-9]{4}$` or None
```

**Rule compilation (§7.2):**  
`RuleEngine.compile_rule(request: IocRuleRequest) -> str`:  
- Escape the `target_ioc_indicator` by running it through `re.escape()` and also stripping any YARA metacharacters (`{ } [ ] / \`).  
- Build a YARA rule string with meta fields: `author`, `description`, `mitre_id`. Condition: `any of them` matching the indicator as a string or hex pattern.  
- Return the rule string. Do not write to disk yet — validation comes first.

**Ephemeral validation (§7.3):**  
`RuleEngine.validate_rule(rule_text: str) -> tuple[bool, str]`:  
- Write the rule to a tempfile in `/tmp` (use `tempfile.NamedTemporaryFile`, `delete=False`).
- Build a benign corpus dir in `/tmp` with 5 known-clean text files (lorem ipsum content — write them at startup and keep the path).  
- Run validation via `systemd-run` with the strict isolation flags from §7.3: `PrivateTmp=yes`, `ProtectSystem=strict`, `PrivateNetwork=yes`, `DynamicUser=yes`, `NoNewPrivileges=yes`, `CapabilityBoundingSet=` (empty), `SystemCallFilter=@system-service`, `RestrictAddressFamilies=none`. Bind-mount the rule file and benign corpus as read-only. Execute `yara <rulefile> <corpus_dir>`.
- If exit code is non-zero: validation failed, return `(False, stderr)`.
- If exit code is 0 but stdout is non-empty: the rule matched a benign file, return `(False, "false positive on benign corpus")`.
- If exit code is 0 and stdout is empty: return `(True, "")`.
- Always delete the tempfile in `finally`.

**Rule storage:**  
On successful validation: write the rule to `storage/threat_intel/rules/<rule_name>.yar`. Load all rules from this directory into a `yara.compile()` ruleset on startup and after any new rule is added. Expose `scanner: yara.Rules` for use by FIM daemon (Block T5-2).

**Wire into `security/schemas.py`:** Add `IocRuleRequest` model (and add `from typing import Any` at top properly while there).

**Wire into `kaia_dashboard.py` command panel (Block 6 from main plan):**  
Add command: `> add rule <name> <indicator> [mitre:T1234]` — parse into `IocRuleRequest`, call `rule_engine.compile_rule()` then `validate_rule()`, display result, on success write to disk and reload.

**Wire into FIM daemon (Block T5-2):**  
On startup, `FIMDaemon` receives a `RuleEngine` instance and calls `yara_scanner = rule_engine.scanner` to get the compiled ruleset. When a new rule is added, call `fim_daemon.reload_rules(rule_engine.scanner)`.

**Acceptance:** `> add rule detect_test EICAR_TEST_STRING` compiles, passes validation on benign corpus (EICAR string isn't in lorem ipsum), writes `storage/threat_intel/rules/detect_test.yar`, and the FIM scanner picks it up on the next file event. `> add rule bad_rule ".*"` fails validation with "false positive" message.

---

## Block T5-7 — GeoLite2 integration (§Appendix C, §11)

**Files:** `security/threat_intel.py`, `scripts/update_geoip.sh` (new)

GeoLite2 requires a free MaxMind account for the download URL — the MMDB file cannot be bundled. Create an update script and wire the existing `lookup_geoip()` function properly.

**`threat_intel.py` changes:**  
- `lookup_geoip()` already imports `maxminddb` and reads from `THREAT_INTEL_DIR / "geoip" / "GeoLite2-City.mmdb"`. Move `GEOIP_DB_PATH` into `config.py` as a named constant.
- If the MMDB file doesn't exist, `lookup_geoip` must return `{"country": "unknown", "city": "unknown", "latitude": None, "longitude": None}` — it already does via the fallback, just make it explicit and log a DEBUG (not WARNING) so the dashboard doesn't spam on fresh installs.
- After any `block_ip` event in the `AuditLogCollector`, call `lookup_geoip(ip)` and annotate the event for display in Pane 2. This is the enrichment that surfaces country codes in the §9.2 mockup.

**New file: `scripts/update_geoip.sh`:**  
Template script that downloads `GeoLite2-City.mmdb.tar.gz` from MaxMind using a `MAXMIND_LICENSE_KEY` environment variable, extracts the MMDB, and places it at the correct path. Includes a cron/systemd-timer setup comment. Do not hardcode any URL that embeds a key — the operator must supply their key.

**Acceptance:** With a valid MMDB at the correct path, `python -c "from security.threat_intel import lookup_geoip; print(lookup_geoip('8.8.8.8'))"` returns `{'country': 'United States', ...}`. Without the MMDB, it returns the safe fallback without errors.

---

## Block T5-8 — Emergency lockdown systemd unit (Appendix B)

**Files:** `scripts/kaia-lockdown.service` (new), `scripts/kaia-lockdown.sh` (new), `security/policy_gate.py`

**New file: `scripts/kaia-lockdown.sh`:**  
Shell script (no Python) that:
1. Calls `nft flush ruleset` to clear all existing nftables rules.
2. Creates a new `inet filter` table with drop-policy chains:
   ```
   nft add table inet filter
   nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
   nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
   nft add chain inet filter output { type filter hook output priority 0 \; policy drop \; }
   ```
3. Logs the lockdown event with `logger -t kaia-lockdown "EMERGENCY LOCKDOWN ACTIVATED"`.
4. Optionally: `systemctl isolate rescue.target` — leave this commented out by default so the operator must explicitly uncomment it. Network isolation is enough for most cases; pulling to rescue could lock out remote admins.

**New file: `scripts/kaia-lockdown.service`:**  
```
[Unit]
Description=Kaia Emergency Network Lockdown
DefaultDependencies=no
Before=network.target shutdown.target

[Service]
Type=oneshot
ExecStart=/bin/bash /path/to/scripts/kaia-lockdown.sh
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
```

**Wire into `security/policy_gate.py`:**  
Add a `trigger_lockdown(reason: str)` function that:
- Logs `type="emergency_lockdown_triggered"` to `security_events.db` with the reason string.
- Calls `subprocess.run(["/usr/bin/systemctl", "start", "kaia-lockdown.service"], check=False)` — no shell, absolute path, `check=False` so a failure to start the service doesn't crash the Gate.
- If the service isn't installed, falls back to executing the shell script directly.

Call `trigger_lockdown()` from:
- The tamper detection handler (Block 9 of main plan) when a core security file is modified.
- The FIM daemon (Block T5-2) on a YARA match in `security/` or `core/config.py`.
- Any Policy Gate exception handler that can't recover.

**Acceptance:** `sudo systemctl start kaia-lockdown.service` results in `nft list ruleset` showing drop policies on all three chains. `systemctl stop kaia-lockdown.service` and restoring rules is operator-manual (documented in script comments).

---

## Block T5-9 — Wire all Tier 5 components into the dashboard (§9.3, §9.4)

**Files:** `kaia_dashboard.py`

After all prior T5 blocks are complete, the dashboard needs to surface Tier 5 data properly. This is a dedicated wiring pass — don't do it piecemeal during each block.

**Pane 2 additions:**
- LAN Assets section from `PassiveDiscoveryEngine.get_recent_assets(5)` — show IP, vendor, vector, last seen.
- Geo annotation on blocked IPs from `lookup_geoip()` — show country code in parentheses.
- Honeypot trigger last timestamp and count from `HoneypotCoordinator.get_decoy_status()`.

**Pane 3 additions:**
- FIM alert count from `FIMDaemon.get_recent_alerts(5)` — path, event type, YARA match (if any).
- eBPF privilege escalation alerts from `EBPFTelemetryEngine.get_privilege_escalations(3)`.

**Pane 4 additions:**
- YARA ruleset count: `len(rule_engine.scanner.rules)` if accessible, else file count in `storage/threat_intel/rules/`.
- Lockdown status: check if `kaia-lockdown.service` is active via `systemctl is-active kaia-lockdown.service` (subprocess, no shell).

**Command panel new commands:**
- `> lockdown` — prompt for confirmation (`Are you sure? [y/N]`), on y call `trigger_lockdown("operator_command")`. Display result.
- `> add rule <name> <indicator> [mitre:T1234]` — per Block T5-6.
- `> show assets` — display contents of `get_recent_assets(20)` in the response area.
- `> show fim alerts` — display last 10 FIM events.

**Acceptance:** All four panes show data from Tier 5 sources. `> lockdown` requires confirmation and on confirmation calls the lockdown function. `> show assets` returns LAN asset rows.

---

## Dependency order

```
T5-1 (deps)
  └─ T5-2 (fanotify FIM) ──── requires: T5-1
  └─ T5-3 (eBPF engine)  ──── requires: T5-1
       └─ T5-5 (honeypots) ── requires: T5-3 (for openat hook)
            └─ T5-8 (lockdown) — requires: T5-5 to wire trigger
  └─ T5-4 (L2/L3 discovery) — requires: T5-1 (CAP_NET_RAW)
  └─ T5-6 (YARA rules) ──────── requires: T5-1 (yara-python), feeds T5-2
  └─ T5-7 (GeoLite2) ─────────── requires: T5-1 (maxminddb), standalone
  └─ T5-9 (dashboard wiring) ─── requires: all prior blocks complete
```

T5-4, T5-6, and T5-7 have no dependencies on each other and can run in parallel.

---

## What Tier 5 does NOT include (§12 scope boundaries)

- Internet-wide IPv6 scanning — out of scope.
- Full Shodan InternetDB mirror — out of scope, lightweight per-IP cache only.
- TLS interception — out of scope.
- DuckDB delta computation from Appendix C — deferred until InternetDB data volume justifies it.
- Supply chain monitoring, baseline drift, secrets scanning from §11 — noted as optional future work, not in this plan.
