# **Kaia Project Historical Archive (v1.0)**

**Document Classification:** Historical Compendium / Consolidated Archive  
**Consolidation Date:** June 25, 2026  
**Target Environment Context:** Arch Linux, Systemd, eBPF Kernel Boundaries, Unix IPC Framework

## **1\. Executive Summary & Consolidated Assets Registry**

This document serves as the single, authoritative historical repository for the Kaia development lifecycle. It incorporates, stabilizes, and archives all early design reports, codebase audits, gaps analysis notes, research drafts, and tactical checklists. By consolidating these materials into an immutable archive, the project clean-rooms its directory topology while retaining absolute historical fidelity for architectural regression tracing.  
The following ten files from the development pipeline have been fully parsed, reconciled, and compiled into this master archive:

* Hardened AI Admin Agents first draft design.md  
* Kaia\_Unified\_Architectural\_Specification.md (v6.0)  
* Kaia: Unified Project Documentation & Implementation Blueprint.md  
* kaia\_status\_review\_and\_masterplan.md  
* Kaia\_Design\_Masterplan\_v7.md (v7.0)  
* Gemini\_Research.md  
* Gemini\_Claude\_ChatGPT\_Review.md  
* kaia\_codebase\_review\_v2.md  
* Phase 3 Implementation Tasks.md  
* Cleanup\_Threat Intel\_Dashboard Stage 1.md

## **2\. Foundational Isolation Paradigms & Mathematical Security Latices**

The project’s initial sandboxing architecture established a zero-trust containment system to handle LLM payloads. Because language models merge instructions and data within a single shared context window, all output must be verified deterministically. The initial specification defined a mathematically derived Restrictiveness Lattice to resolve effective isolation limits dynamically.  
Let *L\_G* represent the global restrictiveness configuration set by the host operator, and *L\_A* represent the local security allocation designated within a specific code workspace directory. The effective system restrictiveness level *L\_E* is resolved via the following partial order lattice formulation:

`L_E = max(L_G, L_A)`

This total ordering is calculated over the set of discrete system security isolation states:

`S = {none < namespace ≤ sandbox-exec < bwrap < gvisor < firecracker < auto}`

### **Historical Containment Tier Definitions**

* **none:** Unrestricted user-space host context execution (permanently deprecated).  
* **namespace:** Unprivileged Linux user namespaces utilizing explicit CLONE\_NEWPID, CLONE\_NEWNET, and CLONE\_NEWNS flags.  
* **sandbox-exec:** Platform-specific declarative application profiles (e.g., macOS sandbox-exec variants evaluated during multi-platform staging).  
* **bwrap:** Bubblewrap-enforced filesystem namespace containment, executing with explicit read-only mount tables, clean tmpfs spaces, and severe directory masking.  
* **gvisor:** Multi-tenant user-space kernel containment utilizing Sentry/Gofer application interceptors.  
* **firecracker:** Isolated microVM instances executing over Kernel-based Virtual Machine (KVM) abstractions.

## **3\. Systems-Level Engineering Research Logs**

### **3.1 File Integrity Monitoring via fanotify**

Early research established that traditional inotify-based recursive tracking introduces massive file descriptor exhaustion patterns over deep directory trees. The system was designed around direct ctypes bindings to the native Linux fanotify API, setting mount-wide tracking masks. This architecture effectively mitigates Time-of-Check to Time-of-Use (TOCTOU) file-swapping race vectors by performing real-time in-memory YARA signature verification on the kernel-provided descriptor at /proc/self/fd/{fd} before letting the target process proceed.

| Excluded Mount Path | Event Mask Omitted | Technical Rationale |
| :---- | :---- | :---- |
| /var/log/\* | FAN\_MODIFY | Prevents catastrophic diagnostic logging echo-loops within the filesystem pipeline. |
| /tmp/\*, /var/tmp/\* | FAN\_CREATE | Avoids severe transient compiler thrashing during automated software updates. |
| /proc/\*, /sys/\*, /dev/\* | ALL\_EVENTS | Complete exclusion of virtual and pseudo-device paths managed by the kernel. |
| /home/\*/.cache/\* | FAN\_ACCESS | Omit browser cache synchronization loops to preserve I/O capacity. |

### **3.2 Shell-less Layer-2/Layer-3 Active Network Topology Mapping**

To adhere to the core constraint banning shell execution or wrapper scripts (e.g., calling out to nmap or arp-scan), the threat intelligence discovery pipeline maps assets via low-level AF\_PACKET native python socket loops. This framework captures binary protocol structures directly from the network interface card, decoding broadcast frames cleanly into structured Pydantic profiles.

* **ARP Frames (0x0806):** Extracts sender hardware address (MAC) and network layer targets to discover active local interfaces.  
* **mDNS Payloads (UDP Port 5353):** Parses multicast DNS answer structures to extract internal network hostnames and device tags.  
* **LLMNR / SSDP Queries:** Captures local link multicast requests to discover unmanaged enterprise endpoints.

## **4\. Historical Codebase Audits & Critical Security Gap Analysis**

During the Phase 1 and Phase 2 refactoring periods, multiple structural vulnerabilities were cataloged across consecutive code assessments. These historical records document the security debt that was subsequently remediated in the v7.4 constitution.

| Vulnerability Target | Risk Profile & Mechanics | Severity | Historical Remediation Path |
| :---- | :---- | :---- | :---- |
| BASE\_DIR Miscalculation | Moving config.py into core/ without updating path depth broke repository root resolution, exposing storage paths to Git leakage. | CRITICAL | Path resolution rewritten to resolve via grandparent parent attributes. |
| Insecure State Modification | write\_file execution pathways originally verified workspace root prefixes but allowed arbitrary write operations over core application scripts, risking RCE. | CRITICAL | Strict extension filters and structural folder path blocks implemented. |
| In-Process Policy Gate Threading | The deterministic policy gate originally ran as a standard Python thread inside the main application context, violating memory boundary isolation. | HIGH | Refactored into an independent out-of-process Unix Domain Socket daemon. |
| Hardcoded Secrets Exposure | CAPABILITY\_TOKEN\_SECRET was initialized as a plaintext string asset inside source configuration code. | HIGH | Converted to mandatory runtime environment load with explicit fail-closed parameters. |
| Telemetry Polling Fallback | The early telemetry\_daemon.py utilized standard psutil polling intervals instead of real-time eBPF streaming. | MEDIUM | Deferred to Phase 5 advanced kernel observer tracking matrices. |

## **5\. Phase 3 Tactical Checklists & Implementation Hygiene Records**

The primary execution sprint for Phase 3 focused on removing orphaned services and transitioning external threat data sourcing away from insecure bulk databases to active on-demand lookup patterns. Below is the verified status map preserved from the Phase 3 sprint documentation:

### **5.1 Structural Hygiene Tasks**

* **\[DONE\]** Purge convert\_video\_to\_gif from config.py system prompts and few-shot classification arrays.  
* **\[DONE\]** Delete the dead convert action plan handling fallback statements from main.py.  
* **\[DONE\]** Remove standalone chroma server strings from the service restart enums inside kaia\_cli.py, policy\_gate.py, and host\_executor.py.  
* **\[DONE\]** Pin the filesystem filesystem observation daemon engine dependency (watchdog==6.0.0) inside the root requirements.txt file.

### **5.2 Cache-First Shodan InternetDB Architecture**

Initial specifications erroneously assumed free local storage mirrors for massive 15-35GB bulk Shodan files. The design was updated to a lazy-loading, cache-first paradigm utilizing the live per-IP InternetDB lookup interface:

1. Query the local SQLite internetdb.db infrastructure cache table first.  
2. On a local cache miss, issue a safe outbound call to the free API at https://internetdb.shodan.io/{ip}.  
3. Apply an inline one-second rate-limiting constraint (time.sleep(1)) to comply with network use parameters.  
4. Store the JSON data record alongside a last\_updated text timestamp.  
5. Enforce a strict 7-day Time-to-Live (TTL) constraint; stale data triggers lazy API refreshing on-demand.

### **5.3 Curses Telemetry Redesign — Read-Only Pane Allocation**

The early TUI companion monitoring script (kaiamon.py) was adapted out of the old documentation workspace into the primary operational interface asset. This design established the four distinct telemetry panes:

* **Pane 1 (Audit Log Feed):** Polled database connection looping reading from security\_events.db and audit\_ledger.json every 250ms.  
* **Pane 2 (Threat Intel Matrix):** Extracted local network interface configurations and active firewall tables from nftables.  
* **Pane 3 (Containment Boundary Status):** Evaluated the effective restrictiveness lattice calculation index, current cgroup resource limits, and Script Sentinel states.  
* **Pane 4 (Platform Security State):** Aggregated capability token metrics, firewall chain drops, and socket connection statistics.

This concludes the full unabridged archival mapping of Kaia's project history. All development history, operational checkpoints, and technical parameters are securely locked in this compendium asset.
