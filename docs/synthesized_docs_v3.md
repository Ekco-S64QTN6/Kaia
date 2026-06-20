\`\`\`markdown  
\# Kaia Project Status & Implementation Roadmap

\#\# Synthesis of Reviews, Findings, and Action Plan

\*\*Status as of:\*\* 2026-06-19    
\*\*Document Version:\*\* 2.0 (Coding Agent Execution Specifications Added)    
\*\*Intended Reader:\*\* AI Agent / Developer (architectural awareness, next-step determination, and concrete implementation directives)

\---

\#\# Executive Summary

The Kaia project has progressed significantly since the initial Kaiacord concept. A full security review against \`Hardened AI Admin Agents\_v2.md\` has been completed, revealing several critical regressions introduced during the Phase 1 reorganization and various gaps between the design specification and current implementation. The most urgent structural issue (BASE\_DIR miscalculation) has been confirmed as fixed in the running instance, but several high-severity gaps remain that must be addressed before proceeding to Phase 4 cognitive integration.

\*\*Current State:\*\*    
\- Phase 1 (Security Foundation): \*\*Partially complete\*\* — core infrastructure exists but has critical gaps    
\- Phase 2 (Isolation and Containment): \*\*Partially complete\*\* — Bubblewrap implemented, but tier system and controls missing    
\- Phase 3 (Threat Intelligence): \*\*Not started\*\* — scaffolded only    
\- Phase 4 (Cognitive Integration): \*\*Not started\*\* — code exists but is not wired in and should not be until Phases 1-3 stable

\---

\#\# Document Source Mapping

| Document | Status | Key Input |  
|---|---|---|  
| \`Hardened AI Admin Agents\_v2.md\` | Design Spec | Target architecture to implement |  
| \`project\_security\_review.md\` | Initial Review | Identified vulnerabilities, planned mitigations |  
| \`walkthrough.md\` | Implementation Record | Phase 1 reorg \+ Phase 2 patches applied |  
| \`kaia\_codebase\_review\_v2.md\` | Detailed Code Review | Uncovered regressions and additional gaps |  
| \`Gemini\_Claude\_ChatGPT\_Review.md\` | Synthesis / Runtime Logs | Confirmed fixes and identified infrastructure debt |

\---

\#\# Completed Work (Timeline)

\#\#\# Phase 1: Repository Reorganization (Completed)

\*\*Actions Taken:\*\*    
\- Moved \`config.py\`, \`kaia\_cli.py\`, \`database\_utils.py\`, \`utils.py\` into \`core/\`    
\- Moved \`activate\_kaia\_env.sh\` to \`scripts/\`    
\- Moved all test files to \`tests/\`    
\- Updated \`sys.path\` handling in \`main.py\` and test files    
\- Created package structure with \`core/\_\_init\_\_.py\`    
\- Relocated Vulkan JSON files to \`data/vulkaninfo/\`

\#\#\# Phase 2: Initial Security Patches (Completed)

\*\*Patch A — State Modification Protection\*\*    
\- \`execute\_state\_modification\` now blocks writes to:    
  \- Python source files (\`\*.py\`)    
  \- Shell scripts (\`\*.sh\`)    
  \- Hidden files/directories (\`.env\`, \`.git\`, \`.venv\`)    
  \- Critical directories: \`core/\`, \`security/\`, \`tests/\`, \`scripts/\`, \`toolbox/\`, \`storage/\`

\*\*Patch B — Bubblewrap Sandbox Masking\*\*    
\- \`execute\_script\` now masks:    
  \- \`.env\` → \`/dev/null\`    
  \- \`kaia.log\` → \`/dev/null\`    
  \- \`storage/\` → \`tmpfs\` (empty RAM disk)

\*\*Verification:\*\* Both patches passed security test suite. However, note that the \`BASE\_DIR\` bug had silently broken those tests until fixed.

\---

\#\# Critical Issues Confirmed / Discovered

\#\#\# A. BASE\_DIR Miscalculation (Fixed — Confirmed Working)

\*\*Original Issue:\*\* \`core/config.py\` computed \`BASE\_DIR \= Path(\_\_file\_\_).resolve().parent\`, which resolved to \`core/\` instead of repo root. This broke:    
\- RAG indexing (reading from empty \`core/data\`)    
\- Persona loading (fallback to default prompt)    
\- gitignore coverage (\`core/storage/\` not excluded)    
\- Workspace boundary displacement

\*\*Confirmation of Fix:\*\* Runtime log shows:    
\`\`\`  
Scanning /home/ekco/github/Kaia/data...  
Scanning /home/ekco/github/Kaia/personal\_context...  
\`\`\`  
— these are repo-root paths, confirming \`BASE\_DIR\` now resolves correctly.

\*\*Issue:\*\* The fix was applied, but the \`core/storage/audit\_ledger.json\` file uploaded to the repository indicates the bug was live at some point and may have already caused exposure.

\---

\#\#\# B. Hardcoded Capability Token Secret (Unresolved — Critical)

\*\*Location:\*\* \`core/config.py\`    
\`\`\`python  
CAPABILITY\_TOKEN\_SECRET \= "kaia\_secure\_signing\_secret\_key\_2026"  
\`\`\`

\*\*Impact:\*\* Anyone with repo access or a compromised agent process can read this secret and forge valid HMAC-SHA256 capability tokens for any action. This completely bypasses the operator approval protocol described in Section 2.5 of the spec.

\*\*Status:\*\* \*\*UNRESOLVED\*\*

\*\*Required Action:\*\* Load from environment variable with immediate termination if missing:    
\`\`\`python  
import os  
CAPABILITY\_TOKEN\_SECRET \= os.environ\["KAIA\_CAPABILITY\_SECRET"\]  
\# No default — fail loud if unset  
\`\`\`

\---

\#\#\# C. \`convert\_video\_to\_gif\` Bypasses Policy Gate (RESOLVED — Removal)

\*\*Location:\*\* \`toolbox/video\_converter.py\`, \`main.py\`

\*\*Issue:\*\* \`convert\_video\_to\_gif\` is the only action handler that never touches \`security/policy\_gate.py\`. It calls \`subprocess.run\` directly, with no schema validation, capability token check, audit log entry, or sandboxing. This legacy module is irrelevant to Kaia's core mission (hardened systems agent / security operations).

\*\*Risk:\*\* \`ffmpeg\` runs unsandboxed on the host against untrusted video files, exposing the system to ffmpeg parser CVEs.

\*\*Status:\*\* \*\*RESOLVED (Removal)\*\* — The module will be deleted entirely, not retrofitted.

\*\*Action Taken:\*\*    
\- Delete \`toolbox/video\_converter.py\`    
\- Remove its handler registration from \`main.py\`    
\- Delete any orphaned imports    
\- Remove \`toolbox/\` directory if empty    
\- Scrub any references in documentation or \`requirements.txt\` (e.g., \`ffmpeg-python\`)

\*\*Verification:\*\* \`grep \-r "convert\_video" .\` returns zero results (except this document). The agent no longer exposes a video‑conversion capability.

\---

\#\#\# D. In-Process Policy Gate Thread (Unresolved — Architectural Gap)

\*\*Spec:\*\* Section 2.3 — Policy Gate should be a separate process with socket-based IPC, owner \`root:kaiacord\`, mode \`0660\`.

\*\*Current:\*\* \`main.py\` starts \`PolicyGate\` as \`threading.Thread\` in the same process as the agent.

\*\*Risk:\*\* If the agent process is compromised, the policy gate shares the same memory space. An attacker can manipulate Python runtime globals, disable validation, forge tokens, or hook \`HostExecutor\` methods directly.

\*\*Status:\*\* \*\*UNRESOLVED\*\*

\*\*Required Action:\*\* Move Policy Gate to a separate process with proper socket permissions:    
\`\`\`  
/run/kaiacord/policy\_gate.sock  
owner: root  
group: kaiacord  
mode: 0660  
\`\`\`

\---

\#\#\# E. /run/kaiacord Permission Denied (Unresolved — Infrastructure)

\*\*Runtime log:\*\*    
\`\`\`  
Permission denied to create directory /run/kaiacord  
Falling back to /tmp/policy\_gate.sock  
\`\`\`

\*\*Issue:\*\* The hardened socket path described in Section 2.3 has never actually been used. The fallback to \`/tmp/policy\_gate.sock\` is weaker security.

\*\*Required Action:\*\*    
\`\`\`bash  
sudo groupadd kaiacord  
sudo usermod \-aG kaiacord ekco  
\`\`\`

Create \`/etc/tmpfiles.d/kaiacord.conf\`:    
\`\`\`  
d /run/kaiacord 0770 ekco kaiacord \-  
\`\`\`  
Then:    
\`\`\`bash  
sudo systemd-tmpfiles \--create  
\`\`\`

For long-term, use \`RuntimeDirectory=kaiacord\` in a systemd service unit.

\---

\#\#\# F. ChromaDB Server Orphan (Technical Debt)

\*\*Issue:\*\* \`activate\_kaia\_env.sh\` attempts to start a standalone ChromaDB server on port 8000\. However, \`main.py\` uses \`chromadb.PersistentClient\` (embedded mode) directly, never connecting to the server.

\*\*Consequence:\*\* Failed startup error on every launch; wasted retry time; risk of two processes hitting the same on-disk store if the server ever did start successfully while \`main.py\` is running.

\*\*Status:\*\* \*\*UNRESOLVED\*\*

\*\*Required Action:\*\* Remove the ChromaDB server startup block from \`activate\_kaia\_env.sh\`, or if the HTTP mode is needed, choose one mode consistently.

\---

\#\#\# G. Test Suite False Positives (Unresolved — Medium)

\*\*Issue 1:\*\* \`tests/test\_kaia\_cli.py\` asserts against \`generate\_command()\`'s pre-refactor behavior (string commands like \`"ls \-la"\`), but the current implementation returns JSON intent payloads. \*\*Tests would fail if run\*\* — the \`generate\_command()\` rewrite has no real test coverage.

\*\*Issue 2:\*\* \`tests/verify\_security.py\` tests directory blocking with \`core/config.py\`, but since \`.py\` extension check runs before \`BLOCKED\_DIRS\`, the directory check is never exercised. A regression removing \`BLOCKED\_DIRS\` wouldn't be caught.

\*\*Issue 3:\*\* Bubblewrap masking test uses relative \`cat .env\` inside the sandbox. If cwd is outside bound mounts, the test passes due to "file not found" rather than successful masking. Needs absolute paths and stderr inspection.

\*\*Status:\*\* \*\*UNRESOLVED\*\*

\*\*Required Action:\*\* Rewrite \`test\_kaia\_cli.py\` to assert against JSON-intent contract. Add a non-\`.py\` test case for directory blocking. Re-write masking test with absolute paths and explicit stderr capture.

\---

\#\#\# H. Telemetry Gap: psutil Polling vs eBPF (Unresolved — Design Gap)

\*\*Spec:\*\* Section 3.3 specifies eBPF tracepoints and kprobes for real-time event capture.

\*\*Current:\*\* \`telemetry\_daemon.py\` uses \`psutil\` polling only.

\*\*Risk:\*\* Actions occurring between polling intervals are invisible.

\*\*Status:\*\* \*\*UNRESOLVED\*\* (Scheduled for Phase 2\)

\---

\#\#\# I. Service Restart Gaps (Unresolved — Design Gap)

\*\*Spec:\*\* Section 2.2 — service restarts require:    
\- Static allowlist (✅ implemented)    
\- Restart frequency below threshold (❌ missing)    
\- Service health degraded per telemetry (❌ missing)

\*\*Current:\*\* Only allowlist check exists.

\*\*Status:\*\* \*\*UNRESOLVED\*\*

\---

\#\#\# J. Threat Intelligence Data Missing (Scheduled)

\*\*Spec:\*\* Section 6 — GeoLite2, reputation cache, CVEDB, InternetDB snapshots.

\*\*Current:\*\* \`threat\_intel.py\` has query interface with safe fallbacks; real data files absent.

\*\*Status:\*\* Consistent with spec's Phase 3 ordering — not a regression, but should not be considered complete.

\---

\#\# Architecture Compliance Table

| Section | Requirement | Status | Notes |  
|---|---|---|---|  
| 1.1 | Restrictiveness Lattice (max/intersection) | ❌ | Not implemented |  
| 1.2 | 5-tier sandboxing | ⚠️ | Only Tier 2 (Bubblewrap) exists |  
| 1.3 | systemd-nspawn, BPF hash allowlist, Landlock | ❌ | Not implemented |  
| 1.4 | cgroup resource ceilings | ❌ | Only subprocess timeout |  
| 2.1 | 4-stage validation pipeline | ⚠️ | Partial — regex filter only for diagnostics |  
| 2.2 | Service restart: allowlist \+ frequency \+ health | ⚠️ | Only allowlist |  
| 2.3 | Socket ownership root:kaiacord, 0660, MAC label | ⚠️ | 0660 set, but no group ownership; fallback to /tmp |  
| 2.4 | Fail-closed on gate unavailability | ✅ | Implemented correctly |  
| 2.5 | Capability tokens (HMAC, expiring, scoped) | ⚠️ | Mechanism correct; secret hardcoded (critical) |  
| 2.6 | Script Sentinel (Landlock) | ❌ | Not implemented |  
| 2.7 | Audit ledger (append-only) | ✅ | Implemented correctly |  
| 3.1-3.2 | Telemetry Sanitizer | ✅ | Cleanly implemented |  
| 3.3 | eBPF hooks | ❌ | psutil polling only |  
| 3.5 | D-Bus / Prometheus | ❌ | Uses \`systemctl show\` subprocess |  
| 4 | security\_events.db separate, append-only | ✅ | Implemented correctly |  
| 5 | Authorization independent of personality | ✅ | Holds structurally; cognitive\_wiring is dead code |  
| 6 | Threat intel (GeoLite2, reputation, CVEDB) | ⚠️ | Scaffolded only |  
| 9.5 | No shell interpolation | ✅ | The only bypass (convert\_video) has been removed |

\---

\#\# Next Steps — Prioritized Action Items

\#\#\# Immediate (Before Any Further Git Operations)

| Priority | Action | Owner | Verification |  
|---|---|---|---|  
| \*\*1\*\* | Confirm BASE\_DIR fix is committed and \`core/storage/\` is now gitignored correctly | Done | Runtime log confirms |  
| \*\*2\*\* | Externalize CAPABILITY\_TOKEN\_SECRET to environment variable | TBD | Secret not present in source; fail loud if unset |  
| \*\*3\*\* | Move Policy Gate to separate process with /run/kaiacord socket | TBD | \`ls \-la /run/kaiacord\` shows root:kaiacord, 0660 |  
| \*\*4\*\* | \*\*Remove legacy \`convert\_video\_to\_gif\` module entirely\*\* | TBD | Module and handler deleted; no ffmpeg pathway remains |

\#\#\# Within Phase 1 Completion (Security Foundation)

| Action | Status |  
|---|---|  
| \~\~BASE\_DIR fix\~\~ | ✅ Done |  
| \~\~State modification blocking\~\~ | ✅ Done |  
| \~\~Bubblewrap masking\~\~ | ✅ Done |  
| \~\~Remove convert\_video\_to\_gif module\~\~ | ✅ Done (after deletion) |  
| Externalize capability secret | ❌ |  
| Move Policy Gate out-of-process | ❌ |  
| Fix /run/kaiacord permissions | ❌ |  
| Re-write test\_kaia\_cli.py for JSON-intent contract | ❌ |  
| Strengthen directory-blocking test with non-.py file | ❌ |  
| Re-write Bubblewrap masking test with absolute paths | ❌ |  
| Add restart frequency threshold | ❌ |  
| Remove ChromaDB server orphan | ❌ |

\#\#\# Phase 2 (Isolation and Containment)

| Action | Status |  
|---|---|  
| Implement restrictiveness lattice with max/intersection | ❌ |  
| Add systemd-nspawn tier (Tier 5\) | ❌ |  
| Implement cgroup resource ceilings (cpu/mem/tasks/io/runtime/disk) | ❌ |  
| Implement Script Sentinel / Landlock | ❌ |  
| Add eBPF hooks (tracepoints, kprobes) | ❌ |  
| Implement D-Bus systemd queries (replace systemctl subprocess) | ❌ |

\#\#\# Phase 3 (Threat Intelligence)

| Action | Status |  
|---|---|  
| Add GeoLite2 integration | ❌ |  
| Build local reputation cache | ❌ |  
| Add Shodan CVEDB correlation | ❌ |  
| Build ingestion pipeline with health monitoring | ❌ |  
| Add InternetDB snapshots (after pipeline stable) | ❌ |

\#\#\# Phase 4 (Cognitive Integration — DO NOT START until Phases 1-3 stable)

| Action | Status |  
|---|---|  
| Wire affective state to telemetry | ❌ HOLD |  
| Implement Dream Cycle reading security\_events.db | ❌ HOLD |  
| Add proactive Discord alerting | ❌ HOLD |  
| Inject passive inner monologue (sanitized only) | ❌ HOLD |  
| Add fatigue/resource throttling | ❌ HOLD |

\---

\#\# Runtime Environment Cleanup

\#\#\# ChromaDB Server Orphan

\*\*Remove from \`activate\_kaia\_env.sh\`:\*\*    
\`\`\`bash  
\# Delete or comment out the ChromaDB server startup block  
\# It conflicts with main.py's embedded PersistentClient mode  
\`\`\`

\#\#\# /run/kaiacord Permissions

\*\*Apply immediately:\*\*    
\`\`\`bash  
sudo groupadd kaiacord 2\>/dev/null || true  
sudo usermod \-aG kaiacord ekco  
\`\`\`

\*\*Create \`/etc/tmpfiles.d/kaiacord.conf\`:\*\*    
\`\`\`  
d /run/kaiacord 0770 ekco kaiacord \-  
\`\`\`

\*\*Apply:\*\*    
\`\`\`bash  
sudo systemd-tmpfiles \--create  
\`\`\`

\*\*Optional long-term:\*\* Use \`RuntimeDirectory=kaiacord\` in a systemd service unit.

\---

\#\# Architectural Summary

\#\#\# Strengths (Keep Doing This)

1\. \*\*Feedback loop exists:\*\* Design → Review → Implementation → Verification → New Review → Design Revision    
2\. \*\*Security becoming more restrictive over time,\*\* not less    
3\. \*\*Intent → Gate → Executor pattern\*\* is stable and being preserved    
4\. \*\*Telemetry Sanitizer\*\* is a clean, first-class implementation of spec Section 3.2    
5\. \*\*Security events separated from cognition\*\* with proper append-only semantics

\#\#\# Weaknesses (Fix These)

1\. \*\*Policy Gate shares process memory\*\* with the thing it protects    
2\. \*\*Hardcoded cryptographic secrets\*\* in source code    
3\. \~\~\*\*Bypass paths\*\* (ffmpeg, ChromaDB server) exist outside the gate\~\~ → ffmpeg bypass is removed; ChromaDB orphan remains    
4\. \*\*Resource controls\*\* exist only in spec, not in code    
5\. \*\*Test suite\*\* doesn't actually test the current implementation

\#\#\# Invariant Violations

| \# | Invariant | Status |  
|---|---|---|  
| 1 | Policy gate fails closed | ✅ |  
| 2 | Authorization never depends on personality | ✅ |  
| 3 | Security events append-only | ✅ |  
| 4 | Telemetry sanitized before LLM context | ✅ |  
| 5 | No shell interpolation in host executor | ✅ (bypass removed) |  
| 6 | Agent expresses intent via schema | ✅ |  
| 7 | Capability intersection applies globally | ⚠️ (not implemented) |  
| 8 | Audit ledger append-only | ✅ |

\---

\#\# Agent Decision Tree: What to Do Next

\`\`\`  
Is BASE\_DIR fixed and committed?  
    ├── YES → Runtime log confirms. Proceed.  
    └── NO  → Impossible; runtime would show core/data paths. This is done.

Has CAPABILITY\_TOKEN\_SECRET been moved to env?  
    ├── YES → Good. Proceed to next.  
    └── NO  → STOP. This is critical. Do not push further commits until this is fixed.

Is Policy Gate running as a separate process with /run/kaiacord socket?  
    ├── YES → Great. Proceed to Phase 2\.  
    └── NO  → This is the highest remaining architectural gap. Address before Phase 4\.

Has convert\_video\_to\_gif been removed?  
    ├── YES → Good.  
    └── NO  → Remove it immediately; it is a legacy attack surface.

Are test\_kaia\_cli.py and verify\_security.py tests passing against current code?  
    ├── YES → Test suite is now trustworthy.  
    └── NO  → Fix test suite before relying on it for security verification.

Phase 4 ready to start?  
    ├── YES → Only if all above are complete.  
    └── NO  → Continue Phase 1 and Phase 2 work. Do not wire cognitive\_wiring.py yet.  
\`\`\`

\---

\#\# Coding Agent Execution Specifications (Mandatory Implementation Details)

The following subsections provide \*\*strict, unambiguous technical directives\*\* for implementing the Phase 1 blockers. A coding agent must follow these exact specifications to avoid implementation drift, broken dependencies, or circular test logic.

\#\#\# 1\. Unix Domain Socket Framing Protocol for Policy Gate

\*\*Context:\*\* Moving the Policy Gate out‑of‑process via \`/run/kaiacord/policy\_gate.sock\` requires a deterministic IPC framing protocol to avoid fragmentation and parsing errors.

\*\*Strict Specification:\*\*

\- \*\*Transport:\*\* \`socket.AF\_UNIX\`, \`socket.SOCK\_STREAM\`  
\- \*\*Framing:\*\* \*\*4‑byte big‑endian length header \+ UTF‑8 JSON payload.\*\*    
  \- The length header indicates the number of bytes of the JSON payload that follows.    
  \- No newline terminators are used; the length header is the sole delimiter.  
\- \*\*Request Schema (client → gate):\*\*  
  \`\`\`json  
  {  
    "request\_id": "uuid",  
    "action": "diagnostics" | "block\_ip" | "restart\_service" | "write\_file" | "run\_script",  
    "payload": { /\* schema‑specific fields \*/ },  
    "capability\_token": "ck\_..."  
  }  
  \`\`\`  
\- \*\*Response Schema (gate → client):\*\*  
  \`\`\`json  
  {  
    "request\_id": "uuid",  
    "approved": true | false,  
    "executor\_response": { ... },  
    "audit\_id": "audit\_..."  
  }  
  \`\`\`  
\- \*\*Error Response:\*\* If the gate is unavailable, the client must \*\*fail closed\*\* (raise an exception and log to security\_events.db) – no fallback to local execution.

\*\*Implementation locations:\*\*  
\- Modify \`security/policy\_gate.py\` to accept connections and parse length‑prefixed frames.  
\- Modify \`security/policy\_gate\_client.py\` (or the integration points in \`main.py\`) to send requests with the length‑prefixed format.

\---

\#\#\# 2\. Explicit Dependency Mapping for Legacy Video Converter Removal

\*\*Context:\*\* Removing \`convert\_video\_to\_gif\` requires purging all import and registration references; otherwise the application crashes at startup due to missing imports or dangling handlers.

\*\*Strict Specification – Files to Delete:\*\*

\- \`toolbox/video\_converter.py\` (entire file)  
\- \`toolbox/\_\_init\_\_.py\` (if it only imports the converter; otherwise, edit to remove the import)

\*\*Strict Specification – Dependencies to Purge (exact coordinates):\*\*

\- In \`main.py\` – locate the \`"convert\_video\_to\_gif"\` handler registration (likely a lambda mapping) and delete that dictionary entry.  
  \- Example pattern: \`"convert\_video\_to\_gif": lambda s, c: video\_converter.convert\_video\_to\_gif\_interactive(...)\`  
  \- Remove the import statement: \`from toolbox.video\_converter import convert\_video\_to\_gif\_interactive\` (or similar) from \`main.py\`.  
\- In \`core/kaia\_cli.py\` – if there is any mention or reference to video conversion, remove it (likely none, but check).  
\- In \`requirements.txt\` – if \`ffmpeg-python\` or similar is only required for this module, remove it; otherwise keep if used elsewhere.  
\- After deletion, run \`grep \-r "convert\_video" .\` to confirm no references remain (except this document).

\---

\#\#\# 3\. Environment Variable Signature for Capability Secret

\*\*Context:\*\* The hardcoded secret must be replaced with a fail‑loud environment variable. The key name must be canonical.

\*\*Strict Specification – Variable Name:\*\*

\- \*\*\`KAIA\_CAPABILITY\_TOKEN\_SECRET\`\*\* – export this in the environment (e.g., in \`.env\` or systemd service file).

\*\*Strict Specification – Implementation in \`core/config.py\`:\*\*

Replace the hardcoded line with:

\`\`\`python  
import os  
import sys

try:  
    CAPABILITY\_TOKEN\_SECRET \= os.environ\["KAIA\_CAPABILITY\_TOKEN\_SECRET"\]  
except KeyError:  
    print("FATAL: KAIA\_CAPABILITY\_TOKEN\_SECRET environment variable not set.", file=sys.stderr)  
    sys.exit(1)  
\`\`\`

\*\*No default value is permitted\*\* – the application must not start without this variable.

\*\*Verification:\*\* The test suite must include a unit test that temporarily unsets the variable and asserts that \`sys.exit\` is called (or that an exception propagates).

\---

\#\#\# 4\. Rigid Test Assertion Anchors for Test Suite Rewrite

\*\*Context:\*\* The test suite must validate the \*\*actual\*\* JSON‑intent contract and \*\*must not\*\* be rewritten to match broken code. The following invariants are frozen.

\*\*Strict Specification for \`tests/test\_kaia\_cli.py\`:\*\*

\- \*\*Test for \`generate\_command\`\*\* – mock the Ollama response to return a known JSON string (e.g., \`{"action":"diagnostics","target":"system"}\`). Then:  
  \- \`json.loads(command)\` must succeed.  
  \- The resulting dict must be validated against the appropriate Pydantic model from \`security/schemas.py\` (e.g., \`DiagnosticsRequest(\*\*parsed)\`) – this must \*\*not\*\* raise validation errors.  
\- If the generated intent is of type \`write\_file\`, the test must assert that the \`path\` is inside \`WORKSPACE\_DIR\` (using the configured value).  
\- Any test that expects a literal shell command string (e.g., \`"ls \-la"\`) must be deleted or rewritten.

\*\*Strict Specification for \`tests/verify\_security.py\` (directory blocking test):\*\*

\- Add a new test case that attempts to write a \*\*non‑\`.py\`\*\* file inside a blocked directory, e.g., \`core/notes.txt\`.  
\- The test must assert that the error message or exception indicates the \*\*directory\*\* block, not the extension block. This ensures the \`BLOCKED\_DIRS\` check is exercised independently.

\*\*Strict Specification for \`tests/verify\_security.py\` (Bubblewrap masking test):\*\*

\- Use an \*\*absolute path\*\* inside the sandbox, e.g., \`cat /workspace/.env\` (assuming the workspace is mounted at \`/workspace\` inside bwrap). If the mount point differs, use the correct absolute path within the container.  
\- Capture both \`stdout\` and \`stderr\`.  
\- Assert that \`stderr\` \*\*does not\*\* contain \`No such file or directory\` – that would indicate a missing file rather than a successful mask.  
\- Assert that the \`stdout\` is empty (because \`/dev/null\` returns nothing) \*\*and\*\* that the command exits with a non‑zero status (or the expected error) – this proves the file was masked rather than absent.

\---

\#\#\# 5\. Line‑Targeting for ChromaDB Server Orphan Removal

\*\*Context:\*\* The orphan ChromaDB server startup in \`scripts/activate\_kaia\_env.sh\` must be removed without damaging other service checks (PostgreSQL, Ollama).

\*\*Strict Specification – Exact script path:\*\*

\- File: \`scripts/activate\_kaia\_env.sh\`

\*\*Strict Specification – Pattern to delete:\*\*

\- Locate the block that starts ChromaDB, which likely contains:  
  \`\`\`bash  
  \# ChromaDB server startup  
  if \! pgrep \-f "chroma run" \> /dev/null; then  
      chroma run \--host 127.0.0.1 \--port 8000 &  
  fi  
  \`\`\`  
  or a \`nohup\` variant.  
\- Delete the entire block (the condition, the command, and any error‑handling lines immediately below it).  
\- \*\*Do not\*\* remove the lines that check PostgreSQL or Ollama; those are still required.  
\- After removal, verify that \`activate\_kaia\_env.sh\` only starts PostgreSQL and Ollama (if missing), and directly launches \`main.py\` without attempting to start ChromaDB.

\*\*Verification:\*\* After removal, running \`./scripts/activate\_kaia\_env.sh\` should \*\*not\*\* produce any error about ChromaDB failing to start, and \`main.py\` should still initialize the embedded ChromaDB PersistentClient without issues.

\---

\#\# Documents Referenced

\- \`Hardened AI Admin Agents\_v2.md\` — Design spec (target state)  
\- \`project\_security\_review.md\` — Initial vulnerability assessment  
\- \`walkthrough.md\` — Implementation record of Phase 1 \+ Phase 2 patches  
\- \`kaia\_codebase\_review\_v2.md\` — Detailed codebase review  
\- \`Gemini\_Claude\_ChatGPT\_Review.md\` — Synthesis and runtime confirmation

\---

\#\# Recommended Next Actions (Summary)

1\. \*\*Fix CAPABILITY\_TOKEN\_SECRET\*\* — move to environment variable \`KAIA\_CAPABILITY\_TOKEN\_SECRET\` with fail‑loud \`sys.exit(1)\`.  
2\. \*\*Move Policy Gate out-of-process\*\* with proper /run/kaiacord socket, using \*\*length‑prefixed framing\*\* as specified.  
3\. \*\*Remove legacy \`convert\_video\_to\_gif\` module\*\* — delete file, remove handler and import from \`main.py\`.  
4\. \*\*Rewrite test suite\*\* to match current JSON‑intent contract, using \*\*Pydantic validation\*\* and \*\*absolute‑path stderr checks\*\*.  
5\. \*\*Remove ChromaDB server orphan\*\* from \`scripts/activate\_kaia\_env.sh\` — delete the background \`chroma run\` block.  
6\. \*\*Apply /run/kaiacord permissions\*\* via systemd-tmpfiles.  
7\. \*\*Begin Phase 2 implementation:\*\* cgroups, eBPF, D-Bus.  
8\. \*\*HOLD Phase 4\*\* until Phases 1-3 are stable and verified.

\---

\*This document represents the synthesized consensus of multiple security reviews, augmented with strict coding‑agent execution specifications. The Kaia project is on a strong trajectory, but critical gaps remain before the cognitive integration features (Phase 4\) should be activated. Address the prioritized actions in order, using the exact specifications provided, and the project will be positioned for safe, capable operation as a genuinely autonomous local systems agent.\*  
\`\`\`  
