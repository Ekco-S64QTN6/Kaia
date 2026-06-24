Phase 3 Implementation: Cleanup → Threat Intel → Dashboard Stage 1
Following the priority order from 
kaia_status_review_and_masterplan.md
 §B.4.

Proposed Changes
Part 1: Immediate Cleanup (B.1 items)
These are small, high-confidence hygiene fixes. Most are already partially resolved.

✅ Item 1 — watchdog in requirements.txt
Already done. Line 161 of 
requirements.txt
 already has watchdog==6.0.0. No action needed.

[MODIFY] 
config.py
 — Scrub dead convert_video_to_gif references
Remove the convert_video_to_gif category from ACTION_PLAN_SYSTEM_PROMPT (line 72) and the two few-shot examples from ACTION_PLAN_EXAMPLES (lines 135-138) that teach the classifier a capability that no longer exists.

[MODIFY] 
config.py
 — Remove dead chroma from restart_service prompt in generate_command()
Wait — the generate_command() function is in 
kaia_cli.py
, not config.py. The masterplan says "config.py's generate_command() service-restart enum" but the actual location is kaia_cli.py line 445.

[MODIFY] 
kaia_cli.py
 — Remove "chroma" from restart_service enum
Remove "chroma" from the service allowlist in the LLM prompt, leaving "nginx" | "postgresql" | "ollama".

[MODIFY] 
policy_gate.py
 — Remove "chroma" from ALLOWED_SERVICES
Remove "chroma" from the deterministic allowlist.

[MODIFY] 
host_executor.py
 — Remove "chroma" from ALLOWED_SERVICES
Same scrub.

[MODIFY] 
main.py
 — Remove convert_video_to_gif from action plan fallback
The inline system prompt at line 596 still lists 'convert_video_to_gif' in its valid action enum. Remove it.

[MODIFY] 
README.md
 — Refresh project structure diagram
The 📂 Project Structure section (lines 82-127) is already close to accurate — it shows core/data/ and core/personal_context/ correctly, and the storage/ tree is correct. The only inaccuracy is the "Getting Started" section (line 169) which still says "Initialize all local dependencies (Ollama, ChromaDB, Postgres)" — the standalone ChromaDB server was removed. ChromaDB still exists as a Python library (used as PersistentClient in main.py), but it's not a separate service to initialize. Update the language.

Also add test_advanced_security.py to the test listing since it's missing from the README.

✅ Item 5 — Socket os.chown()
Already done. 
policy_gate.py
 already has explicit os.chown() on the socket file with group kaiacord. No action needed.

Part 2: Live InternetDB Lookups + Cache (B.2 step 1)
This is the first concrete Phase 3 feature: convert lookup_internetdb() from querying a local mock database to performing lazy, on-demand lookups against Shodan's free InternetDB API, with local SQLite caching.

[MODIFY] 
threat_intel.py
Changes:

Add a new function _fetch_internetdb_api(ip) that calls https://internetdb.shodan.io/{ip} (free, no API key, no auth).
Modify lookup_internetdb(ip) to implement cache-first, API-fallback logic:
Check the local SQLite internetdb.db first (existing behavior).
On cache miss, call the free API, parse the JSON response, store in the local cache, and return.
On API error/timeout, return the "not found" fallback (fail-open for enrichment, not for security decisions).
Add a _cache_internetdb_result(ip, data) helper to write API responses into the existing SQLite schema.
Add a INTERNETDB_CACHE_TTL_DAYS = 7 constant — re-fetch if the cached entry is older than 7 days (InternetDB updates weekly).
Add last_updated TEXT column to the data table schema in initialize_intel().
Respect ~1 req/sec rate limit via a simple time.sleep(1) after each API call.
IMPORTANT

This makes outbound HTTP requests from the threat intel module. The requests library is already in requirements.txt. The API is https://internetdb.shodan.io/{ip} — no API key needed, genuinely free for non-commercial use.

Part 3: Dashboard Stage 1 — Read-Only Panes (B.3 sequencing step 1)
Move kaiamon.py out of docs/ and adapt it as the Kaia dashboard with four read-only panes (no chat panel yet).

[NEW] 
kaia_dashboard.py
Adapt the existing 
kaiamon.py
 TUI into a Kaia-specific dashboard. This is a large file (~51KB), so I'll need to study it carefully and make targeted modifications rather than rewriting.

Collector changes:

PingCollector → Repoint targets from Discord/Bluesky to:

Ollama: http://localhost:11434
DNS reachability check (e.g., 1.1.1.1)
Policy Gate socket existence check (/run/kaiacord/policy_gate.sock or fallback)
ServiceCollector → Repoint from old services to:

ollama.service
postgresql.service
Policy Gate daemon process (pgrep -f "security/policy_gate.py")
TelemetryCollector (CPU/GPU thermals) → Keep mostly as-is, it's generic system health.

LogCollector → Convert to AuditLogCollector:

Read from security_events.db via SQLite (SELECT ... WHERE rowid > last_seen_rowid)
Read from audit_ledger.json
Poll every 250ms instead of tailing journalctl
Keep existing LPS-bar and severity-coloring logic, feed from policy gate decision rate
Layout changes:

Keep the existing color scheme (cyan borders, magenta headers, green/yellow/red severity)
Adapt the panel layout to match the §10.3 mockup from the architecture spec
Four panes: System/Ping, Services, Thermals/Telemetry, Audit Log
Open Questions
IMPORTANT

InternetDB rate limiting strategy: The masterplan suggests ~1 req/sec. Should I implement a simple time.sleep(1) inline, or a proper token-bucket rate limiter? Simple sleep seems appropriate for the current scale.

IMPORTANT

Dashboard: kaiamon.py is 51KB. I need to read it in full to understand its collector architecture before adapting it. The plan is to copy it to kaia_dashboard.py and make targeted modifications, preserving the existing curses/color/bar logic. Does this approach sound right, or would you prefer a clean rewrite?

IMPORTANT

ChromaDB references in main.py: ChromaDB is still actively used as an embedded PersistentClient for vector storage (LlamaIndex integration). The masterplan only says to remove the standalone server references from the service restart allowlists and activation scripts, NOT to remove ChromaDB usage entirely. I will leave main.py's ChromaDB import/usage untouched and only remove it from the service restart allowlists. Correct?

Verification Plan
Automated Tests
bash

# Run the existing test suite to ensure cleanup changes don't break anything
python -m pytest tests/ -v
# Verify no remaining dead references
grep -rn "convert_video_to_gif" core/ security/ main.py
grep -rn '"chroma"' core/kaia_cli.py security/policy_gate.py security/host_executor.py
Manual Verification
InternetDB live lookup: Test with a known IP (e.g., 8.8.8.8) and verify cache-miss triggers API call, second call hits cache.
Dashboard: Launch python kaia_dashboard.py and verify the four read-only panes render correctly with live data.
