# Kaia: Verified Status Review & Phase 3/4 Master Plan
**Reviewer:** Claude, based on a live `git clone` of `github.com/Ekco-S64QTN6/Kaia` (not a static doc read) plus current (June 2026) verification of external data-source terms.
**Relationship to existing docs:** `docs/Kaia_Unified_Architectural_Specification.md` (v6.0) remains the architectural constitution — invariants, axioms, and the general shape of Phases 3–5 there are still correct. This document **replaces its §11.2 "Current Implementation Status"** with verified ground truth, **corrects its threat-intel data-sourcing assumptions**, and **adds the concrete kaiamon.py → live-dashboard integration spec** that v6.0 only sketched. Treat this as a delta, not a rewrite.

---

# PART A — Verified Review of the Antigravity/Gemini Refactor

I cloned the repo directly rather than trusting static file uploads, so everything below is checked against actual current source, not a description of it.

## A.1 What's genuinely done — credit where due

The agent did substantially more than the two patches in `walkthrough.md`. Confirmed by direct code inspection:

- **`BASE_DIR` regression — fixed**, and fixed well. Rather than just reverting the move, `core/data`, `core/personal_context` are now the deliberate, real locations, `BASE_DIR` correctly resolves to repo root, and `.gitignore` was updated to match (`/core/personal_context/`, `/core/data/books/`, `/storage/` all correctly anchored now).
- **`CAPABILITY_TOKEN_SECRET` — fixed.** Loads from `KAIA_CAPABILITY_TOKEN_SECRET` env var, `sys.exit(1)` if unset, no default.
- **`convert_video_to_gif` gate-bypass — fixed by clean deletion.** `toolbox/` is gone entirely, confirmed via a fresh `grep -rni ffmpeg` across the whole repo: zero hits anywhere in the codebase. The only loose end is cosmetic — see A.3.
- **In-process Policy Gate threading (the #1 architectural gap in `project_security_review.md`) — fixed.** `main.py` no longer references `PolicyGate` at all. It's now a genuine standalone OS process, launched via `nohup` from `scripts/activate_kaia_env.sh`, with `/run/kaiacord` created and `chown $USER:kaiacord`'d before launch. This was the single highest-value architectural fix on the whole list, and it's done.
- **Restrictiveness Lattice (spec §1.1) — implemented.** `host_executor.py` resolves `effective_level` via `max(GLOBAL_LATTICE_LEVEL, WORKSPACE_LATTICE_LEVEL)` index lookup, and `policy_gate.py` does real capability-intersection (`GLOBAL_PERMISSIONS ∩ WORKSPACE_PERMISSIONS`) with a "Lattice violation" denial path. `tests/test_advanced_security.py::test_lattice_intersection_denial` exercises it.
- **Multi-tier sandboxing (spec §1.2) — implemented.** `none` / `namespace` / `sandbox-exec` / `bwrap` / `systemd-nspawn` are all real branches in `execute_script`, with sane bwrap fallback for unavailable tiers.
- **cgroup resource ceilings (spec §1.4) — implemented.** Every sandboxed execution gets wrapped in `systemd-run --user --scope -p CPUQuota=... -p MemoryMax=... -p TasksMax=... -p IOWeight=...`. Tested.
- **Script Sentinel (spec §2.6) — implemented.** `telemetry_daemon.py` has a real `watchdog`-based filesystem observer (`ScriptSentinelHandler`), wired into `PolicyGate.start()`/`.stop()`, logging `telemetry_script_sentinel_alert` events. Tested. (See A.3 — there's a packaging bug here.)
- **D-Bus systemd queries (spec §3.5) — implemented with graceful fallback.** `get_systemd_unit_status` tries `gi.repository.Gio`/D-Bus first, falls back to `systemctl show` subprocess if PyGObject isn't available. In practice it'll almost always be using the fallback (PyGObject isn't in `requirements.txt` and isn't reliably pip-installable across distros without system packages) — but the fallback is graceful, not broken, so this is a minor note, not a bug.
- **Service restart gating (spec §2.2) — implemented.** Both the restart-frequency threshold (`RESTART_MAX_FREQUENCY_COUNT`/`WINDOW_SECONDS`) and the telemetry-based health check (via `get_systemd_unit_status`) are real now, not just the allowlist.
- **ChromaDB standalone-server orphan — removed.** Confirmed: zero "chroma" references left in `scripts/activate_kaia_env.sh`. This is also the direct fix for the `nohup ... .venv/bin/chroma: No such file or directory` error from your last log — that error is gone on this version, you don't need to do anything further about it.
- **Test suite hardening — done.** `tests/verify_security.py` now has the isolated `BLOCKED_DIRS` test (`core/notes.txt`, a non-`.py` file) and the bwrap-masking test now uses absolute, `WORKSPACE_DIR`-anchored paths instead of an ambiguous relative `cat .env`. `tests/test_kaia_cli.py` now asserts against the real JSON-intent contract instead of the pre-refactor string-command behavior.

That's a genuinely large, correctly-executed chunk of Phase 1 and Phase 2. Phase 1 ("Security Foundation") is close to actually stable now, not just checkbox-stable.

## A.2 Status of every finding from my last review (`kaia_codebase_review_v2.md`)

| Finding | Status |
|---|---|
| A — `BASE_DIR` bug | ✅ Fixed |
| B — Hardcoded capability secret | ✅ Fixed |
| C — Stale `test_kaia_cli.py` | ✅ Fixed (mostly — see A.3) |
| D — `convert_video_to_gif` bypassing the gate | ✅ Fixed (over-fixed — see A.3) |
| E — `BLOCKED_DIRS` test not isolated | ✅ Fixed |
| F — bwrap masking test using a relative path | ✅ Fixed |
| In-process Policy Gate (project_security_review.md #3) | ✅ Fixed |

Strong showing. The remaining gaps are either new since that review, or were never on either list.

## A.3 New findings from this pass

### 🔴 `watchdog` is missing from `requirements.txt` — breaks Policy Gate on a clean install
`security/telemetry_daemon.py` has an **unguarded, top-level** import:
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
```
`watchdog` isn't in `requirements.txt`. `policy_gate.py` imports `start_script_sentinel`/`stop_script_sentinel` from this module at its own top level, and `policy_gate.py` is now the standalone daemon entry point. On a genuinely clean `pip install -r requirements.txt` (fresh venv, new machine, CI, anyone else cloning the repo), `python security/policy_gate.py` will raise `ModuleNotFoundError` immediately and the entire Policy Gate — meaning the entire privileged-action subsystem — won't start. This is a one-line fix (`echo "watchdog==6.0.0" >> requirements.txt`, pin whatever version you've actually tested against) but it's a real landmine for anyone (including future-you on a reinstall) who isn't carrying an un-tracked manual `pip install watchdog` in their head.

### 🟢 Correction to an earlier version of this report
An earlier draft of this document claimed ffmpeg had a fully-built `FfmpegRequest` schema, a `HostExecutor.execute_ffmpeg` method, and a "Test 5.7" sandboxing test. **That was wrong** — re-checked just now with a fresh `grep -rni ffmpeg` across the entire live repo: zero matches anywhere, and `verify_security.py`'s numbered tests stop at 5.6. There is no ffmpeg subsystem, built or otherwise. The actual state is simpler: `toolbox/video_converter.py` was cleanly deleted. The only remaining trace is the item below.

### 🟡 Dead `convert_video_to_gif` category in the LLM classifier (the only real loose end here)
`config.py`'s `ACTION_PLAN_SYSTEM_PROMPT` still lists `"convert_video_to_gif"` as a valid classification category, and `ACTION_PLAN_EXAMPLES` still has two few-shot examples teaching the LLM to emit it. There's no handler registered for it in `main.py`'s `action_handlers` dict (correctly so, since the feature is gone), so it harmlessly falls through to a generic chat response — but it's worth a two-line prompt edit to remove both references so the classifier stops being taught a capability that doesn't exist.

### 🟡 Minor / polish
- `config.py`'s `generate_command()` service-restart enum still lists `"chroma"` as a restartable service, even though the standalone Chroma server no longer exists anywhere in the project.
- The Policy Gate socket file itself still only gets `os.chmod(0o660)` — group ownership is handled at the `/run/kaiacord` *directory* level by the activate script (`chown $USER:kaiacord`, `chmod 0770`), but without an explicit `chmod g+s`/setgid bit on that directory, new files created inside don't reliably inherit the `kaiacord` group. Worth an explicit `os.chown()` call on the socket itself in `policy_gate.py` rather than relying on directory inheritance.
- `README.md`'s project-structure diagram still shows root-level `data/`/`storage/` siblings of `core/` — that's the pre-fix layout. `data/` and `personal_context/` now correctly live under `core/`. Purely a documentation-drift item, but worth a quick refresh so it doesn't mislead the next agent (or you, in six months).

## A.4 Updated spec compliance (supersedes my prior table for these rows)

| Section | Item | Status |
|---|---|---|
| 1.1 | Restrictiveness Lattice | ✅ Implemented & tested |
| 1.2 | Multi-tier sandboxing | ✅ Implemented (5 of 5 tiers have real branches; gVisor/Firecracker still fall back to bwrap, which is correct/expected — those need external binaries you haven't installed) |
| 1.4 | cgroup ceilings | ✅ Implemented & tested |
| 2.2 | Restart frequency + health gating | ✅ Implemented |
| 2.3 | Socket ownership | ⚠️ Directory-level done; socket-file-level still chmod-only |
| 2.6 | Script Sentinel | ✅ Implemented & tested (pending the `watchdog` requirements fix) |
| 3.5 | D-Bus | ✅ Implemented with correct fallback behavior |
| 9 inv. #5 | No shell interpolation anywhere | ✅ Now holds everywhere — the old `convert_video_to_gif`/`shlex.split` bypass path is gone |

Everything else from my prior table (1.3 systemd-nspawn details/Landlock/BPF hash allowlist, 3.3 eBPF, 3.4 TLS metadata, threat intel, beliefs.json/cognitive wiring) is unchanged — still not started, still correctly deferred per the phase ordering.

---

# PART B — Phase 3/4 Master Plan

## B.1 Immediate cleanup (do this before any new feature work)

1. Add `watchdog` to `requirements.txt` (check your venv's installed version and pin it — this is currently the only thing standing between "clean clone" and "Policy Gate won't start").
2. Scrub the two leftover `"convert_video_to_gif"` references from `config.py`'s `ACTION_PLAN_SYSTEM_PROMPT` and `ACTION_PLAN_EXAMPLES` — the feature is gone, the classifier shouldn't still be taught it exists.
3. Scrub the leftover `"chroma"` entry from the `restart_service` enum in `generate_command()`'s prompt.
4. Refresh `README.md`'s project-structure diagram to match the real `core/data`, `core/personal_context` layout.
5. Add explicit `os.chown()` on the Policy Gate socket file in `policy_gate.py` (not just the parent directory).

None of this is architecturally interesting, but all of it is the kind of thing that causes a confusing failure six weeks from now if left alone, and it's cheap to clear out now while it's fresh.

## B.2 Local Threat Intelligence — corrected data-sourcing plan

I verified current (June 2026) access terms for the three sources floating around in your docs, since the original design assumed things that aren't quite accurate anymore (or never were):

**Shodan InternetDB — free, but not the bulk file.** The per-IP lookup API (`https://internetdb.shodan.io/<ip>`) is genuinely free for non-commercial use, no API key, no Shodan account needed, updated weekly. What is **not** free is the bulk `internetdb.sqlite` snapshot file your docs describe mirroring locally — that's a Shodan Enterprise/bulk-data product now, not something you download for free. The `Hardened AI Admin Agents` design docs' "~15-35GB local SQLite mirror" plan assumed free bulk access that doesn't exist at that price point.

**This is actually good news architecturally, not bad.** Kaia doesn't need the whole internet pre-cached — it only needs to enrich IPs its own telemetry actually observes. So the right design is **lazy, on-demand lookup with local caching**, not a bulk mirror:
- On a new external IP showing up in telemetry (a connection event, a blocked mitigation, etc.), call the free InternetDB API once, cache the result in the `internetdb` SQLite table that already exists in `threat_intel.py`'s schema, respect the ~1 req/sec rate limit.
- This is a smaller, more tractable piece of engineering than the original "ingest a 15-35GB snapshot" plan, and it reuses the existing scaffolding almost exactly — `lookup_internetdb()` just needs its body changed from "query 3 mock rows" to "check cache, miss → call API, store, return."

**MaxMind GeoLite2 — still free, as originally planned.** Free account + license key, download `GeoLite2-City.mmdb` via their `geoipupdate` tool (or a scheduled authenticated `curl`), refresh roughly monthly (their license terms expect periodic re-download, not a one-time pull). No correction needed here, this part of the original plan was accurate.

**Rapid7 Project Sonar — this is probably the "Shodan or somewhere free" you half-remembered.** It's a genuinely free (account required), large-scale, internet-wide IPv4 scan dataset — SYN scan results across common ports for the whole IPv4 space, HTTP/HTTPS response data, certificate metadata, DNS records, refreshed roughly monthly, available at `sonardata.rapid7.com`. The catch: the **total dataset is ~56TB** across all their scan types — nowhere close to something you want to mirror wholesale onto a home workstation. If you want this as a Phase 3 stretch goal, pull exactly one narrow dataset (e.g. just the TCP SYN scan results for a couple of high-signal ports) rather than attempting anything resembling a full mirror.

**Recommended Phase 3 build order, revised:**
1. Live InternetDB lookups + local cache (cheapest, fits existing schema almost exactly, immediately useful).
2. GeoLite2 integration (as originally planned).
3. Local reputation cache population — wire `update_ip_reputation()` into the actual event flow (when `security_events.db` logs a block/violation involving an external IP, call it) instead of leaving it as an unused function.
4. Rapid7 Sonar (one narrow dataset only) — stretch goal, not core Phase 3.
5. Drop the "mirror the full Shodan InternetDB" idea entirely unless you specifically want to pay for Enterprise access later.

## B.3 Curses Dashboard — `docs/kaiamon.py` → live integration spec

`docs/kaiamon.py` is solid, well-structured code (frozen `KaiamonSnapshot` dataclass, proper thread-owned collectors, signal handlers, clean terminal restoration) — it's a good base, not a rewrite. But it's a generic system-companion TUI from a different project; it doesn't know anything about Kaia yet. Here's what actually needs to change to turn it into the dashboard the v6.0 spec's §10.2 ASCII mockup describes:

**Where it goes:** out of `docs/` and into the project root as `kaia_dashboard.py`, sibling to `main.py`. Keep it a single file for now (matching its current monolithic style and the project's generally flat structure) — split into a package later only if it actually grows past a few thousand lines.

**Collector-by-collector changes:**
- `PingCollector` — repoint `PING_TARGETS` away from Discord/Bluesky (leftovers from the old Discord-bot version) toward things actually relevant here: Ollama (`http://localhost:11434`), a DNS reachability check, maybe the Policy Gate socket itself.
- `ServiceCollector` — repoint `MONITORED_SERVICES`/`SERVICE_SEARCH_TERMS` from `ollama.service`/`kaiacord.service`/`NetworkManager.service` to what you actually run: Ollama, PostgreSQL, and critically the Policy Gate daemon process (reuse the exact `pgrep -f "security/policy_gate.py"` pattern `activate_kaia_env.sh` already uses for consistency).
- `TelemetryCollector` (CPU/GPU thermals) — keep close to as-is. It's generic and still useful (you want to know if the LLM workload is cooking your GPU regardless of what else changed).
- `LogCollector` — this is the part that needs real rework, not a repoint. Right now it tails generic `journalctl` with hardcoded keyword filters. It needs to become an **`AuditLogCollector`** that reads `security_events.db` (via `security/db.py`'s `query_security_events`) and `audit_ledger.json` instead — since SQLite has no native `tail -f`, poll on a timer (e.g. every 250ms, `SELECT ... WHERE rowid > last_seen_rowid`) rather than trying to stream it. The existing LPS-bar math and severity-coloring logic are still good and worth keeping, just feed them from policy-gate decision rate instead of generic syslog rate.
- **New: `CognitiveStateCollector`** — reads the current `AffectiveState` (valence/arousal/energy + mood string from `cognitive_wiring.py`). The real design question here: `AffectiveState` currently only exists inside `main.py`'s in-process `AppState`. If the dashboard runs as a genuinely separate process (which I'd recommend — keeps a slow LLM response from ever blocking a screen redraw, and vice versa), there's no way for it to read that in-memory object directly. Simplest fix: have `main.py` periodically (every few seconds, alongside its existing `update_system_prompts` cycle) write a small snapshot — `{valence, arousal, energy, mood, mood_status, timestamp}` — to `storage/runtime_status.json`, and have the dashboard poll that file. Cheap, fully decouples the two processes, and doesn't require any IPC machinery you don't already have.

**The chat panel is new work, not a port.** `kaiamon.py` as written is read-only — no input handling at all (its docstring literally says it's built for a tiled-terminal-quadrant companion role, sitting *next to* another terminal, not replacing one). The v6.0 mockup's bottom "Interactive Operator Chat" panel with a live input cursor is a genuinely new feature: curses input handling, routing submitted text into `process_user_input()`, and rendering a *streaming* LLM response token-by-token into a fixed-height scrolling sub-pane. That last part is the hard one — `kaiamon.py`'s existing log-pane scrolling logic is a reasonable starting point to adapt for it, but budget this as the single largest chunk of new work in the whole dashboard effort, not a quick wire-up like the collectors above.

**Suggested sequencing**, since trying to do all of this in one pass invites exactly the kind of half-finished state Part A just catalogued:
1. Ship the four read-only panes first (ping/services/thermals/audit-log), no chat panel yet — this alone gets you the "actually see what's going on" visibility you said is the real goal, and it's a much smaller, self-contained piece of work.
2. Add `CognitiveStateCollector` once the affective-state JSON snapshot exists.
3. Add the interactive chat panel last, as its own pass, once the rest is stable.

## B.4 Priority order for the next agent handoff

1. `watchdog` → `requirements.txt` (one line, unblocks clean installs)
2. Scrub dead `convert_video_to_gif`/`chroma` references from the LLM prompts (two minutes, pure hygiene)
3. Threat intel Phase 3, step 1 only (live InternetDB lookups + cache) — small, self-contained, immediately useful
4. Dashboard, stage 1 only (read-only panes, per B.3 sequencing)
5. Everything else in this doc, in roughly the order written

I'd hand this document to the agent as-is alongside `docs/Kaia_Unified_Architectural_Specification.md` — this one has the corrected/verified specifics, that one still has the full architectural reasoning and invariants behind them.
