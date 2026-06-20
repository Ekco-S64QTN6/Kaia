# Kaia Codebase Review — Against Hardened AI Admin Agents_v2.md
**Reviewer:** Claude (project-knowledge static review, no code execution)
**Scope:** Full codebase as currently held in project knowledge, checked against `Hardened AI Admin Agents_v2.md` and cross-referenced against `walkthrough.md` / `project_security_review.md`.

---

## 0. Sync Status

The Phase 1 reorganization and both Phase 2 patches described in `walkthrough.md` are present in the codebase I have:

- `core/`, `security/`, `scripts/`, `tests/`, `toolbox/` layout — present.
- `HostExecutor.execute_state_modification` blocks `.py`/`.sh`, hidden files, and `BLOCKED_DIRS = {core, security, tests, scripts, toolbox, storage}` — present.
- `HostExecutor.execute_script` masks `.env`/`kaia.log` via `/dev/null` and `storage/` via `tmpfs` — present.

So: synced for these two items. The findings below are things the existing review didn't flag yet.

---

## 1. 🔴 Critical — Not in your prior review

### A. `BASE_DIR` miscalculation in `core/config.py` — regression introduced by the Phase 1 move

```python
BASE_DIR = Path(__file__).resolve().parent
```

`config.py` now lives at `core/config.py`. `Path(__file__).resolve().parent` therefore resolves to `<repo>/core`, **not** the repo root. Before the reorg (when `config.py` sat at the repo root) this line was correct; moving the file into `core/` without adjusting it broke every path derived from `BASE_DIR`:

| Constant | Computed as | Should be |
|---|---|---|
| `GENERAL_KNOWLEDGE_DIR` | `core/data` | `<repo>/data` |
| `PERSONAL_CONTEXT_DIR` | `core/personal_context` | `<repo>/personal_context` |
| `PERSONA_DIR` | `core/data` | `<repo>/data` |
| `PERSIST_DIR` | `core/storage` | `<repo>/storage` |
| `WORKSPACE_DIR` | `core/` | `<repo>` |

**Confirmed by your own artifacts, not just inference:** the `core/storage/audit_ledger.json` file you uploaded is sitting at exactly the buggy path this bug predicts. That's hard evidence the bug is already live in the running instance, not theoretical.

**Functional impact:**
- `GENERAL_KNOWLEDGE_DIR.mkdir(exist_ok=True)` / `PERSONAL_CONTEXT_DIR.mkdir(exist_ok=True)` silently create *empty* `core/data` and `core/personal_context` directories. `main.py`'s `build_or_load_index()` reads from these — meaning the RAG index is built from empty folders instead of your real `data/` and `personal_context/` content. Kaia's actual knowledge base is effectively not being indexed.
- `load_persona()` looks for `core/data/Kaia_Desktop_Persona.md`, won't find it, and silently falls back to the generic default prompt — Kaia's persona is not loading.

**Security impact:**
- `.gitignore` has `/storage/` (anchored to repo root). With the real storage now at `core/storage/`, that gitignore pattern **no longer matches it**. `core/storage/` — containing `security_events.db`, `audit_ledger.json`, and ChromaDB vectors of your `personal_context/` files — is no longer excluded from git and is one `git add .` away from landing in the repo (public, per the GitHub mirror).
- `WORKSPACE_DIR` being `core/` instead of repo root means the `write_file` boundary check (`abs_path.startswith(workspace_abs)`) now scopes "workspace" to `core/` only. Files outside `core/` (including `main.py` at repo root) fall outside the checked boundary entirely — they're not reachable by `write_file` at all right now, which is incidentally safe, but it also means the boundary doesn't cover what the architecture diagram says it should.

**Fix:** one line.
```python
BASE_DIR = Path(__file__).resolve().parent.parent
```

---

### B. Hardcoded HMAC signing secret for capability tokens

`core/config.py`:
```python
CAPABILITY_TOKEN_SECRET = "kaia_secure_signing_secret_key_2026"  # In production, load from secure environment
```

The entire capability-token model (Section 2.5 of the spec) depends on this secret being unknown to anything that could be compromised. It's currently a plaintext literal in a tracked source file. Anyone who can read `config.py` (repo access, or a compromised agent process reading its own working tree) can compute valid HMAC-SHA256 signatures for `generate_capability_token()` and forge a token for **any** capability/target pair — `restart_service` on any service, `run_script` on any allowlisted script, `write_file` on any path. This silently defeats the "operator approval" step described in Section 2.5, because forged tokens don't need operator approval at all.

The code comment shows this was already known to be a placeholder — flagging it here so it doesn't get lost as "Status: Planned" never gets revisited.

**Fix:** load from environment (it's already using `python-dotenv` and `.env` elsewhere — same pattern):
```python
import os
CAPABILITY_TOKEN_SECRET = os.environ["KAIA_CAPABILITY_SECRET"]  # fail loud if unset, don't default
```

---

### C. `tests/test_kaia_cli.py` tests a contract `generate_command()` no longer has

```python
command, error = self.cli.generate_command("list files")
self.assertEqual(command, "ls -la")
...
self.assertIn("Command not in allowlist", error)
```

The current `KaiaCLI.generate_command()` (in `core/kaia_cli.py`) returns a structured intent **JSON string** (the diagnostics/block_ip/restart_service/write_file schema) and performs no allowlist check or "Command not in allowlist" logic anywhere — that string doesn't appear in the current implementation at all. This test suite is asserting against the pre-refactor behavior and would fail if actually run against current `core/kaia_cli.py`.

This matters beyond housekeeping: it means **this part of the security refactor shipped without test coverage** — `generate_command()`'s new JSON-intent behavior has zero assertions checking it produces valid, well-formed output for the four schema types.

**Fix:** rewrite `test_generate_command_safe`/`unsafe` to mock an Ollama JSON response and assert the parsed dict matches one of the `security/schemas.py` models (e.g. `DiagnosticsRequest(**json.loads(command))` shouldn't raise).

---

## 2. 🟠 High

### D. `convert_video_to_gif` bypasses the Policy Gate entirely

`toolbox/video_converter.py` calls `cli.execute_command(...)` directly (`core/kaia_cli.py`'s raw `subprocess.run` wrapper), and `main.py` wires it up outside the gate:

```python
"convert_video_to_gif": lambda s, c: video_converter.convert_video_to_gif_interactive(s.cli, s.user_id)['response'],
```

No schema validation, no capability token, no allowlist, no audit log entry, no sandboxing — this is the one action handler in `main.py` that never touches `security/policy_gate.py`. It directly violates Design Principle #2 ("all privileged actions are schema-validated by a deterministic host process").

Practically: since `subprocess.run` is called with a list (not `shell=True`), classic shell-injection via a malicious filename isn't possible — worst case from a crafted filename is argument confusion, not arbitrary command execution. The more realistic risk is that `ffmpeg` runs **unsandboxed, on the host**, against video files the user downloaded (i.e., untrusted input) — ffmpeg has a long CVE history for malformed-media parsing bugs. Every other piece of code that touches untrusted/external input runs in bwrap; this one doesn't.

**Fix:** route this through the same `run_script`/`diagnostics`-style flow (schema + audit log at minimum), and consider running the actual ffmpeg invocation inside bwrap given it's processing attacker-influenceable files.

---

### E. `BLOCKED_DIRS` protection isn't actually isolated by the existing tests

`tests/verify_security.py`:
```python
success, stdout, stderr = HostExecutor.execute_state_modification("core/config.py", "VERSION='hacked'")
```
This is reported as testing the `BLOCKED_DIRS` directory check, but `execute_state_modification` checks the `.py` extension **before** it checks `BLOCKED_DIRS`. Since `core/config.py` ends in `.py`, the test passes purely off the extension rule — it never actually exercises the directory-blocking branch. A regression that removed `BLOCKED_DIRS` entirely would not be caught by this test.

**Fix:** add a case with a non-code file inside a blocked directory, e.g. `HostExecutor.execute_state_modification("core/notes.txt", "...")`, asserting it's blocked specifically via the directory check.

---

## 3. 🟡 Worth re-verifying (I can't execute code here to confirm)

### F. The `.env`/`storage` Bubblewrap masking test may pass regardless of whether masking works

In `tests/verify_security.py` Test 5.6, the leak-check script does a **relative** `cat .env` inside the sandbox. `bwrap` is invoked without `--chdir`, and the sandbox's mount namespace only binds `/usr`, fresh `/tmp /run /proc /dev`, and `workspace_abs`. If the process's cwd at exec time isn't itself a path inside one of those binds (e.g. it's the repo root, which — per Finding A — isn't bound at all since `workspace_abs` is `core/`), a relative `cat .env` would fail with "no such file or directory" and print nothing to stdout regardless of whether the `/dev/null` masking line exists. The test currently only checks that stdout is empty, which is also what you'd see from that unrelated failure mode.

I'd treat this masking protection as **unverified** rather than confirmed, until it's re-tested with an explicit absolute path and explicit stderr inspection (to distinguish "blocked by masking" from "file simply not found").

---

## 4. Status of gaps already flagged in your `project_security_review.md`

These were already identified in your review and intentionally left out of the "implement immediately" table — confirming they're still open, not regressed:

| # | Gap | Status |
|---|---|---|
| 3 | In-process Policy Gate threading | Unchanged — `main.py` still starts `PolicyGate` via `threading.Thread` in the same process as the agent loop. |
| 4 | Telemetry polling vs. eBPF | Unchanged — `telemetry_daemon.py` is `psutil`-based; no bpftrace/eBPF hooks. `get_systemd_unit_status` also still shells out to `systemctl show` rather than using D-Bus (Section 3.5). |
| 5 | Threat intel DBs missing | Unchanged — `threat_intel.py` has working query interfaces with safe fallback defaults, but no actual GeoLite2/Shodan data files. Consistent with the spec's own Phase 3 ordering (GeoLite2 + reputation cache first), so not a regression. |

---

## 5. Spec section-by-section compliance

| Section | Item | Status |
|---|---|---|
| 1.1 | Restrictiveness Lattice (max/intersection math) | ❌ Not implemented anywhere in code |
| 1.2 | 5-tier sandboxing | ⚠️ Only Tier 2 (Bubblewrap) implemented; no dynamic tier selection |
| 1.3 | systemd-nspawn, BPF binary-hash allowlist, Script Sentinel/Landlock | ❌ Not implemented |
| 1.3 | SSH/credential/history exclusion from bind-mounts | ✅ Holds, but only because the home directory is never bound at all (not via an explicit exclude-list as spec describes) |
| 1.4 | cgroup resource ceilings (cpu/mem/tasks/io/runtime/disk) | ❌ Not implemented. Only a generic `subprocess` timeout exists, no kernel-enforced limits |
| 2.1 | 4-stage validation pipeline (regex filter → schema → evaluator → executor) | ⚠️ Schema validation and policy evaluation are real, but the "static regex filter" stage only applies to `diagnostics` args — `write_file`/`run_script` validation is embedded directly in `HostExecutor`, not a separate pre-stage |
| 2.2 | Service restart gating: allowlist + frequency threshold + health check | ⚠️ Only the allowlist check exists. No restart-frequency rate limiting, no telemetry-based health gating, despite the spec text implying both are deterministic checks already in place |
| 2.3 | Socket ownership (root:kaiacord, 0660) + SELinux/AppArmor label | ⚠️ Only `chmod 0o660` is set; no explicit group ownership, no MAC label |
| 2.4 | Fail-closed on gate unavailability | ✅ Implemented in `handle_command`/`handle_run_script` |
| 2.5 | Capability tokens (HMAC, expiring, scoped) | ⚠️ Mechanism is correct; secret management is broken (Finding B) |
| 2.6 | Script Sentinel | ❌ Not implemented |
| 2.7 | Audit ledger (append-only, agent can't modify) | ✅ Implemented and correctly protected by `BLOCKED_DIRS` (the protection happens to still cover the real `audit_ledger.json` location because both `WORKSPACE_DIR` and `PERSIST_DIR` derive from the same buggy `BASE_DIR`) |
| 3.1–3.2 | Telemetry Sanitizer | ✅ Cleanly implemented, matches spec closely |
| 3.3 | eBPF hooks | ❌ psutil polling only |
| 3.4 | TLS metadata capture | ➖ Explicitly Phase 5 in spec — correctly not started |
| 3.5 | D-Bus / Prometheus | ❌ Not implemented (uses `systemctl show` subprocess instead) |
| 4 | `security_events.db` separated from cognition store, append-only, agent can't write | ✅ Implemented correctly |
| 4 | `beliefs.json` cognition store | ➖ Not built yet — consistent with Phase 4 ordering |
| 5 | Authorization independent of personality/affect | ✅ Holds structurally — `cognitive_wiring.py`'s `AffectiveState` isn't imported or wired into `main.py`, `policy_gate.py`, or `host_executor.py` anywhere yet. (Worth noting: this also means it's currently dead code, not yet integrated at all.) |
| 6 | Local threat intel (GeoLite2, reputation cache, CVEDB) | ⚠️ Scaffolded with safe fallbacks; real data files absent — matches stated Phase 3 ordering |
| 9 | Invariant #5: no shell interpolation anywhere | ⚠️ Holds for `security/host_executor.py`. Does **not** hold for the `execute_command` bypass path (Finding D), which uses `shlex.split` on a rejoined string — not shell interpolation exactly, but a different, untested parsing path outside the documented executor |

---

## 6. Recommended order of fixes

1. **`BASE_DIR` one-liner** (Finding A) — highest leverage, fixes RAG indexing, persona loading, gitignore coverage, and workspace boundary correctness in one change.
2. **Externalize `CAPABILITY_TOKEN_SECRET`** (Finding B) — currently the weakest link in the whole capability-token model.
3. **Route `convert_video_to_gif` through the gate**, or at minimum log it to the audit ledger and consider sandboxing the ffmpeg call (Finding D).
4. **Update/rewrite `tests/test_kaia_cli.py`** so `generate_command()`'s actual JSON-intent contract has real coverage (Finding C).
5. **Re-verify the Bubblewrap masking test with absolute paths + stderr capture** (Finding F) before trusting it as a proven control.
6. **Strengthen the `BLOCKED_DIRS` test** with a non-`.py` file so the directory check is actually isolated (Finding E).
7. Once 1–6 are done, Phase 1 ("Security Foundation") is genuinely stable — that's the prerequisite the spec itself sets before touching Phase 4 cognitive wiring, so I'd hold off connecting `cognitive_wiring.py` to anything until then.
