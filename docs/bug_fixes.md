# Kaia Coding Agent Action Plan

**Date:** 2026-06-25  
**Based on:** `docs/master_plan.md` (treat as `master_plan.md`)  
**Repo:** `Ekco-S64QTN6/Kaia`  
**Agent role:** implement; Claude reviews next cycle

---

## Verified Current State (live repo grep, not memory)

Before touching anything, the agent must know what is actually broken:

| \# | File | Finding |
| :---- | :---- | :---- |
| 1 | `kaia_dashboard.py:1020` | `CollectorManager.__init__` instantiates `LogCollector` — **class does not exist** in this file. Must be `AuditLogCollector`. Blocker: dashboard crashes on launch. |
| 2 | `kaia_dashboard.py:883` | `AuditLogCollector._poll_security_db` queries columns `details, result` — **neither exists** in `security/db.py` schema. Actual columns are `payload_hash, disposition`. Query will raise `OperationalError` at runtime. |
| 3 | `kaia_dashboard.py:1197` | Filter modes cycle `("All", "Net", "Hw")` — `"Sec"` mode is defined as `SEC_FILTER_KEYWORDS` at line 123 but is **never wired** into `handle_input` or `_draw_logs_pane`. Dead constant. |
| 4 | `kaia_dashboard.py:1023` | `log_collector` property return type annotated as `LogCollector` — must be `AuditLogCollector`. |
| 5 | `core/config.py:176-177` | `SECURITY_DB_PATH` → `storage/security_events.db`, `AUDIT_LOG_PATH` → `storage/audit_ledger.json`. master\_plan §2.2 requires `storage/security/security_events.db` and `storage/security/audit_ledger.json`. Paths diverge. |
| 6 | `security/schemas.py` | `from typing import Any` is at **bottom of file**, after it is used in `AuditRecord`. Move to top-of-file imports or the module fails on some import orderings. |
| 7 | `security/cognitive_wiring.py` | File exists. master\_plan §8 declares it **obsolete, remove**. `main.py` still imports it at lines 88-89, 758-762, 864-888 (`AffectiveState`, `beliefs.json`). All references must be purged together. |

---

## Task List (Priority Order)

### TASK 1 — Fix the dashboard crash (blocker)

**Files:** `kaia_dashboard.py`

1. In `CollectorManager.__init__` (line 1020): replace `LogCollector(self._stop_event)` → `AuditLogCollector(self._stop_event)`.  
2. Update the `log_collector` property return type annotation (line 1023): `-> LogCollector` → `-> AuditLogCollector`.  
3. Update all comments still referencing `LogCollector` in `TelemetryCollector` (lines 460, 499\) to say `AuditLogCollector`.

**Verify:** `python kaia_dashboard.py` must launch without `NameError`.

---

### TASK 2 — Fix the DB schema mismatch in AuditLogCollector

**Files:** `kaia_dashboard.py`

The actual `security_events` table columns (from `security/db.py`) are:

event\_id, timestamp, type, source, actor, payload\_hash, disposition, session\_id

Replace the query at line 883 with:

SELECT rowid, timestamp, type, source, actor, payload\_hash, disposition, session\_id

FROM security\_events

WHERE rowid \> ?

ORDER BY rowid ASC

LIMIT 50

Update the unpacking tuple at the next line accordingly:

rowid, ts, event\_type, source, actor, payload\_hash, disposition, session\_id \= row

Update the message-building block to use `disposition` (was `result`) and `payload_hash` (was `details`). The display message should read:

result\_tag \= f"\[{disposition}\]" if disposition else ""

msg \= f"GATE {result\_tag}: {event\_type}"

if source:

    msg \+= f" ({source})"

\# payload\_hash is a hash, not human-readable — omit from display or show truncated

Severity classification: map `disposition == "blocked"` → `ERROR`, `disposition == "approved"` → `SUCCESS`.

---

### TASK 3 — Wire the "Sec" filter mode

**Files:** `kaia_dashboard.py`

The `SEC_FILTER_KEYWORDS` frozenset already exists (line 123). Connect it:

1. In `handle_input`, change filter cycle to `("All", "Sec", "Net", "Hw")`.  
   Per master\_plan §9.4: `s` key should toggle Security-Only. Since `F` is the current filter key and already cycles, just add `Sec` to the tuple and add to `_draw_logs_pane`.  
     
2. In `_draw_logs_pane` filter block, add after the `elif snap.filter_mode == "Hw":` branch:

elif snap.filter\_mode \== "Sec":

    if not any(k in log.message.lower() for k in SEC\_FILTER\_KEYWORDS):

        continue

3. Update the footer hint string from `[F]ilter` to `[F]ilter(All/Sec/Net/Hw)`.

---

### TASK 4 — Fix schemas.py import order

**Files:** `security/schemas.py`

Move `from typing import Any` to the top of the file alongside the other imports. Remove the duplicate `from typing import Any` comment at the bottom. While there, confirm `Dict, List, Literal, Optional, Union` are all imported at the top in one clean block.

---

### TASK 5 — Fix storage path layout

**Files:** `core/config.py`, `security/db.py`, `security/threat_intel.py`

Per master\_plan §2.2, security artifacts belong in `storage/security/`:

In `core/config.py`:

STORAGE\_DIR \= BASE\_DIR / "storage"

SECURITY\_STORAGE\_DIR \= STORAGE\_DIR / "security"

SECURITY\_STORAGE\_DIR.mkdir(parents=True, exist\_ok=True)

SECURITY\_DB\_PATH \= str(SECURITY\_STORAGE\_DIR / "security\_events.db")

AUDIT\_LOG\_PATH \= str(SECURITY\_STORAGE\_DIR / "audit\_ledger.json")

Also add `STORAGE_DIR` as a named constant (currently it's anonymous inline as `BASE_DIR / "storage"`). Keep `PERSIST_DIR = STORAGE_DIR` as an alias so existing references don't break.

**Important:** `security/threat_intel.py` builds its own path off `SECURITY_DB_PATH`'s dirname. After the path move its `THREAT_INTEL_DIR` calculation will naturally resolve to `storage/security/threat_intel/` — confirm this is acceptable per master\_plan §Appendix C, which puts threat intel under `storage/threat_intel/`. If not, define a separate `THREAT_INTEL_DIR = STORAGE_DIR / "threat_intel"` constant in config and import it in `threat_intel.py`.

---

### TASK 6 — Remove cognitive\_wiring.py and all references

**Files:** `security/cognitive_wiring.py` (delete), `main.py`

Per master\_plan §8: Kaia has no affective state. This is not optional.

1. **Delete** `security/cognitive_wiring.py`.  
     
2. In `main.py`, remove:  
     
   - The import `from security.cognitive_wiring import AffectiveState` (line 88\)  
   - `self.affective_state = AffectiveState()` in `AppState.__init__` (line 89\)  
   - The entire `update_system_prompts()` function (lines 721-795) — this function exists only to inject mood directives based on `AffectiveState`. Without cognitive state, it has no purpose. Remove its call at line 899 in `process_user_input` as well.  
   - The `"dream_cycle": handle_dream_cycle` handler entry in the action dispatcher (line 974\) and the `handle_dream_cycle()` function itself (lines 798-888). The function's only meaningful non-cognitive work (querying security events) can be deferred to a future standalone utility.  
   - The `beliefs.json` write block inside `handle_dream_cycle` — gone with the function.

   

3. After removal, `AppState.__init__` should not reference `affective_state` at all.

---

### TASK 7 — Add "Sec" filter label to dashboard footer

**Files:** `kaia_dashboard.py`

In `LayoutManager.calculate_layout`, update the `menu_footer` string for the logs pane:

menu\_footer \= "\[Q\]uit \[C\]lear \[P\]ause \[F\]ilter:All/Sec/Net/Hw"

Keep it under \~50 chars so it fits in narrow terminals.

---

## What NOT to touch this cycle

- `main.py` beyond the cognitive removal in Task 6\. The CLI is a debugging tool per master\_plan §9.5; leave its other logic alone.  
- `security/policy_gate.py`, `security/host_executor.py` — no changes required this cycle.  
- `tests/` — do not modify tests; the agent should verify they still pass after changes.  
- eBPF / fanotify / honeypot / Stage 2 command input — all deferred per master\_plan §10.3 items 4-8.

---

## Acceptance Criteria

The agent's output is accepted when:

1. `python kaia_dashboard.py` launches and all four panes render without crashing.  
2. The `Sec` filter mode cycles correctly via `F` key and filters to `blocked/denied/violation/unauthorized/policy_gate/lattice/sandbox/capability/token/sentinel` keywords.  
3. `python tests/test_advanced_security.py` passes.  
4. `python tests/verify_security.py` passes (Policy Gate socket test).  
5. `python tests/test_database_utils.py` passes.  
6. `security/cognitive_wiring.py` does not exist in the repo.  
7. `grep -r "AffectiveState\|beliefs.json\|cognitive_wiring" . --include="*.py"` returns no matches.  
8. `grep -n "LogCollector" kaia_dashboard.py` returns only comments, no live code.  
9. `python -c "from security.schemas import AuditRecord"` succeeds without error.

---

## Commit message suggestion

fix(dashboard): wire AuditLogCollector, fix schema mismatch, add Sec filter

\- CollectorManager: LogCollector → AuditLogCollector (crash fix)

\- AuditLogCollector.\_poll\_security\_db: align SELECT to actual DB schema

  (payload\_hash/disposition instead of details/result)

\- Wire SEC\_FILTER\_KEYWORDS into handle\_input \+ \_draw\_logs\_pane

\- schemas.py: move Any import to top-of-file

\- config.py: move security artifacts to storage/security/ per spec §2.2

\- Remove cognitive\_wiring.py, AffectiveState, beliefs.json, dream\_cycle

  per master\_plan §8 (Kaia has no affective state)  
