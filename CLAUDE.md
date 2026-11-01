# CLAUDE.md — working notes for agents

`README.md` is the front page, `docs/OPERATIONS.md` is how it runs on a host,
`docs/DEPLOYMENT_CHECKLIST.md` is how it gets there, `CHANGELOG.md` is what
changed. This file records what an agent must know before touching the code:
decisions that look wrong and are not, and fixes that must not be undone.

## Running the tests

```bash
PATH="$PWD/.venv/bin:$PATH" bash scripts/run_ci_tests.sh
```

That is the suite CI runs (Python 3.12 in `.github/workflows/security-regressions.yml`).
Running a single file needs the import paths: `PYTHONPATH=.:core pytest tests/<file>.py`.

- The local `.venv` is Python 3.14. A test that passes in CI can still fail here
  if it relies on interpreter limits (see the nesting test below).
- `tests/test_advanced_security.py` is a legacy integration test outside the CI
  list. It fails on its own and leaves module state behind, so
  `tests/test_capability_token_replay.py` fails when run *after* it in one
  `pytest tests/` session. Both were already true before 2026-10-30; neither is a
  regression. Use the CI script.
- **Stale bytecode can fake a result.** Swapping a source file for one of the
  same length within the same second leaves Python using the old `.pyc`. When a
  test result makes no sense after an edit, delete `__pycache__` first.

## Fixes that must not be reverted (2026-10-30)

Committed together, after reviewing ChatGPT's PRs #85–#89.

- **NBSTAT `NAME_FLAGS` masks** (`security/network_discovery.py:_decode_nbstat_hostname`).
  RFC 1002 numbers the bits from the most significant end: G `0x8000`,
  ONT `0x6000`, DRG `0x1000`, CNF `0x0800`, ACT `0x0400`, PRM `0x0200`,
  reserved `0x01FF`. PR #87 had them bit-reversed (ACT `0x0020`, reserved
  `0xFF80`) and built its tests from the same reversed constants, so the tests
  passed while every real node-status name was thrown away: Windows answers
  `0x0400`. Tests now use wire values, and
  `test_nbstat_response_accepts_windows_wire_flags` fails against the old masks.
  → **Build parser fixtures from the wire format, never from the parser's own constants.**
- **`kaia-restore-blocks.service` has `EnvironmentFile=/etc/kaia/secret.env`.**
  `core/config.py` exits at import without `KAIA_CAPABILITY_TOKEN_SECRET`; without
  the line the unit failed every boot and no recorded IP block was ever
  re-applied. `test_units_importing_config_load_the_signing_secret` guards it.
  Existing installs need `sudo scripts/install_services.sh` (or the same line
  added to `/etc/systemd/system/kaia-restore-blocks.service`).
- **The lockdown table is `inet kaia_lockdown`.** It was `inet filter`, which is
  also the name of the table in Arch's stock `/etc/nftables.conf`. The owner's
  `~/kaia-panic-unlock.sh` deleted `inet filter` to lift a lockdown, which on a
  healthy host deleted the real firewall. Do not rename it back to anything a
  distribution ruleset might use.
- **`kaia-unlock.sh` restores `nftables.service` before `ufw reload`**, then
  re-runs `kaia-restore-blocks`. The lockdown flushes the whole ruleset; `ufw
  reload` alone brought back ufw's tables but not `/etc/nftables.conf` or Kaia's
  own blocks. nftables first, because `/etc/nftables.conf` begins with
  `flush ruleset` and would wipe ufw's tables if loaded second.
- **The nesting test accepts either rejection path.** Python 3.12 raises
  `RecursionError` on 10,000 nested brackets ("JSON parsing failed"); 3.14
  parses them and the verifier refuses the list ("must be a JSON object").
  Both reject the token; the test checked only one message.

- **`kaia-baseline.sh` must survive a changed tree.** The "changed:" display
  pipeline ends in `|| true`. Without it, `diff`'s exit 1 killed the script
  under `pipefail` before the baseline was written — exactly when a baseline is
  needed. On 2026-10-30 that left a redeployed host with a stale baseline, and
  the enabled gate locked the network down at boot, nine times, until the
  owner deleted `inet kaia_lockdown` by hand.
  `tests/test_baseline_diff_survives_pipefail.py` guards it.
- **Deploy order is not optional.** `install_services.sh` *enables* the gate.
  If the baseline is not recorded before the next boot, the gate's startup
  integrity check fails and it locks the host down. Never leave a deploy with
  the gate enabled and the baseline stale; disable the gate instead.

## Fixes that must not be reverted (2026-11-01)

- **`#include <net/sock.h>` in the eBPF program.** Without it `struct sock` is
  incomplete on current kernels and the *whole* BPF program fails to compile;
  every probe was off, honeypot file watching included, from at least
  2026-10-25 until this fix. `test_bpf_program_includes_sock_header` guards it.
- **The gate unit carries `CAP_BPF CAP_PERFMON`.** BCC was not installed when the
  2026-09-08 checklist dropped capabilities; with only `CAP_NET_ADMIN CAP_NET_RAW`
  the probes cannot load even when they compile. `CAP_SYS_ADMIN` is still a
  separate decision (fanotify FIM, checklist step 1) — do not add it for eBPF.
- **Honeypots match by inode as well as by path string.** The openat probes only
  see the literal string passed to `open()`, so `cd ~; cat flag.txt`, `./flag.txt`
  and `grep -r` never matched an absolute decoy path. `kprobe__security_file_open`
  matches the opened file's `(s_dev, i_ino)`; an absolute open that hits both
  probes is reported once (2 s dedupe per pid+path).
- **Canary flag files** (`security/canaries/`, installed by
  `scripts/install_canaries.py` from `install_services.sh`): `~/flag.txt` for the
  sudo user and `/root/root.txt`, each with a unique `KAIA-CANARY-…` ID recorded in
  `/etc/kaia/canary_ids`. Reads log `honeypot_canary_read`. The unit uses
  `ProtectHome=tmpfs` plus `BindReadOnlyPaths=` of exactly those two files — do not
  widen it to `read-only`. The installer never overwrites a file it cannot prove is
  a canary. If the owner ever *authorizes* an AI agent or pentest on this host,
  remove the canaries first: their text states no such authorization exists.
- **Honeypot IP attribution uses only sockets the process owns.**
  `/proc/<pid>/net/tcp` lists every socket in the network namespace, so the old
  "first remote address" logic blamed — and persistently auto-blocked — whatever
  unrelated connection came first (an API, Discord, GitHub). Now: socket inodes
  from `/proc/<pid>/fd`, walking only to the grandparent (reverse shell, ssh,
  agent → its API). Unknown ownership means no block. Consequence: if an agent
  *you* run reads a decoy, its model API endpoint gets blocked — by design for a
  hijacked agent; lift it with the HUD unblock. Never self-test a canary from an
  AI agent session.

- **The privilege watch is alert-only by design.** `security/privilege_watch.py`
  watches sudoers, `sudoers.d`, passwd, group and `ld.so.preload`. Those files change
  legitimately (pacman sysusers, the owner's own sudo edits), so the watch must never
  join the trusted baseline, block startup, or call `trigger_lockdown`: wiring it to the
  lockdown tier would lock the owner out on a routine package update. Every call from
  `TamperDetector` is wrapped so a watch failure cannot stop integrity monitoring.
  `tests/test_privilege_watch.py` guards both properties.
- **Every test file must be in `scripts/run_ci_tests.sh`.** CI ran green for a whole
  PR (#95) without ever running its new test file. `tests/test_ci_covers_all_tests.py`
  now fails on an unlisted file; legacy host-dependent scripts are excluded by name.

## Reviewed and correct — do not "fix"

- **Single-use capability tokens** (#86). Every production caller
  (`core/web_bridge.py`, `kaia_dashboard.py`, `security/honeypot.py`, `main.py`)
  mints a fresh token per request, so durable consumption breaks nothing. The
  `consumed_capability_tokens` table is never pruned on purpose: pruning by
  wall-clock time lets a clock rollback revive a used token.
- **HostExecutor executable allowlist** (#89). `execute_script` at lattice level
  `none` builds `[script_path]`, which is not on the allowlist — but the
  mandatory `systemd-run` cgroup wrapper is always prepended (or the call fails
  closed), so `argv[0]` is always allowlisted. The PR's claim that every call
  site uses `SYSTEM_BINARIES` is true only because of that wrapper; keep it
  mandatory.

## Host context

The owner's machine runs both `ufw` and `nftables.service`. `nftables.service`
starts about 2 s after ufw at boot and `/etc/nftables.conf` begins with
`flush ruleset`, so ufw's rules do not survive boot there; nftables.conf is the
effective firewall. Recovery code must not assume ufw is the firewall.
