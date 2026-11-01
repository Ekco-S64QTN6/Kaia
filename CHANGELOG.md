# Changelog

## Unreleased

### Network discovery

- Parse NBNS NB and NBSTAT answer records, including scoped/compressed owner names and bounded active/unique node-name extraction; reject truncated or malformed records without partial persistence.
- Reject fragmented IPv4 datagrams and reserved-flag frames before passive UDP discovery parsing; retain support for valid DF-only packets and cover both paths with regressions.
- Fix NBSTAT `NAME_FLAGS` masks: they were bit-reversed (ACT `0x0020`, reserved `0xFF80`), so every real node-status name (Windows sends `0x0400`) was discarded. Now RFC 1002 order: G `0x8000`, DRG `0x1000`, CNF `0x0800`, ACT `0x0400`, PRM `0x0200`, reserved `0x01FF`. Tests use wire values.

### Security

- eBPF telemetry compiles again: add `<net/sock.h>`; `struct sock` was incomplete on current kernels, so the whole program (honeypot file watching included) failed to load on every boot.
- Gate unit grants `CAP_BPF` + `CAP_PERFMON` so the probes can load; `CAP_SYS_ADMIN` remains a separate FIM decision.
- Honeypot decoys are also matched by inode at `security_file_open`, catching relative paths, `./file` and recursive greps that never equalled the absolute decoy path; duplicate hits from both probes are reported once.
- Canary flag files `~/flag.txt` and `/root/root.txt` with per-host `KAIA-CANARY-` IDs, installed safely by `scripts/install_canaries.py` and watched as `honeypot_canary_read`. The unit exposes only these two files from home directories (`ProtectHome=tmpfs` + read-only binds).

- `kaia-baseline.sh` aborted after printing its diff whenever any protected file had changed (`diff` exits 1 under `set -euo pipefail`, added in e790749), so a changed tree could never be re-baselined. After a redeploy the Policy Gate then failed its startup integrity check and locked the host down on every restart attempt. The display pipeline is guarded; a behavioural test runs the script's own block under its shell options.
- Enforce timezone-aware, signed capability-token issue and expiry timestamps; reject future-issued, expired, non-positive and over-one-hour tokens, reject malformed signatures, non-string scope/timestamp fields, and excessively nested JSON.
- Preserve the existing capability-token JSON fields and HMAC format. Enforce durable single-use token consumption with an atomic SQLite fingerprint registry; fail closed when the registry cannot be updated. Raw tokens and signatures are not persisted.
- Extend synthetic Policy Gate IPC coverage to fragmented headers/payloads and truncated-frame EOF, asserting incomplete requests close without reaching policy evaluation.
- Restrict HostExecutor subprocess launches to exact registered system-binary paths; reject malformed or NUL-containing argv before spawning a child.
- Load the trusted integrity baseline through a no-follow descriptor; reject symlinks, non-regular files, hardlinks and insecure system-mode ownership/permissions.
- `kaia-restore-blocks.service` loads `/etc/kaia/secret.env`. `core/config.py` exits at import without the signing secret, so the unit failed on every boot and recorded blocks were never re-applied.
- The lockdown table is `inet kaia_lockdown`, not `inet filter` — the stock `/etc/nftables.conf` table has that name, so recovery tooling could not tell the lockdown from the host firewall.
- `kaia-unlock.sh` restores `/etc/nftables.conf` (via `nftables.service`) before `ufw reload` when that service is enabled, and re-applies ledger blocks afterwards; previously a ufw-path unlock left the nftables.conf ruleset and every recorded block missing.
- The capability-token nesting test accepts either rejection path (Python 3.12 raises `RecursionError`; 3.14 parses the list and refuses it as a non-object).
