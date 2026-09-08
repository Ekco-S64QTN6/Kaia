# Pending Setup — commands to run

Work completed 2026-09-08 that needs privileged steps to take effect. Everything
here needs `sudo`; none of it has been applied yet. Safe to leave for later — the
system runs fine in its current state, it just isn't running the hardened
configuration yet.

**Current state:** the Policy Gate is running from the developer checkout at
`~/github/Kaia` with `CAP_SYS_ADMIN` and `CAP_DAC_OVERRIDE`. The code and units
for the hardened setup are committed but not installed.

---

## 1. Decide whether `CAP_SYS_ADMIN` is needed

```bash
cd ~/github/Kaia
sudo python3 scripts/fanotify_probe.py "$PWD"
```

`CAP_SYS_ADMIN` is granted solely for mount-wide `fanotify` (FIM) and BCC/eBPF.
Neither has ever worked on this host — fanotify failed `EINVAL` on every start
since installation, and BCC is not installed — so the unit currently drops it.

`security/fim_daemon.py` was fixed to degrade across four mask tiers instead of
retrying with a mask that fails for the same reason as the first attempt. This
probe reports which tiers the kernel actually accepts.

- **Any tier reports `OK`** → mount-wide FIM is viable. Add `CAP_SYS_ADMIN` back to
  **both** `CapabilityBoundingSet` and `AmbientCapabilities` in
  `scripts/kaia-policy-gate.service`, then continue to step 2.
- **All tiers `FAIL`** → leave it dropped. FIM keeps using the watchdog sentinel
  (which has the scan-interval blind spot INV-009 warns about), and the capability
  would be granting near-root privilege for nothing.

---

## 2. Deploy

```bash
cd ~/github/Kaia
sudo scripts/install_services.sh
```

What this changes:

- Deploys a **root-owned** copy of the code to `/opt/kaia` (`root:root`, 0644/0755).
  The daemon runs as root; running it from a checkout under `$HOME` means root
  executes code any unprivileged process can rewrite. Tamper detection can report
  that after the fact but cannot prevent it.
- Creates `/var/lib/kaia` (`root:kaiacord`, 0750) and **migrates** the existing
  audit ledger and event database out of `storage/security/` — satisfying INV-008,
  which requires the ledger to live outside the agent's working tree.
- Installs `kaia-policy-gate`, `kaia-lockdown` and the new
  `kaia-restore-blocks` units, all pointing at `/opt/kaia`.
- Aborts if `KAIA_CAPABILITY_TOKEN_SECRET` is missing, empty, or set to the value
  that leaked into this repository's git history.

**From here on, `~/github/Kaia` is the source and `/opt/kaia` is what runs.**
Re-run the installer after changes.

---

## 3. Baseline and start

```bash
sudo /opt/kaia/scripts/kaia-baseline.sh
sudo systemctl start kaia-policy-gate
systemctl status kaia-policy-gate --no-pager -n 15
```

The baseline records SHA-256 hashes of every protected file to root-owned
`/var/lib/kaia/baseline.sha256`. Without it, tamper detection hashes whatever is on
disk at startup — so a file edited while the daemon was stopped simply *becomes* the
baseline and is never reported.

**Run the baseline only from a state you have reviewed.** Baselining a compromise
makes the compromise the reference.

Expected in the log: `Loaded N trusted hashes`, `Tamper detection thread started`,
and `Policy Gate listening on Unix socket`.

---

## 4. Live-fire the mitigation path

Not yet exercised against a real ruleset. `execute_mitigation` used to write into
`ip filter` — ufw's own table — so every block was silently discarded by the next
`ufw reload` while the audit ledger still recorded it as applied. It now writes to a
dedicated `inet kaia_block` table at hook priority −10.

Verify with a documentation address (`203.0.113.42`, TEST-NET-3, never routable):

```bash
sudo nft list table inet kaia_block            # expect: no such table, yet
# issue a block_ip intent for 203.0.113.42 through the gate
sudo nft list table inet kaia_block            # expect: the drop rule
sudo ufw reload
sudo nft list table inet kaia_block            # MUST still be there
```

That last step is the actual proof, and what the old implementation failed.

To clean up afterwards:

```bash
sudo nft delete table inet kaia_block
```

---

## 5. Verify block persistence across reboot

`kaia-restore-blocks.service` replays approved `block_ip` entries from the ledger at
boot, ordered `Before=kaia-policy-gate`, so there is no window where previously
blocked traffic is accepted. Untested end to end because no block has ever
successfully applied.

After step 4 succeeds, reboot and confirm the rule returns:

```bash
sudo nft list table inet kaia_block
journalctl -u kaia-restore-blocks -b --no-pager
```

---

## If something goes wrong

Network drops and you suspect the tamper tripwire fired:

```bash
sudo ~/kaia-panic-unlock.sh
```

Standalone by design — no dependency on this repo, no network needed. It stops the
gate first (otherwise the 30s tamper loop re-locks you on the next tick), removes
only Kaia's own lockdown table, reloads ufw and verifies connectivity.

A reboot also clears any lockdown: nftables rules are runtime-only.

---

## Known open items

- **No `unblock_ip` intent exists.** The ledger only ever accumulates blocks, and
  the boot replay has nothing to honour as a removal. Blocks can currently only be
  undone by hand with `nft delete`.
- **Dashboard Stage 2** — interactive command input and response streaming remains
  the largest declared-but-unbuilt feature.
- **`KAIA_CAPABILITY_TOKEN_SECRET` remains in git history** (4 commits, public repo).
  It was rotated 2026-09-08, which is the actual mitigation; scrubbing history does
  not un-publish a value that was already public. `install_services.sh` refuses the
  leaked value outright.
- **`sdb` (the backup drive) reports 2 pending sectors.** Unrelated to Kaia, but it
  is the only backup target on this host.
