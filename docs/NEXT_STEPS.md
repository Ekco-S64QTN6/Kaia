# Deployment and Host Verification Checklist

This checklist records deployment findings last documented on **2026-09-08**. Most steps require `sudo`. Its statements about active services, privileges and installed units are historical observations, not a live host probe. Verify the current host before running commands or assuming a service is in the recorded state.

**Last recorded state (2026-09-08):** the Policy Gate was running from the developer checkout at `~/github/Kaia` with `CAP_SYS_ADMIN` and `CAP_DAC_OVERRIDE`; the hardened deployment was committed but not installed. Re-check `systemctl status kaia-policy-gate`, the unit's `WorkingDirectory` / `ExecStart`, and `KAIA_STATE_DIR` before following this checklist.

---

## 1. Decide whether `CAP_SYS_ADMIN` is needed

```bash
cd ~/github/Kaia
sudo python3 scripts/fanotify_probe.py "$PWD"
```

The last recorded host observation is 2026-09-08: fanotify returned `EINVAL` for
the available mask attempts and BCC was not installed, so that unit dropped
`CAP_SYS_ADMIN`. Re-run the probe against the currently deployed kernel and unit;
do not assume the old host state is still true or infer that the source implementation
is absent.

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

## 4. Live-fire the block/unblock lifecycle

This section had not yet been exercised against a real target ruleset in the last
recorded host checklist. The software now has block, unblock and chronological ledger
replay for both actions. Verify actual kernel behavior before relying on it.

Use a documentation address such as 203.0.113.42 (TEST-NET-3; not routable):

1. Confirm there is no active Kaia block for the test address.
2. Through the local HUD, issue: block 203.0.113.42
3. Verify the rule in: sudo nft list table inet kaia_block
4. Run sudo ufw reload and verify the block still exists.
5. Check that the ledger contains an approved block_ip record.
6. Issue through the HUD: unblock 203.0.113.42
7. Verify the exact rule is absent and the ledger has an approved unblock_ip record.

Do not use "nft delete table inet kaia_block" as cleanup for this test. Deleting the
table manually does not record the intended removal, so boot replay may correctly
re-create the still-active ledger block.

---

## 5. Verify block and unblock replay across reboot

After the live-fire steps above, test both states. Use two separate reboot cycles:

**A. Active block survives reboot**

- Apply block 203.0.113.42 and verify it in inet kaia_block.
- Reboot the host.
- Confirm the table contains the rule.
- Inspect journalctl -u kaia-restore-blocks -b --no-pager for replay results.

**B. Approved unblock survives reboot**

- Issue unblock 203.0.113.42 through the HUD and verify it is absent.
- Reboot again.
- Confirm the rule remains absent; the later approved unblock tombstone must prevent replay from resurrecting the older block.
- Inspect the audit ledger and restore-service log if the rule returns.

These are target-host acceptance tests. CI verifies ledger folding and command construction,
not kernel/firewall-manager integration.

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

## Known open items — status after the 2026-10-22 code updates

- **Incremental command-response streaming is not implemented.** The web HUD has interactive input, command history and one-shot response rendering. Telemetry itself streams over WebSocket; long-running command output is not yet forwarded chunk by chunk.
- **Execution-blocking FIM remains unbuilt.** Current fanotify uses notification mode and does not hold the originating process for a verdict. A separate permission-event design and target-kernel tests are required.
- **eBPF is partial and host-dependent.** The unlinkat callback is wired to the HUD as an observed syscall-entry attempt; broader hook coverage, successful-delete correlation and target-kernel/BCC verification remain open.
- **Passive discovery is missing NetBIOS parsing.** The roadmap lists it, but it has not been implemented.
- **Supply-chain monitoring is a first increment.** Completed pacman transactions are recorded, but AUR build/repository change tracking and dependency/signature attestation remain open.
- **Tier 5 systems not yet implemented:** process/network/file behavioral baselines; alert-confidence scoring; secrets-exposure scanning; backup existence, recency and restorability monitoring; independent service/data-freshness supervision; expanded GeoLite2/CVE/passive-DNS/historical threat-intelligence stores and DuckDB analysis.
- **Target-host acceptance tests remain required.** Verify the installed unit and trusted baseline, fanotify/BCC behavior, real block/unblock, UFW reload survival and two-step block/unblock replay across reboot. The 2026-09-08 setup notes above are historical, not a live system probe.
- **Credential history:** the old KAIA_CAPABILITY_TOKEN_SECRET value remains in public Git history, though it was rotated on 2026-09-08 and the installer rejects it. Rotation is the mitigation; deleting history cannot make the published value secret again.
- **Backup-drive note:** the last handoff mentioned pending sectors on sdb; re-check the device independently rather than treating that old note as current.
