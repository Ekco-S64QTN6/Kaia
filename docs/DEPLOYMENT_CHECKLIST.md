# Deployment and Host Verification Checklist

This checklist records deployment findings last documented on **2026-09-08**. Most steps require `sudo`. Its statements about active services, privileges and installed units are historical observations, not a live host probe. Verify the current host before running commands or assuming a service is in the recorded state.

**Last recorded state (2026-09-08):** the Policy Gate was running from the developer checkout at `~/github/Kaia` with `CAP_SYS_ADMIN` and `CAP_DAC_OVERRIDE`; the hardened deployment was committed but not installed. Re-check `systemctl status kaia-policy-gate`, the unit's `WorkingDirectory` / `ExecStart`, and `KAIA_STATE_DIR` before following this checklist.

**Re-checked 2026-10-24:** unchanged. `/etc/systemd/system/kaia-policy-gate.service` still runs
`~/github/Kaia/.venv/bin/python security/policy_gate.py` as root from the checkout, with
`CAP_SYS_ADMIN` and `CAP_DAC_OVERRIDE`; nothing is deployed under `/opt/kaia`.

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

**eBPF (2026-11-01):** BCC is now installed, and the eBPF probes need only
`CAP_BPF` + `CAP_PERFMON`, which the unit grants. After deploying, confirm
`journalctl -u kaia-policy-gate -b | grep -i 'ebpf\|inode watches'` shows the
probes loaded and the honeypot inode watches active. This does not change the
`CAP_SYS_ADMIN` decision above.

---

## 2. Deploy

**Stop the gate before updating the checkout.** While it runs from `~/github/Kaia`, its
tamper detector hashes every `security/*.py`, `core/config.py`, `main.py`,
`kaia_dashboard.py`, `scripts/kaia-lockdown.sh` and its own unit file every 30 s. A
`git pull` that changes them under it is a tamper event on files that gate authority:
an emergency lockdown (the nftables ruleset flushed, up to three times an hour).

```bash
sudo systemctl stop kaia-policy-gate.service
cd ~/github/Kaia
git pull --ff-only
sudo scripts/install_services.sh
```

What this changes:

- Deploys a **root-owned** copy of the code to `/opt/kaia` (`root:root`, 0644/0755).
  The daemon runs as root; running it from a checkout under `$HOME` means root
  executes code any unprivileged process can rewrite. Tamper detection can report
  that after the fact but cannot prevent it.
- Preflights the signing secret before stopping an active Policy Gate or syncing
  monitored files. It enables but does not start the gate; review the deployed diff,
  create the trusted baseline, then start the service explicitly.
- Creates the configured state directory (default `/var/lib/kaia`, `root:kaiacord`,
  0750) and **migrates** existing audit/event data out of the checkout. Custom state
  paths must be below `/var/lib`, `/srv`, or `/mnt`; deploy paths must be below `/opt`.
  The installer rejects paths overlapping the source checkout or each other. It refuses
unexpected code-tree symlinks before recursive root ownership changes; the sole allowed
`static/dashboard` symlink must resolve to `core/web/static` inside the tree.
- Refuses symlinked or non-regular legacy runtime files during migration and normalizes
  existing ledger/DB files to `root:kaiacord` mode `0640`.
- Installs `kaia-policy-gate`, `kaia-lockdown` and the new
  `kaia-restore-blocks` units, all pointing at `/opt/kaia`.
- Aborts if `KAIA_CAPABILITY_TOKEN_SECRET` is missing, empty, historically leaked,
  or not exactly 64 hexadecimal characters (32 random bytes). Generate a key with
  `openssl rand -hex 32`; the format also prevents newlines or shell/systemd parsing
  syntax from being injected into the installed EnvironmentFile.

**From here on, `~/github/Kaia` is the source and `/opt/kaia` is what runs.**
Re-run the installer after changes.

---

## 3. Review the deployed tree, record a trusted baseline, then start

The installed unit and `kaia-baseline.sh` run `/usr/bin/python3` as root, not the checkout's
`.venv`, so the gate's dependencies must be installed for the system interpreter. Check as
root would see it — `-s` and a clean environment keep your own `~/.local` packages out, which
otherwise make a missing module look present:

```bash
for m in dotenv pydantic psutil watchdog requests aiohttp sqlalchemy yara maxminddb bcc; do
  env -i /usr/bin/python3 -s -c "import $m" 2>/dev/null && echo "ok $m" || echo "MISSING $m"; done
```

On 2026-10-24 seven were missing and `kaia-baseline.sh` failed on `dotenv`. All are packaged
in Arch's `extra`:

```bash
sudo pacman -S --needed python-dotenv python-pydantic python-watchdog python-aiohttp \
    python-sqlalchemy python-yara python-maxminddb python-bcc
```

`yara`, `maxminddb` and `bcc` are imported under `try`: without them the gate still starts,
with YARA scanning, GeoIP and eBPF switched off.

```bash
sudo /opt/kaia/scripts/kaia-baseline.sh
sudo systemctl start kaia-policy-gate
systemctl status kaia-policy-gate --no-pager -n 15
```

System mode refuses to start without a valid baseline. Review the deployed tree and expected diff first; never baseline a suspected compromise. The baseline command refuses missing, symlinked or non-regular protected paths (except the intentionally absent deployed `.env`) before creating a replacement file. The baseline records SHA-256 hashes of every protected file to root-owned
`<KAIA_STATE_DIR>/baseline.sha256` (default: `/var/lib/kaia/baseline.sha256`). The runtime reader opens it without following symlinks, rejects non-regular or multiply-linked files, and in system mode requires UID 0 and no group/world permission bits. Verify its metadata with `stat -c '%u %a %h' "$KAIA_STATE_DIR/baseline.sha256"` (expected owner `0`, private mode such as `600`, link count `1`). In unprivileged development mode only, a missing baseline permits a weaker current-disk bootstrap; edits made while stopped then cannot be detected. The FIM audit database is stored at `<KAIA_STATE_DIR>/fim_audit.db`; the installer migrates legacy history from `/var/lib/secdaemon/fim_audit.db` if the new destination does not already exist. The security database at `<KAIA_STATE_DIR>/security_events.db` also contains the `consumed_capability_tokens` registry. Confirm the table is present after startup and that the database remains under the configured root-owned state directory with the expected `root:kaiacord` ownership and `0640` file mode. The registry stores SHA-256 fingerprints and expiry/consumption timestamps only, not raw token strings or signatures.

**Run the baseline only from a state you have reviewed.** Baselining a compromise
makes the compromise the reference.

Protected-file hash verification opens a no-follow descriptor, refuses non-regular paths and streams in 1 MiB chunks; it preserves the historical SHA-256 digest format and prefix-hash behavior.

Expected in the log: `Loaded N trusted hashes`, `Tamper detection thread started`,
and `Policy Gate listening on Unix socket`. On a custom `KAIA_STATE_DIR`, verify
that `core/config.py` and `scripts/kaia-baseline.sh` resolve the baseline to
`<KAIA_STATE_DIR>/baseline.sha256`, and that `security/fim_daemon.py` and the web HUD
both use `<KAIA_STATE_DIR>/fim_audit.db`. The installer migrates the legacy FIM DB
when applicable. The manual baseline command reads the root-owned `/etc/kaia/secret.env`
and `/etc/kaia/state.env` generated by the installer. The baseline writer hashes to a
temporary file and atomically replaces the trust record only after every hash succeeds;
it preserves state directory access as `root:kaiacord` mode `0750` and keeps the baseline
itself `root:root` mode `0600`.

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
not kernel/firewall-manager integration. The replay parser caps active memory to the newest
configured rules and reads one record at a time with a 16 KiB maximum; it does not yet
compact or impose a total-size limit on the append-only ledger.

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

## Open items

Everything still to build or verify is in [ROADMAP.md](ROADMAP.md).
