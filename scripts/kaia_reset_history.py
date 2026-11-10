#!/usr/bin/env python3
"""Start Kaia's history fresh, archiving (never deleting) what came before.

For the move from testing to real use: the security event database and the
audit ledger hold test activity (decoy reads, deliberate privilege tests,
dashboard polls) that would otherwise sit in every count on the HUD. Both are
append-only and watched by tamper detection, so this is an operator action with
the Policy Gate stopped:

1. stop the gate;
2. refuse if Kaia has active IP blocks (boot replay rebuilds them from the
   ledger; unblock them first or keep the history);
3. move security_events.db (+WAL), audit_ledger.json and the behaviour baselines
   into ``$STATE_DIR/archive/<UTC time>/`` (root only, 0700);
4. create a fresh event database that keeps the capability-token replay
   registry (tokens already used stay used), and an empty ledger;
5. start the gate (tamper detection baselines the new files at start).

Run as root:  sudo /usr/bin/python3 scripts/kaia_reset_history.py [--yes]
"""

import argparse
import importlib.util
import os
import sqlite3
import subprocess
import sys
import time

STATE_DIR = "/var/lib/kaia"
HISTORY = ("security_events.db", "security_events.db-wal", "security_events.db-shm", "audit_ledger.json")
BASELINES = ("exec_baseline.json", "conn_baseline.json", "delete_rate_baseline.json")
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS security_events (
        event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, type TEXT NOT NULL, source TEXT NOT NULL,
        actor TEXT NOT NULL, payload_hash TEXT, disposition TEXT NOT NULL, session_id TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS consumed_capability_tokens (
        token_fingerprint TEXT PRIMARY KEY, expires_at_epoch REAL NOT NULL, consumed_at_epoch REAL NOT NULL)""",
)


def active_blocks(ledger: str) -> list:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaia_restore_blocks.py")
    spec = importlib.util.spec_from_file_location("kaia_restore_blocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_blocks(ledger)


def reset(state_dir: str, now: float = None, chown=True) -> dict:
    """Archive history and create fresh files. The gate must be stopped."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    archive = os.path.join(state_dir, "archive", stamp)
    os.makedirs(os.path.join(state_dir, "archive"), mode=0o700, exist_ok=True)
    os.mkdir(archive, 0o700)

    db = os.path.join(state_dir, "security_events.db")
    if os.path.isfile(db) and not os.path.islink(db):
        conn = sqlite3.connect(db)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # fold the WAL in before moving
        finally:
            conn.close()

    moved = []
    for name in HISTORY + BASELINES:
        src = os.path.join(state_dir, name)
        if os.path.lexists(src):
            if os.path.islink(src):
                raise RuntimeError(f"refusing a symlink in the state directory: {src}")
            os.rename(src, os.path.join(archive, name))
            moved.append(name)

    old_db = os.path.join(archive, "security_events.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        for stmt in SCHEMA:
            conn.execute(stmt)
        kept = 0
        if os.path.exists(old_db):
            conn.execute("ATTACH DATABASE ? AS old", (old_db,))   # only read from
            try:
                kept = conn.execute(
                    "INSERT OR IGNORE INTO consumed_capability_tokens SELECT * FROM old.consumed_capability_tokens"
                ).rowcount
            except sqlite3.Error:
                kept = 0
            conn.commit()
            conn.execute("DETACH DATABASE old")
        conn.commit()
    finally:
        conn.close()
    ledger = os.path.join(state_dir, "audit_ledger.json")
    fd = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    os.close(fd)
    for path in (db, ledger):
        os.chmod(path, 0o640)
        if chown:
            import grp
            os.chown(path, 0, grp.getgrnam("kaiacord").gr_gid)
    return {"archive": archive, "moved": moved, "tokens_kept": kept}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("run as root: sudo /usr/bin/python3 scripts/kaia_reset_history.py", file=sys.stderr)
        return 1
    real = os.path.realpath(STATE_DIR)
    st = os.stat(real)
    if real != STATE_DIR or st.st_uid != 0 or st.st_mode & 0o022:
        print(f"refusing: {STATE_DIR} must be a canonical root-owned directory, not group/world-writable",
              file=sys.stderr)
        return 1
    blocks = active_blocks(os.path.join(STATE_DIR, "audit_ledger.json"))
    if blocks:
        print("Kaia has active IP blocks, which boot replay rebuilds from the ledger:")
        for b in blocks:
            print(f"  {b['target_ip']} {b['protocol']} {b['port'] or ''}")
        print("Unblock them first (HUD console: unblock <ip>), or keep the history.")
        return 1
    if not args.yes and input("Archive the event history and the ledger and start fresh? [y/N] ").strip().lower() != "y":
        print("nothing changed")
        return 1

    subprocess.run(["/usr/bin/systemctl", "stop", "kaia-policy-gate.service"], check=True)
    try:
        result = reset(STATE_DIR)
    finally:
        subprocess.run(["/usr/bin/systemctl", "start", "kaia-policy-gate.service"], check=False)
    print(f"archived {', '.join(result['moved'])}")
    print(f"  into {result['archive']} (root only)")
    print(f"kept {result['tokens_kept']} used-token records; the gate is running on fresh history")
    print("behaviour baselines restart their 7-day learning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
