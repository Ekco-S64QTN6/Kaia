#!/usr/bin/env python3
"""Re-apply IP blocks recorded in the audit ledger.

Blocks live in the `inet kaia_block` nftables table, which is runtime-only: a
reboot clears every address Kaia has ever blocked, silently. The ledger is the
system of record for what was blocked and why, so it is also the right source
to rebuild from.

Runs at boot via kaia-restore-blocks.service, before the Policy Gate. Re-uses
HostExecutor.execute_mitigation so the table, chain and address-family logic
have exactly one implementation.

Only entries with result == "approved" are replayed. Malformed lines are
skipped, not fatal -- a corrupt ledger must not prevent the host from booting
with its blocks in place.

Exit: 0 (always -- failure to restore is logged, never fatal at boot)
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("kaia-restore-blocks")

# Cap the replay so a very long ledger cannot stall boot indefinitely.
MAX_BLOCKS = int(os.environ.get("KAIA_MAX_RESTORED_BLOCKS", "500"))


def collect_blocks(ledger_path: str) -> list:
    """Return de-duplicated approved block_ip entries, newest definition wins."""
    if not os.path.exists(ledger_path):
        logger.info("No audit ledger at %s; nothing to restore.", ledger_path)
        return []

    blocks = {}
    malformed = 0
    with open(ledger_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                malformed += 1
                continue
            if rec.get("result") != "approved":
                continue
            req = rec.get("request") or {}
            if req.get("action") != "block_ip":
                continue
            ip = req.get("target_ip")
            if not ip:
                continue
            # Keyed by the full tuple so a later, narrower block does not silently
            # replace a broader one for the same address.
            blocks[(ip, req.get("protocol", "all"), req.get("port"))] = req

    if malformed:
        logger.warning("Skipped %d malformed ledger line(s).", malformed)
    return list(blocks.values())


def main() -> int:
    try:
        import config
        from security.host_executor import HostExecutor
    except Exception as e:
        logger.error("Could not import Kaia modules: %s", e)
        return 0

    ledger = config.AUDIT_LOG_PATH
    logger.info("Restoring blocks from %s", ledger)
    blocks = collect_blocks(ledger)

    if not blocks:
        logger.info("No approved block_ip entries to restore.")
        return 0

    if len(blocks) > MAX_BLOCKS:
        logger.warning(
            "Ledger holds %d distinct blocks; restoring the first %d "
            "(raise KAIA_MAX_RESTORED_BLOCKS to change).", len(blocks), MAX_BLOCKS
        )
        blocks = blocks[:MAX_BLOCKS]

    ok = failed = 0
    for req in blocks:
        ip = req.get("target_ip")
        proto = req.get("protocol", "all")
        port = req.get("port")
        success, _, err = HostExecutor.execute_mitigation(ip, proto, port)
        if success:
            ok += 1
        else:
            failed += 1
            logger.error("Failed to restore block for %s: %s", ip, err)

    logger.info("Restored %d block(s); %d failed.", ok, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
