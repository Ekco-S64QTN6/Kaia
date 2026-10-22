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
import ipaddress
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


def _effective_rule_key(ip: str, protocol: str, port, action: str = "block_ip"):
    """Normalize a ledger request to the exact rule shape HostExecutor creates.

    Older approved block records may contain protocol/port combinations that
    the original executor accepted but effectively treated as an IP-wide rule.
    Preserve those historical semantics when replaying blocks. New unblock
    requests are stricter, because ambiguity must never delete a broader rule.
    """
    address = ipaddress.ip_address(ip).compressed
    if protocol not in ("all", "tcp", "udp"):
        raise ValueError("unsupported protocol")
    if port is not None and (
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
    ):
        raise ValueError("invalid port")

    if action == "unblock_ip":
        if protocol == "all" and port is not None:
            raise ValueError("port may only be specified with tcp or udp")
        if protocol in ("tcp", "udp") and port is None:
            raise ValueError("a port is required when protocol is tcp or udp")
        effective_protocol, effective_port = protocol, port
    elif protocol in ("tcp", "udp") and port is not None:
        effective_protocol, effective_port = protocol, port
    else:
        # Preserve the actual legacy executor behavior: protocol without a
        # port, or a port with protocol='all', generated an IP-wide DROP rule.
        effective_protocol, effective_port = "all", None

    return (address, effective_protocol, effective_port), {
        "target_ip": address,
        "protocol": effective_protocol,
        "port": effective_port,
    }

def collect_blocks(ledger_path: str) -> list:
    """Reconstruct the latest approved state for each firewall-rule key.

    A successful unblock is a tombstone for an earlier approved block. Denied
    or failed operations never change desired state. Legacy block_ip-only
    ledgers remain valid and restore exactly as before.
    """
    if not os.path.exists(ledger_path):
        logger.info("No audit ledger at %s; nothing to restore.", ledger_path)
        return []

    active = {}
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
            if not isinstance(rec, dict):
                malformed += 1
                continue
            if rec.get("result") != "approved":
                continue
            req = rec.get("request")
            if not isinstance(req, dict):
                malformed += 1
                continue
            action = req.get("action")
            if action not in ("block_ip", "unblock_ip"):
                continue

            ip = req.get("target_ip")
            protocol = req.get("protocol", "all")
            port = req.get("port")
            try:
                if not isinstance(ip, str):
                    raise ValueError("target_ip must be a string")
                key, normalized = _effective_rule_key(ip, protocol, port, action)
            except (ValueError, TypeError):
                malformed += 1
                continue

            if action == "unblock_ip":
                active.pop(key, None)
            else:
                active.pop(key, None)
                active[key] = normalized

    if malformed:
        logger.warning("Skipped %d malformed ledger line(s).", malformed)
    return list(active.values())

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

    if MAX_BLOCKS < 1:
        logger.error("KAIA_MAX_RESTORED_BLOCKS must be at least 1; refusing to restore.")
        return 1

    if len(blocks) > MAX_BLOCKS:
        logger.warning(
            "Ledger holds %d distinct blocks; restoring the newest %d "
            "(raise KAIA_MAX_RESTORED_BLOCKS to change).", len(blocks), MAX_BLOCKS
        )
        blocks = blocks[-MAX_BLOCKS:]

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
