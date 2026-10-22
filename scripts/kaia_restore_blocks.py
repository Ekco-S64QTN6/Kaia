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
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("kaia-restore-blocks")

# Bound per-record parsing and in-memory desired state. The ledger itself remains
# append-only; operators should monitor its growth as part of the host checklist.
DEFAULT_MAX_BLOCKS = 500
HARD_MAX_BLOCKS = 10_000
MAX_LEDGER_LINE_BYTES = 16 * 1024


def _parse_max_blocks(raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid KAIA_MAX_RESTORED_BLOCKS=%r; using default %d.",
            raw, DEFAULT_MAX_BLOCKS,
        )
        return DEFAULT_MAX_BLOCKS
    if not 1 <= value <= HARD_MAX_BLOCKS:
        logger.warning(
            "KAIA_MAX_RESTORED_BLOCKS must be 1..%d; using default %d.",
            HARD_MAX_BLOCKS, DEFAULT_MAX_BLOCKS,
        )
        return DEFAULT_MAX_BLOCKS
    return value


MAX_BLOCKS = _parse_max_blocks(
    os.environ.get("KAIA_MAX_RESTORED_BLOCKS", str(DEFAULT_MAX_BLOCKS))
)


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

def collect_blocks(ledger_path: str, *, max_blocks: int = None) -> list:
    """Reconstruct newest active rules with bounded memory and per-record reads.

    A successful unblock is a tombstone for an earlier approved block. Denied
    or failed operations never change desired state. Older active entries are
    evicted as soon as the retention limit is exceeded, equivalent to slicing
    the final active list to the newest max_blocks entries.
    """
    if max_blocks is None:
        max_blocks = MAX_BLOCKS
    if (
        isinstance(max_blocks, bool)
        or not isinstance(max_blocks, int)
        or not 1 <= max_blocks <= HARD_MAX_BLOCKS
    ):
        raise ValueError(
            f"max_blocks must be an integer between 1 and {HARD_MAX_BLOCKS}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(ledger_path, flags)
    except FileNotFoundError:
        logger.info("No audit ledger at %s; nothing to restore.", ledger_path)
        return []
    except OSError:
        logger.exception("Could not safely open audit ledger %s", ledger_path)
        raise

    active = {}
    malformed = 0
    evicted = 0
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Audit ledger is not a regular file: {ledger_path}")
        with os.fdopen(fd, "rb") as f:
            fd = -1  # fdopen owns and closes the descriptor
            while True:
                raw_line = f.readline(MAX_LEDGER_LINE_BYTES + 1)
                if not raw_line:
                    break
                if len(raw_line) > MAX_LEDGER_LINE_BYTES:
                    malformed += 1
                    # Drain the remainder of this record in bounded chunks.
                    while raw_line and raw_line[-1:] != bytes([10]):
                        raw_line = f.readline(MAX_LEDGER_LINE_BYTES + 1)
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip().rstrip(",")
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
                    if len(active) > max_blocks:
                        active.pop(next(iter(active)))
                        evicted += 1

    finally:
        if fd >= 0:
            os.close(fd)

    if malformed:
        logger.warning("Skipped %d malformed or oversized ledger record(s).", malformed)
    if evicted:
        logger.warning(
            "Active-rule replay limit %d exceeded; older active rules were omitted (%d eviction(s)).",
            max_blocks, evicted,
        )
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
    try:
        blocks = collect_blocks(ledger)
    except (OSError, ValueError) as e:
        logger.error("Could not safely reconstruct firewall state from %s: %s", ledger, e)
        return 1

    if not blocks:
        logger.info("No approved block_ip entries to restore.")
        return 0

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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
