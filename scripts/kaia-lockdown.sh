#!/bin/bash
# ============================================================
# KAIA EMERGENCY NETWORK LOCKDOWN
# Drops ALL traffic (input, forward, output) on this host.
#
# Invoked by security/policy_gate.py::trigger_lockdown(), normally
# via kaia-lockdown.service, when tamper detection or an operator
# command asserts a compromise.
#
# Reverse with: scripts/kaia-unlock.sh
# ============================================================
set -uo pipefail

COLOR_RED="\033[91m"
COLOR_YELLOW="\033[93m"
COLOR_RESET="\033[0m"

STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"
SNAPSHOT="${STATE_DIR}/pre-lockdown.nft"

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: Lockdown must be executed as root.${COLOR_RESET}"
    exit 1
fi

logger -t kaia-lockdown "EMERGENCY LOCKDOWN ACTIVATED"

# --- Snapshot the existing ruleset BEFORE destroying it ----------------------
# `nft flush ruleset` is total: it removes UFW's tables too, with no way back.
# Without this snapshot a lockdown leaves the host both cut off AND firewall-less
# once it is lifted, which is the worst possible state to investigate from.
# A snapshot failure must NOT prevent containment -- log loudly and continue.
# Keep operator-readable ledger/database traversal intact. Snapshot files are
# separately restricted to root-only mode; lockdown must not change the whole
# state directory to 0700 and break group-readable operator tools.
umask 077
if mkdir -p "$STATE_DIR" 2>/dev/null && getent group kaiacord >/dev/null 2>&1; then
    chown root:kaiacord "$STATE_DIR" 2>/dev/null || true
    chmod 750 "$STATE_DIR" 2>/dev/null || true
fi
if nft list ruleset > "${SNAPSHOT}.tmp" 2>/dev/null && [ -s "${SNAPSHOT}.tmp" ]; then
    # Keep one timestamped copy for forensics, plus a stable name to restore from.
    cp -a "${SNAPSHOT}.tmp" "${STATE_DIR}/pre-lockdown-$(date +%Y%m%d-%H%M%S).nft" 2>/dev/null
    mv "${SNAPSHOT}.tmp" "$SNAPSHOT"
    chmod 600 "$SNAPSHOT" 2>/dev/null
    logger -t kaia-lockdown "Pre-lockdown ruleset snapshot saved to $SNAPSHOT"
else
    rm -f "${SNAPSHOT}.tmp"
    echo -e "${COLOR_YELLOW}WARNING: could not snapshot the current nftables ruleset.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Proceeding with lockdown anyway -- containment takes priority.${COLOR_RESET}"
    logger -t kaia-lockdown "WARNING: ruleset snapshot FAILED; locking down without a restore point"
fi

# --- Contain ----------------------------------------------------------------
# Apply the flush + replacement chains as one nftables batch transaction.
# Running individual commands after a successful flush can leave the host
# without a firewall if a later add fails. Syntax-check first, then apply and
# verify every drop-policy chain before claiming lockdown is active.
LOCKDOWN_RULES="$(mktemp /run/kaia-lockdown.XXXXXX.nft)" || {
    logger -t kaia-lockdown "ERROR: unable to create temporary lockdown ruleset"
    exit 1
}
trap 'rm -f "$LOCKDOWN_RULES"' EXIT

cat > "$LOCKDOWN_RULES" <<'NFT'
flush ruleset
add table inet filter
add chain inet filter input { type filter hook input priority 0; policy drop; }
add chain inet filter forward { type filter hook forward priority 0; policy drop; }
add chain inet filter output { type filter hook output priority 0; policy drop; }
NFT

if ! nft -c -f "$LOCKDOWN_RULES"; then
    logger -t kaia-lockdown "ERROR: lockdown ruleset validation failed; current firewall left unchanged"
    echo -e "${COLOR_RED}ERROR: lockdown ruleset validation failed; current firewall left unchanged.${COLOR_RESET}" >&2
    exit 1
fi

if ! nft -f "$LOCKDOWN_RULES"; then
    logger -t kaia-lockdown "ERROR: atomic lockdown ruleset application failed"
    echo -e "${COLOR_RED}ERROR: could not apply lockdown ruleset.${COLOR_RESET}" >&2
    exit 1
fi

for chain in input forward output; do
    if ! nft list chain inet filter "$chain" 2>/dev/null | grep -q 'policy drop'; then
        logger -t kaia-lockdown "ERROR: lockdown verification failed for chain $chain"
        echo -e "${COLOR_RED}ERROR: lockdown verification failed for chain $chain.${COLOR_RESET}" >&2
        exit 1
    fi
done

logger -t kaia-lockdown "Lockdown ruleset verified: input/forward/output = drop"
echo -e "${COLOR_RED}EMERGENCY LOCKDOWN ACTIVE — all traffic dropped.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}Restore with: sudo ${0%/*}/kaia-unlock.sh${COLOR_RESET}"
