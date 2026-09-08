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

STATE_DIR="/var/lib/kaia"
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
mkdir -p "$STATE_DIR" 2>/dev/null && chmod 700 "$STATE_DIR" 2>/dev/null
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
nft flush ruleset
nft add table inet filter
# Braces are nft syntax, so quote them rather than escaping the semicolons --
# same result, and the shell is not left guessing what is a block.
nft add chain inet filter input   '{ type filter hook input   priority 0; policy drop; }'
nft add chain inet filter forward '{ type filter hook forward priority 0; policy drop; }'
nft add chain inet filter output  '{ type filter hook output  priority 0; policy drop; }' 

logger -t kaia-lockdown "Lockdown ruleset applied: input/forward/output = drop"
echo -e "${COLOR_RED}EMERGENCY LOCKDOWN ACTIVE — all traffic dropped.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}Restore with: sudo ${0%/*}/kaia-unlock.sh${COLOR_RESET}"
