#!/bin/bash
# Kaia Emergency Network Lockdown Script
# 
# Purpose: Instantly isolate the host network during active security breaches.
# Under lockdown: All INPUT, FORWARD, and OUTPUT packets are DROPPED.
# 
# Restoration instructions:
#   sudo nft flush ruleset
#   sudo systemctl restart nftables.service (or reload rules)

COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: Lockdown must be executed as root.${COLOR_RESET}"
    exit 1
fi

logger -t kaia-lockdown "EMERGENCY LOCKDOWN ACTIVATED"

# Flush ruleset
nft flush ruleset

# Create table and set DROP policy chains
nft add table inet filter
nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
nft add chain inet filter output { type filter hook output priority 0 \; policy drop \; }

# systemctl isolate rescue.target (Commented out by default to avoid ssh lockouts)
# systemctl isolate rescue.target
