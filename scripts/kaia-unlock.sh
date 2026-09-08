#!/bin/bash
# ============================================================
# KAIA LOCKDOWN RELEASE
# Lifts the emergency network lockdown applied by kaia-lockdown.sh
# and restores normal firewalling.
#
# This re-opens the network after a compromise assertion, so it is
# deliberately manual and requires explicit confirmation. Nothing
# calls this automatically.
#
# Exit: 0 restored, 1 error/aborted
# ============================================================
set -uo pipefail

COLOR_RED="\033[91m"
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_CYAN="\033[96m"
COLOR_RESET="\033[0m"

STATE_DIR="/var/lib/kaia"
SNAPSHOT="${STATE_DIR}/pre-lockdown.nft"
FORCE=0
FROM_SNAPSHOT=0

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes)           FORCE=1 ;;
        -s|--from-snapshot) FROM_SNAPSHOT=1 ;;
        -h|--help)
            echo "Usage: kaia-unlock.sh [-y] [-s]"
            echo "  -y, --yes            Skip the confirmation prompt"
            echo "  -s, --from-snapshot  Restore the raw pre-lockdown ruleset instead of"
            echo "                       regenerating from ufw (use if ufw is not managing"
            echo "                       this host's firewall)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: unlock must be executed as root.${COLOR_RESET}"
    exit 1
fi

echo -e "${COLOR_CYAN}═══════════════════════════════════════${COLOR_RESET}"
echo -e "${COLOR_CYAN}  Kaia lockdown release${COLOR_RESET}"
echo -e "${COLOR_CYAN}═══════════════════════════════════════${COLOR_RESET}"
echo

# --- Show what we are about to undo -----------------------------------------
if nft list table inet filter >/dev/null 2>&1; then
    echo -e "  Lockdown table ${COLOR_YELLOW}inet filter${COLOR_RESET} is present."
else
    echo -e "  ${COLOR_YELLOW}No lockdown table found — the lockdown may already be lifted.${COLOR_RESET}"
fi
if [ -f "$SNAPSHOT" ]; then
    echo "  Pre-lockdown snapshot: $SNAPSHOT ($(date -r "$SNAPSHOT" '+%Y-%m-%d %H:%M:%S'))"
else
    echo -e "  ${COLOR_YELLOW}No snapshot found at $SNAPSHOT${COLOR_RESET}"
fi
echo

# --- Confirm ----------------------------------------------------------------
# Releasing a lockdown re-exposes a host that something asserted was compromised.
# Make that an explicit decision.
if [ "$FORCE" -ne 1 ]; then
    echo -e "${COLOR_YELLOW}This re-opens network access on a host that was locked down after a${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}compromise assertion. Confirm the cause has been investigated.${COLOR_RESET}"
    read -rp "Type RESTORE to continue: " reply
    if [ "$reply" != "RESTORE" ]; then
        echo "Aborted. Lockdown remains in effect."
        exit 1
    fi
fi

logger -t kaia-unlock "Lockdown release initiated by uid=$EUID"

# --- Restore ----------------------------------------------------------------
nft flush ruleset

restored=""
if [ "$FROM_SNAPSHOT" -eq 1 ]; then
    if [ -f "$SNAPSHOT" ]; then
        if nft -f "$SNAPSHOT" 2>/dev/null; then
            restored="raw snapshot"
        else
            echo -e "${COLOR_RED}Snapshot restore failed.${COLOR_RESET}"
        fi
    else
        echo -e "${COLOR_RED}No snapshot to restore from.${COLOR_RESET}"
    fi
elif command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi '^Status: active'; then
    # Preferred path when ufw manages the firewall: let ufw regenerate its own
    # ruleset authoritatively rather than replaying a captured snapshot, which
    # would leave ufw's userspace state and the kernel ruleset out of sync.
    if ufw reload >/dev/null 2>&1; then
        restored="ufw reload"
    fi
fi

if [ -z "$restored" ] && [ -f "$SNAPSHOT" ]; then
    nft -f "$SNAPSHOT" 2>/dev/null && restored="raw snapshot (fallback)"
fi

echo
if [ -n "$restored" ]; then
    echo -e "  ${COLOR_GREEN}Firewall restored via: $restored${COLOR_RESET}"
    logger -t kaia-unlock "Lockdown lifted; firewall restored via $restored"
else
    echo -e "  ${COLOR_RED}Ruleset was flushed but NOT restored.${COLOR_RESET}"
    echo -e "  ${COLOR_RED}The host is now UNFIREWALLED. Run 'ufw reload' or reapply rules now.${COLOR_RESET}"
    logger -t kaia-unlock "WARNING: lockdown flushed but no ruleset restored"
    exit 1
fi

echo
echo "  Current tables:"
nft list tables 2>/dev/null | sed 's/^/    /'
echo
echo -e "  ${COLOR_YELLOW}Reminder: the tamper baseline is re-established when the policy gate${COLOR_RESET}"
echo -e "  ${COLOR_YELLOW}restarts. If files were legitimately changed, restart it now:${COLOR_RESET}"
echo "      systemctl restart kaia-policy-gate"
