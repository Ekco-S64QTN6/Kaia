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
set -euo pipefail

COLOR_RED="\033[91m"
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_CYAN="\033[96m"
COLOR_RESET="\033[0m"

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: unlock must be executed as root.${COLOR_RESET}" >&2
    exit 1
fi

# Do not source a root-run recovery configuration as shell code. Parse only the
# single expected path assignment, and refuse an untrusted configuration file.
STATE_DIR="/var/lib/kaia"
STATE_CONFIG="/etc/kaia/state.env"
if [ -e "$STATE_CONFIG" ] || [ -L "$STATE_CONFIG" ]; then
    if [ -L "$STATE_CONFIG" ] || [ ! -f "$STATE_CONFIG" ]; then
        echo "ERROR: refusing non-regular installed state configuration: $STATE_CONFIG" >&2
        exit 1
    fi
    config_owner="$(stat -c '%u' -- "$STATE_CONFIG")"
    config_mode="$(stat -c '%a' -- "$STATE_CONFIG")"
    config_links="$(stat -c '%h' -- "$STATE_CONFIG")"
    if [ "$config_owner" != "0" ] || (( (8#$config_mode & 0077) != 0 )) || [ "$config_links" != "1" ]; then
        echo "ERROR: installed state configuration must be root-owned, private and single-link." >&2
        exit 1
    fi
    state_line="$(grep -E '^KAIA_STATE_DIR=' "$STATE_CONFIG" || true)"
    if [ "$(printf '%s\n' "$state_line" | grep -c '^KAIA_STATE_DIR=/' || true)" -ne 1 ] || \
       [ "$(grep -c '^KAIA_STATE_DIR=' "$STATE_CONFIG" || true)" -ne 1 ]; then
        echo "ERROR: installed state configuration must contain exactly one absolute KAIA_STATE_DIR." >&2
        exit 1
    fi
    state_value="${state_line#*=}"
    if [[ ! "$state_value" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
        echo "ERROR: installed KAIA_STATE_DIR contains unsupported characters." >&2
        exit 1
    fi
    STATE_DIR="$state_value"
fi

state_dir_real="$(realpath -e -- "$STATE_DIR")" || {
    echo "ERROR: configured state directory does not exist." >&2
    exit 1
}
state_input="${STATE_DIR%/}"
if [ "$state_dir_real" != "$state_input" ]; then
    echo "ERROR: state directory must be canonical and must not use symlinks." >&2
    exit 1
fi
STATE_DIR="$state_dir_real"
case "$STATE_DIR" in
    /var/lib/*|/srv/*|/mnt/*) ;;
    *) echo "ERROR: state directory must be below /var/lib, /srv, or /mnt." >&2; exit 1 ;;
esac
if [ ! -d "$STATE_DIR" ]; then
    echo "ERROR: configured state path is not a directory." >&2
    exit 1
fi

# Prevent unprivileged users from replacing any directory in the trusted path.
current="$STATE_DIR"
while :; do
    current_owner="$(stat -c '%u' -- "$current")"
    current_mode="$(stat -c '%a' -- "$current")"
    if [ "$current_owner" != "0" ] || (( (8#$current_mode & 0022) != 0 )); then
        echo "ERROR: state path component must be root-owned and not group/world-writable: $current" >&2
        exit 1
    fi
    [ "$current" = "/" ] && break
    current="$(dirname -- "$current")"
done

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

echo -e "${COLOR_CYAN}═══════════════════════════════════════${COLOR_RESET}"
echo -e "${COLOR_CYAN}  Kaia lockdown release${COLOR_RESET}"
echo -e "${COLOR_CYAN}═══════════════════════════════════════${COLOR_RESET}"
echo

# --- Validate restore point before any destructive firewall operation ---------
if ! command -v nft >/dev/null 2>&1; then
    echo "ERROR: nft command is unavailable; refusing to modify the ruleset." >&2
    exit 1
fi

UFW_ACTIVE=0
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi '^Status: active'; then
    UFW_ACTIVE=1
fi

SNAPSHOT_READY=0
if [ -L "$SNAPSHOT" ]; then
    echo "ERROR: refusing a symlinked pre-lockdown snapshot: $SNAPSHOT" >&2
    exit 1
elif [ -e "$SNAPSHOT" ]; then
    if [ ! -f "$SNAPSHOT" ]; then
        echo "ERROR: pre-lockdown snapshot is not a regular file: $SNAPSHOT" >&2
        exit 1
    fi
    snapshot_owner="$(stat -c '%u' -- "$SNAPSHOT")"
    snapshot_mode="$(stat -c '%a' -- "$SNAPSHOT")"
    snapshot_links="$(stat -c '%h' -- "$SNAPSHOT")"
    if [ "$snapshot_owner" != "0" ] || (( (8#$snapshot_mode & 0077) != 0 )) || [ "$snapshot_links" != "1" ]; then
        echo "ERROR: pre-lockdown snapshot must be root-owned, private and single-link." >&2
        exit 1
    fi
    if [ ! -s "$SNAPSHOT" ] || ! nft -c -f "$SNAPSHOT" >/dev/null 2>&1; then
        echo "ERROR: pre-lockdown snapshot is empty or fails nftables validation." >&2
        exit 1
    fi
    SNAPSHOT_READY=1
fi

if [ "$FROM_SNAPSHOT" -eq 1 ] && [ "$SNAPSHOT_READY" -ne 1 ]; then
    echo "ERROR: snapshot restore was requested, but there is no validated snapshot." >&2
    exit 1
fi
if [ "$UFW_ACTIVE" -ne 1 ] && [ "$SNAPSHOT_READY" -ne 1 ]; then
    echo "ERROR: no active UFW manager or validated snapshot is available; firewall unchanged." >&2
    exit 1
fi

# --- Show what we are about to undo -----------------------------------------
if nft list table inet filter >/dev/null 2>&1; then
    echo -e "  Lockdown table ${COLOR_YELLOW}inet filter${COLOR_RESET} is present."
else
    echo -e "  ${COLOR_YELLOW}No lockdown table found — the lockdown may already be lifted.${COLOR_RESET}"
fi
if [ "$SNAPSHOT_READY" -eq 1 ]; then
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
    if [ "$SNAPSHOT_READY" -eq 1 ]; then
        if nft -f "$SNAPSHOT" 2>/dev/null; then
            restored="raw snapshot"
        else
            echo -e "${COLOR_RED}Snapshot restore failed.${COLOR_RESET}"
        fi
    else
        echo -e "${COLOR_RED}No snapshot to restore from.${COLOR_RESET}"
    fi
elif [ "$UFW_ACTIVE" -eq 1 ]; then
    # Preferred path when ufw manages the firewall: let ufw regenerate its own
    # ruleset authoritatively rather than replaying a captured snapshot, which
    # would leave ufw's userspace state and the kernel ruleset out of sync.
    if ufw reload >/dev/null 2>&1; then
        restored="ufw reload"
    fi
fi

if [ -z "$restored" ] && [ "$SNAPSHOT_READY" -eq 1 ]; then
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
