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

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: Lockdown must be executed as root.${COLOR_RESET}" >&2
    exit 1
fi

# Emergency containment must not trust inherited path overrides. Only use the
# installer-generated state path after validating it as data, never shell code.
STATE_DIR="/var/lib/kaia"
STATE_CONFIG="/etc/kaia/state.env"
STATE_PATH_SAFE=1
if [ -e "$STATE_CONFIG" ] || [ -L "$STATE_CONFIG" ]; then
    if [ -L "$STATE_CONFIG" ] || [ ! -f "$STATE_CONFIG" ]; then
        STATE_PATH_SAFE=0
    else
        config_owner="$(stat -c '%u' -- "$STATE_CONFIG" 2>/dev/null || echo invalid)"
        config_mode="$(stat -c '%a' -- "$STATE_CONFIG" 2>/dev/null || echo invalid)"
        config_links="$(stat -c '%h' -- "$STATE_CONFIG" 2>/dev/null || echo invalid)"
        state_line="$(grep -E '^KAIA_STATE_DIR=' "$STATE_CONFIG" 2>/dev/null || true)"
        if [ "$config_owner" != "0" ] || [[ ! "$config_mode" =~ ^[0-7]+$ ]] || (( (8#$config_mode & 0077) != 0 )) || \
           [ "$config_links" != "1" ] || \
           [ "$(grep -c '^KAIA_STATE_DIR=' "$STATE_CONFIG" 2>/dev/null || true)" -ne 1 ] || \
           [[ ! "$state_line" =~ ^KAIA_STATE_DIR=(/[A-Za-z0-9_./-]+)$ ]]; then
            STATE_PATH_SAFE=0
        else
            STATE_DIR="${state_line#*=}"
        fi
    fi
fi

if [ "$STATE_PATH_SAFE" -eq 1 ]; then
    state_dir_real="$(realpath -e -- "$STATE_DIR" 2>/dev/null || true)"
    state_input="${STATE_DIR%/}"
    if [ -z "$state_dir_real" ] || [ "$state_dir_real" != "$state_input" ] || [ ! -d "$STATE_DIR" ]; then
        STATE_PATH_SAFE=0
    else
        STATE_DIR="$state_dir_real"
        case "$STATE_DIR" in
            /var/lib/*|/srv/*|/mnt/*) ;;
            *) STATE_PATH_SAFE=0 ;;
        esac
    fi
fi

# Validate every existing ancestor so an unprivileged process cannot redirect
# the snapshot location. Snapshot failure is non-fatal: containment must proceed.
if [ "$STATE_PATH_SAFE" -eq 1 ]; then
    current="$STATE_DIR"
    while :; do
        current_owner="$(stat -c '%u' -- "$current" 2>/dev/null || echo invalid)"
        current_mode="$(stat -c '%a' -- "$current" 2>/dev/null || echo invalid)"
        if [ "$current_owner" != "0" ] || [[ ! "$current_mode" =~ ^[0-7]+$ ]] || (( (8#$current_mode & 0022) != 0 )); then
            STATE_PATH_SAFE=0
            break
        fi
        [ "$current" = "/" ] && break
        current="$(dirname -- "$current")"
    done
fi

SNAPSHOT=""
if [ "$STATE_PATH_SAFE" -eq 1 ]; then
    SNAPSHOT="${STATE_DIR}/pre-lockdown.nft"
    # Never rename over a symlink, non-regular file or multiply-linked snapshot.
    if [ -L "$SNAPSHOT" ]; then
        STATE_PATH_SAFE=0
    elif [ -e "$SNAPSHOT" ]; then
        snapshot_owner="$(stat -c '%u' -- "$SNAPSHOT" 2>/dev/null || echo invalid)"
        snapshot_mode="$(stat -c '%a' -- "$SNAPSHOT" 2>/dev/null || echo invalid)"
        snapshot_links="$(stat -c '%h' -- "$SNAPSHOT" 2>/dev/null || echo invalid)"
        if [ ! -f "$SNAPSHOT" ] || [ "$snapshot_owner" != "0" ] || \
           [[ ! "$snapshot_mode" =~ ^[0-7]+$ ]] || (( (8#$snapshot_mode & 0077) != 0 )) || \
           [ "$snapshot_links" != "1" ]; then
            STATE_PATH_SAFE=0
        fi
    fi
fi

if [ "$STATE_PATH_SAFE" -ne 1 ]; then
    SNAPSHOT=""
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
if [ -n "$SNAPSHOT" ]; then
    snapshot_tmp="$(mktemp "$STATE_DIR/.pre-lockdown.XXXXXX" 2>/dev/null || true)"
else
    snapshot_tmp=""
fi

if [ -n "$snapshot_tmp" ] && nft list ruleset > "$snapshot_tmp" 2>/dev/null && [ -s "$snapshot_tmp" ]; then
    chmod 600 "$snapshot_tmp" 2>/dev/null || true

    # Keep a uniquely named forensic copy; an existing destination is never followed.
    history_tmp="$(mktemp "$STATE_DIR/.pre-lockdown-history.XXXXXX" 2>/dev/null || true)"
    if [ -n "$history_tmp" ]; then
        if cp -- "$snapshot_tmp" "$history_tmp" 2>/dev/null && chmod 600 "$history_tmp" 2>/dev/null; then
            history="${STATE_DIR}/pre-lockdown-$(date +%Y%m%d-%H%M%S-%N).nft"
            if [ ! -e "$history" ] && [ ! -L "$history" ] && mv -- "$history_tmp" "$history"; then
                history_tmp=""
            fi
        fi
        if [ -n "$history_tmp" ]; then rm -f -- "$history_tmp"; fi
    fi

    if mv -f -- "$snapshot_tmp" "$SNAPSHOT"; then
        snapshot_tmp=""
        chmod 600 "$SNAPSHOT" 2>/dev/null || true
        logger -t kaia-lockdown "Pre-lockdown ruleset snapshot saved to $SNAPSHOT"
    else
        echo -e "${COLOR_YELLOW}WARNING: could not install the pre-lockdown snapshot.${COLOR_RESET}" >&2
    fi
else
    if [ -n "$snapshot_tmp" ]; then rm -f -- "$snapshot_tmp"; fi
    echo -e "${COLOR_YELLOW}WARNING: could not safely snapshot the current nftables ruleset.${COLOR_RESET}" >&2
    echo -e "${COLOR_YELLOW}Proceeding with lockdown anyway -- containment takes priority.${COLOR_RESET}" >&2
    logger -t kaia-lockdown "WARNING: ruleset snapshot FAILED; locking down without a restore point"
fi
if [ -n "$snapshot_tmp" ]; then rm -f -- "$snapshot_tmp"; fi

# --- Contain ----------------------------------------------------------------
# Apply the flush + replacement chains as one nftables batch transaction.
# The table is named kaia_lockdown, not "filter": Arch's stock /etc/nftables.conf
# names its own table "inet filter", so an unlock tool that looked for or deleted
# that name could not tell the lockdown from the host's real firewall.
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
add table inet kaia_lockdown
add chain inet kaia_lockdown input { type filter hook input priority 0; policy drop; }
add chain inet kaia_lockdown forward { type filter hook forward priority 0; policy drop; }
add chain inet kaia_lockdown output { type filter hook output priority 0; policy drop; }
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
    if ! nft list chain inet kaia_lockdown "$chain" 2>/dev/null | grep -q 'policy drop'; then
        logger -t kaia-lockdown "ERROR: lockdown verification failed for chain $chain"
        echo -e "${COLOR_RED}ERROR: lockdown verification failed for chain $chain.${COLOR_RESET}" >&2
        exit 1
    fi
done

logger -t kaia-lockdown "Lockdown ruleset verified: input/forward/output = drop"
echo -e "${COLOR_RED}EMERGENCY LOCKDOWN ACTIVE — all traffic dropped.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}Restore with: sudo ${0%/*}/kaia-unlock.sh${COLOR_RESET}"
