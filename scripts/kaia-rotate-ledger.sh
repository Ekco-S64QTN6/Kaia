#!/bin/bash
# ============================================================
# ROTATE THE AUDIT LEDGER
#
# INV-003 makes the ledger append-only and gives the agent zero write,
# update or delete capability over it, so rotation is deliberately an
# OPERATOR action and is not performed by the daemon. Tamper detection
# also prefix-hashes the ledger, so rotation must be followed by a baseline
# update while the Policy Gate remains stopped.
#
#     sudo systemctl stop kaia-policy-gate
#     sudo ./scripts/kaia-rotate-ledger.sh
#     sudo ./scripts/kaia-baseline.sh
#     sudo systemctl start kaia-policy-gate
#
# Exit: 0 rotated, 1 error
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; NC='\033[0m'

[ "$EUID" -eq 0 ] || { echo -e "\${RED}Must run as root: sudo $0\${NC}" >&2; exit 1; }

# This root-run utility does not trust inherited paths. The installed config
# is authoritative; before installation, use the fixed default state directory.
STATE_DIR="/var/lib/kaia"
STATE_CONFIG="/etc/kaia/state.env"
if [ -e "$STATE_CONFIG" ] || [ -L "$STATE_CONFIG" ]; then
    if [ -L "$STATE_CONFIG" ] || [ ! -f "$STATE_CONFIG" ]; then
        echo "ERROR: refusing non-regular installed state configuration: $STATE_CONFIG" >&2
        exit 1
    fi
    state_config_real="$(realpath -e -- "$STATE_CONFIG")"
    if [ "$state_config_real" != "$STATE_CONFIG" ]; then
        echo "ERROR: installed state configuration path must not use symlinks." >&2
        exit 1
    fi
    state_config_owner="$(stat -c '%u' -- "$STATE_CONFIG")"
    state_config_mode="$(stat -c '%a' -- "$STATE_CONFIG")"
    if [ "$state_config_owner" != "0" ] || (( (8#$state_config_mode & 0022) != 0 )); then
        echo "ERROR: installed state configuration must be root-owned and not group/world-writable." >&2
        exit 1
    fi
    state_line="$(grep -E '^KAIA_STATE_DIR=/' "$STATE_CONFIG" | head -n 1 || true)"
    if [ -z "$state_line" ]; then
        echo "ERROR: installed state configuration has no valid absolute KAIA_STATE_DIR." >&2
        exit 1
    fi
    state_value="\${state_line#*=}"
    case "$state_value" in
        /*) ;;
        *) echo "ERROR: invalid KAIA_STATE_DIR in installed state configuration." >&2; exit 1 ;;
    esac
    case "$state_value" in
        *[!A-Za-z0-9_./-]*) echo "ERROR: invalid characters in installed KAIA_STATE_DIR." >&2; exit 1 ;;
    esac
    STATE_DIR="$state_value"
fi

state_dir_real="$(realpath -e -- "$STATE_DIR")"
state_input="\${STATE_DIR%/}"
if [ "$state_dir_real" != "$state_input" ]; then
    echo "ERROR: state directory path must be canonical and must not use symlinks." >&2
    exit 1
fi
STATE_DIR="$state_dir_real"
case "$STATE_DIR" in
    /var/lib/*|/srv/*|/mnt/*) ;;
    *) echo "ERROR: KAIA_STATE_DIR must be below /var/lib, /srv, or /mnt." >&2; exit 1 ;;
esac
[ -d "$STATE_DIR" ] || { echo "ERROR: state path is not a directory." >&2; exit 1; }

# Every component must be root-owned and non-writable by group/other; otherwise
# an unprivileged process might replace a parent or redirect the ledger.
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

LEDGER="\${STATE_DIR}/audit_ledger.json"
KEEP="\${KAIA_LEDGER_KEEP:-6}"
if [[ ! "$KEEP" =~ ^[0-9]{1,3}$ ]]; then
    echo "ERROR: KAIA_LEDGER_KEEP must be an integer from 1 to 100." >&2
    exit 1
fi
KEEP_COUNT=$((10#$KEEP))
if (( KEEP_COUNT < 1 || KEEP_COUNT > 100 )); then
    echo "ERROR: KAIA_LEDGER_KEEP must be an integer from 1 to 100." >&2
    exit 1
fi

new_ledger_tmp=""
archives_tmp=""
cleanup() {
    if [ -n "$new_ledger_tmp" ] && [ -e "$new_ledger_tmp" ]; then rm -f -- "$new_ledger_tmp"; fi
    if [ -n "$archives_tmp" ] && [ -e "$archives_tmp" ]; then rm -f -- "$archives_tmp"; fi
}
trap cleanup EXIT

echo -e "\${CYAN}═══════════════════════════════════════\${NC}"
echo -e "\${CYAN}  Kaia audit ledger rotation\${NC}"
echo -e "\${CYAN}═══════════════════════════════════════\${NC}\\n"

if systemctl is-active --quiet kaia-policy-gate 2>/dev/null; then
    echo -e "\${RED}The Policy Gate is running.\${NC}"
    echo "Stop it first, or rotation will race the writer and trip tamper detection:"
    echo "    sudo systemctl stop kaia-policy-gate"
    exit 1
fi

if [ -L "$LEDGER" ]; then
    echo "ERROR: refusing to rotate a symlinked audit ledger: $LEDGER" >&2
    exit 1
fi
if [ ! -e "$LEDGER" ]; then
    echo -e "\${YELLOW}No ledger at $LEDGER — nothing to rotate.\${NC}"
    exit 0
fi
if [ ! -f "$LEDGER" ]; then
    echo "ERROR: audit ledger is not a regular file: $LEDGER" >&2
    exit 1
fi
ledger_owner="$(stat -c '%u' -- "$LEDGER")"
ledger_mode="$(stat -c '%a' -- "$LEDGER")"
ledger_links="$(stat -c '%h' -- "$LEDGER")"
if [ "$ledger_owner" != "0" ] || (( (8#$ledger_mode & 0022) != 0 )) || [ "$ledger_links" != "1" ]; then
    echo "ERROR: audit ledger must be root-owned, non-writable by group/other, and a single-link regular file." >&2
    exit 1
fi

size="$(stat -c%s -- "$LEDGER")"
lines="$(grep -c . "$LEDGER" 2>/dev/null || echo 0)"
echo "  current: $LEDGER"
echo "           $(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes"), $lines entries"

stamp="$(date +%Y%m%d-%H%M%S%N)"
archive="\${LEDGER}.\${stamp}"
if [ -e "$archive" ] || [ -L "$archive" ]; then
    echo "ERROR: archive destination already exists: $archive" >&2
    exit 1
fi

# Prepare a replacement first, then rename both files on the same filesystem.
# If installing the replacement fails, restore the original archive where possible.
new_ledger_tmp="$(mktemp "\${STATE_DIR}/.audit_ledger.json.XXXXXX")"
chown --reference="$LEDGER" -- "$new_ledger_tmp"
chmod 640 -- "$new_ledger_tmp"
if ! mv -- "$LEDGER" "$archive"; then
    echo "ERROR: could not archive the current ledger; no rotation completed." >&2
    exit 1
fi
if ! mv -- "$new_ledger_tmp" "$LEDGER"; then
    if mv -- "$archive" "$LEDGER"; then
        echo "ERROR: could not install new ledger; original ledger restored." >&2
    else
        echo "ERROR: could not install new ledger or restore old ledger; preserve archive at $archive." >&2
    fi
    exit 1
fi
new_ledger_tmp=""

# Compression is best effort so a gzip failure preserves the original archive.
if gzip -9 -- "$archive" 2>/dev/null; then
    archive="\${archive}.gz"
else
    echo "WARNING: archive compression failed; preserving $archive uncompressed." >&2
fi
chmod 600 -- "$archive"

echo -e "  \${GREEN}archived: $archive\${NC}"
echo -e "  \${GREEN}new empty ledger created\${NC}"

# A checked file list avoids process-substitution masking errors from find/sort.
archives_tmp="$(mktemp "\${STATE_DIR}/.ledger-archives.XXXXXX")"
if ! find "$STATE_DIR" -maxdepth 1 -type f -name 'audit_ledger.json.*' -printf '%T@ %p\\n' | sort -n > "$archives_tmp"; then
    echo "ERROR: rotation completed, but archive enumeration failed; manual pruning is required." >&2
    exit 1
fi
mapfile -t archives < "$archives_tmp"
rm -f -- "$archives_tmp"
archives_tmp=""
remove_count=$((\${#archives[@]} - KEEP_COUNT))
if (( remove_count > 0 )); then
    echo "  pruning $remove_count archive(s) beyond the newest $KEEP_COUNT:"
    for (( i=0; i<remove_count; i++ )); do
        line="\${archives[$i]}"
        f="\${line#* }"
        if [ -L "$f" ] || [ ! -f "$f" ] || [ "$(stat -c '%u' -- "$f")" != "0" ] || [ "$(stat -c '%h' -- "$f")" != "1" ]; then
            echo "ERROR: refusing to prune an untrusted archive path: $f" >&2
            exit 1
        fi
        rm -f -- "$f"
        echo "    removed $(basename "$f")"
    done
fi

echo
echo -e "  \${YELLOW}The ledger hash has changed. Re-baseline before restarting:\${NC}"
echo "      sudo ./scripts/kaia-baseline.sh"
echo "      sudo systemctl start kaia-policy-gate"
