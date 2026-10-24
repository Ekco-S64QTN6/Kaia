#!/bin/bash
# ============================================================
# RECORD TRUSTED INTEGRITY BASELINE
#
# Writes SHA-256 hashes of every tamper-protected file to
# ${KAIA_STATE_DIR}/baseline.sha256 (root-owned, 0600; default /var/lib/kaia).
#
# WHY: tamper detection otherwise hashes whatever is on disk when it
# starts, so a file edited while the daemon was stopped simply becomes
# the new baseline. Anyone who can write a file can also restart a
# service, which made that a complete bypass. This records the state
# you have decided to trust, so startup can detect drift from it.
#
# RUN THIS ONLY FROM A STATE YOU TRUST -- after reviewing your changes,
# not blindly after every edit. Baselining a compromise makes the
# compromise the reference.
#
#     sudo ./scripts/kaia-baseline.sh
#
# Exit: 0 written, 1 error
# ============================================================
set -euo pipefail

# Never leave a partial trust record behind if enumeration, hashing or writing fails.
tmp=""
files_tmp=""
cleanup() {
    if [ -n "$tmp" ] && [ -e "$tmp" ]; then rm -f -- "$tmp"; fi
    if [ -n "$files_tmp" ] && [ -e "$files_tmp" ]; then rm -f -- "$files_tmp"; fi
}
trap cleanup EXIT

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

# Parse only the intended key/value assignments. Never source a project-controlled
# .env file as root; sourcing it would execute arbitrary shell code.
_read_env_assignment() {
    local file="$1" key="$2" line value
    [ -r "$file" ] || return 1
    line="$(grep -E "^(export )?${key}=" "$file" | head -n 1 || true)"
    [ -n "$line" ] || return 1
    line="$(printf "%s" "$line" | sed "s/^export //")"
    value="$(printf "%s" "$line" | cut -d= -f2-)"
    case "$value" in
        \"*\") value="$(printf "%s" "$value" | sed "s/^\\\"//;s/\\\"$//")" ;;
        \'*\') value="$(printf "%s" "$value" | sed "s/^\\\x27//;s/\\\x27$//")" ;;
    esac
    case "$value" in *[[:cntrl:]]*) return 1 ;; esac
    [ -n "$value" ] || return 1
    printf "%s" "$value"
}

# System-mode installs remove /opt/kaia/.env. Prefer the root-owned installer
# secret; the checkout .env is only a development fallback.
if [ -z "${KAIA_CAPABILITY_TOKEN_SECRET:-}" ]; then
    secret="$(_read_env_assignment /etc/kaia/secret.env KAIA_CAPABILITY_TOKEN_SECRET || true)"
    if [ -z "$secret" ]; then
        secret="$(_read_env_assignment "$PROJECT_DIR/.env" KAIA_CAPABILITY_TOKEN_SECRET || true)"
    fi
    if [ -z "$secret" ]; then
        echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET missing from /etc/kaia/secret.env and project .env." >&2
        exit 1
    fi
    export KAIA_CAPABILITY_TOKEN_SECRET="$secret"
fi

# The installer state file is authoritative for privileged manual runs. Parse
# only its single path assignment; never source this root-run configuration.
STATE_CONFIG="/etc/kaia/state.env"
if [ -e "$STATE_CONFIG" ] || [ -L "$STATE_CONFIG" ]; then
    if [ -L "$STATE_CONFIG" ] || [ ! -f "$STATE_CONFIG" ]; then
        echo "ERROR: refusing non-regular installed state configuration: $STATE_CONFIG" >&2
        exit 1
    fi
    config_owner="$(stat -c '%u' -- "$STATE_CONFIG")"
    config_mode="$(stat -c '%a' -- "$STATE_CONFIG")"
    config_links="$(stat -c '%h' -- "$STATE_CONFIG")"
    state_line="$(grep -E '^KAIA_STATE_DIR=' "$STATE_CONFIG" || true)"
    if [ "$config_owner" != "0" ] || [[ ! "$config_mode" =~ ^[0-7]+$ ]] || \
       (( (8#$config_mode & 0077) != 0 )) || [ "$config_links" != "1" ] || \
       [ "$(grep -c '^KAIA_STATE_DIR=' "$STATE_CONFIG" || true)" -ne 1 ] || \
       [[ ! "$state_line" =~ ^KAIA_STATE_DIR=(/[A-Za-z0-9_./-]+)$ ]]; then
        echo "ERROR: installed state configuration must be a root-owned private file with one absolute KAIA_STATE_DIR." >&2
        exit 1
    fi
    KAIA_STATE_DIR="${state_line#*=}"
fi
KAIA_STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"
STATE_DIR="$(realpath -m -- "$KAIA_STATE_DIR")"
state_input="${KAIA_STATE_DIR%/}"
if [ "$STATE_DIR" != "$state_input" ]; then
    echo "ERROR: KAIA_STATE_DIR must be canonical and must not contain symlink/parent-traversal components." >&2
    exit 1
fi
case "$STATE_DIR" in
    /var/lib/*|/srv/*|/mnt/*) ;;
    *) echo "ERROR: KAIA_STATE_DIR must be below /var/lib, /srv, or /mnt." >&2; exit 1 ;;
esac

# The trusted baseline must not be placed in or above the tree it authenticates.
case "$PROJECT_DIR/" in
    "$STATE_DIR/"*) echo "ERROR: KAIA_STATE_DIR must not contain the project checkout." >&2; exit 1 ;;
esac
case "$STATE_DIR/" in
    "$PROJECT_DIR/"*) echo "ERROR: KAIA_STATE_DIR must not be inside the project checkout." >&2; exit 1 ;;
esac

validate_state_path_components() {
    local current="$STATE_DIR" owner mode
    while :; do
        if [ -L "$current" ]; then
            echo "ERROR: state path contains a symlink component: $current" >&2
            return 1
        fi
        if [ -e "$current" ]; then
            if [ ! -d "$current" ]; then
                echo "ERROR: state path component is not a directory: $current" >&2
                return 1
            fi
            owner="$(stat -c '%u' -- "$current")" || return 1
            mode="$(stat -c '%a' -- "$current")" || return 1
            if [ "$owner" != "0" ] || [[ ! "$mode" =~ ^[0-7]+$ ]] || (( (8#$mode & 0022) != 0 )); then
                echo "ERROR: state path component must be root-owned and not group/world-writable: $current" >&2
                return 1
            fi
        fi
        [ "$current" = "/" ] && break
        current="$(dirname -- "$current")"
    done
}
if ! validate_state_path_components; then
    exit 1
fi

# Keep subprocess configuration aligned with the canonical trusted path.
export KAIA_STATE_DIR="$STATE_DIR"

BASELINE="${STATE_DIR}/baseline.sha256"
if [ -L "$BASELINE" ]; then
    echo "ERROR: refusing a symlinked trusted baseline: $BASELINE" >&2
    exit 1
elif [ -e "$BASELINE" ]; then
    if [ ! -f "$BASELINE" ]; then
        echo "ERROR: trusted baseline is not a regular file: $BASELINE" >&2
        exit 1
    fi
    baseline_owner="$(stat -c '%u' -- "$BASELINE")"
    baseline_mode="$(stat -c '%a' -- "$BASELINE")"
    baseline_links="$(stat -c '%h' -- "$BASELINE")"
    if [ "$baseline_owner" != "0" ] || [[ ! "$baseline_mode" =~ ^[0-7]+$ ]] || (( (8#$baseline_mode & 0077) != 0 )) || [ "$baseline_links" != "1" ]; then
        echo "ERROR: trusted baseline must be root-owned, private and single-link." >&2
        exit 1
    fi
fi

echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${CYAN}  Kaia trusted integrity baseline${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}\n"

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Must run as root (writes ${BASELINE}):  sudo $0${NC}"
    exit 1
fi

PY="${PROJECT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
[ -x "$PY" ] || { echo -e "${RED}No usable python interpreter.${NC}"; exit 1; }

# config.py now sees the validated capability token loaded above. Do not print it.

# Ask the detector itself which files it protects, so this script and the
# runtime watchlist cannot drift apart.
files_tmp="$(mktemp)"
if ! (
    cd "$PROJECT_DIR" && PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/core" "$PY" - <<'PYEOF'
from security.tamper_detection import TamperDetector
import os
import stat

# The installed checkout deliberately has no .env. That one absence is valid;
# every other listed protected path must exist as a regular, non-symlink file.
for f in sorted(TamperDetector().immutable_files):
    try:
        metadata = os.lstat(f)
    except FileNotFoundError:
        if os.path.basename(f) == ".env":
            continue
        raise RuntimeError(f"Missing protected file in integrity watchlist: {f}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Protected watchlist path is not a regular file: {f}")
    print(os.path.abspath(f))
PYEOF
) > "$files_tmp"; then
    echo -e "${RED}Could not enumerate protected files. Refusing to write a new baseline.${NC}" >&2
    exit 1
fi
mapfile -t FILES < "$files_tmp"
rm -f -- "$files_tmp"
files_tmp=""

if [ "${#FILES[@]}" -eq 0 ]; then
    echo -e "${RED}Could not enumerate protected files. Is the project importable?${NC}"
    exit 1
fi

echo "  Hashing ${#FILES[@]} protected files..."
mkdir -p "$STATE_DIR"
# Keep the installer contract: group-readable operator data under root:kaiacord 0750.
# The baseline file itself is root:root 0600; the directory need not be 0700.
if getent group kaiacord >/dev/null; then
    chown root:kaiacord "$STATE_DIR"
    chmod 750 "$STATE_DIR"
else
    echo -e "${RED}Required group 'kaiacord' is missing; run install_services.sh first.${NC}" >&2
    exit 1
fi

tmp="$(mktemp "${STATE_DIR}/.baseline.XXXXXX")"
{
    echo "# Kaia trusted integrity baseline"
    echo "# Generated $(date -Iseconds) by $(logname 2>/dev/null || echo "uid=$EUID")"
    echo "# Regenerate after reviewed changes:  sudo ./scripts/kaia-baseline.sh"
    for f in "${FILES[@]}"; do
        sha256sum "$f"
    done
} > "$tmp"

# Show what changed against any previous baseline before replacing it.
if [ -f "$BASELINE" ]; then
    changed=$(diff <(grep -v '^#' "$BASELINE" | sort) <(grep -v '^#' "$tmp" | sort) | grep -c '^[<>]' || true)
    if [ "$changed" -gt 0 ]; then
        echo -e "  ${YELLOW}${changed} line(s) differ from the previous baseline:${NC}"
        diff <(grep -v '^#' "$BASELINE" | sort) <(grep -v '^#' "$tmp" | sort) \
            | grep '^>' | awk '{print "    changed: "$3}' | head -20
    else
        echo "  No changes since the previous baseline."
    fi
fi

chmod 600 "$tmp"
chown root:root "$tmp"
mv -f -- "$tmp" "$BASELINE"
tmp=""

echo
echo -e "  ${GREEN}Baseline written: ${BASELINE}${NC}"
echo "  entries: $(grep -vc '^#' "$BASELINE")"
echo
echo -e "  ${YELLOW}Restart the Policy Gate to adopt it:${NC}"
echo "      systemctl restart kaia-policy-gate"
