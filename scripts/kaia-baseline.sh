#!/bin/bash
# ============================================================
# RECORD TRUSTED INTEGRITY BASELINE
#
# Writes SHA-256 hashes of every tamper-protected file to
# <KAIA_STATE_DIR>/baseline.sha256 (root-owned, 0600; defaults to /var/lib/kaia).
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
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

# The installed checkout intentionally has no .env. Prefer the root-owned,
# installer-generated system secret; in a developer checkout, safely parse only
# the token assignment from .env. Never source a project-controlled env file as
# root: it is executable shell code.
_read_secret_assignment() {
    local file="$1" line value
    [ -r "$file" ] || return 1
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            "export KAIA_CAPABILITY_TOKEN_SECRET="*) value="${line#export KAIA_CAPABILITY_TOKEN_SECRET=}" ;;
            "KAIA_CAPABILITY_TOKEN_SECRET="*) value="${line#KAIA_CAPABILITY_TOKEN_SECRET=}" ;;
            *) continue ;;
        esac
        value="${value%

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

# config.py exits if the capability-token secret is unset. It was loaded above
# from a single parsed assignment; arbitrary shell code from .env is never run.

# Ask the detector itself which files it protects, so this script and the
# runtime watchlist cannot drift apart.
mapfile -t FILES < <(
    cd "$PROJECT_DIR" && PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/core" "$PY" - <<'PYEOF'
from security.tamper_detection import TamperDetector
import os
for f in sorted(TamperDetector().immutable_files):
    if os.path.exists(f):
        print(f)
PYEOF
)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo -e "${RED}Could not enumerate protected files. Is the project importable?${NC}"
    exit 1
fi

echo "  Hashing ${#FILES[@]} protected files..."
mkdir -p "$STATE_DIR" && chmod 700 "$STATE_DIR"

tmp="$(mktemp "${STATE_DIR}/.baseline.XXXXXX")" || exit 1
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

mv "$tmp" "$BASELINE"
chmod 600 "$BASELINE"
chown root:root "$BASELINE" 2>/dev/null || true

echo
echo -e "  ${GREEN}Baseline written: ${BASELINE}${NC}"
echo "  entries: $(grep -vc '^#' "$BASELINE")"
echo
echo -e "  ${YELLOW}Restart the Policy Gate to adopt it:${NC}"
echo "      systemctl restart kaia-policy-gate"
\\r'}"
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        if [ -n "$value" ] && [[ "$value" != *

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

# config.py exits if KAIA_CAPABILITY_TOKEN_SECRET is unset, so load .env the
# same way the daemon does. Never echo it.
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "${PROJECT_DIR}/.env"
    set +a
fi

# Ask the detector itself which files it protects, so this script and the
# runtime watchlist cannot drift apart.
mapfile -t FILES < <(
    cd "$PROJECT_DIR" && PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/core" "$PY" - <<'PYEOF'
from security.tamper_detection import TamperDetector
import os
for f in sorted(TamperDetector().immutable_files):
    if os.path.exists(f):
        print(f)
PYEOF
)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo -e "${RED}Could not enumerate protected files. Is the project importable?${NC}"
    exit 1
fi

echo "  Hashing ${#FILES[@]} protected files..."
mkdir -p "$STATE_DIR" && chmod 700 "$STATE_DIR"

tmp="$(mktemp "${STATE_DIR}/.baseline.XXXXXX")" || exit 1
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

mv "$tmp" "$BASELINE"
chmod 600 "$BASELINE"
chown root:root "$BASELINE" 2>/dev/null || true

echo
echo -e "  ${GREEN}Baseline written: ${BASELINE}${NC}"
echo "  entries: $(grep -vc '^#' "$BASELINE")"
echo
echo -e "  ${YELLOW}Restart the Policy Gate to adopt it:${NC}"
echo "      systemctl restart kaia-policy-gate"
\\r'* ]]; then
            printf '%s' "$value"
            return 0
        fi
    done < "$file"
    return 1
}

if [ -z "${KAIA_CAPABILITY_TOKEN_SECRET:-}" ]; then
    secret=""
    if [ -r /etc/kaia/secret.env ]; then
        secret="$(_read_secret_assignment /etc/kaia/secret.env)" || secret=""
    fi
    if [ -z "$secret" ] && [ -r "${PROJECT_DIR}/.env" ]; then
        secret="$(_read_secret_assignment "${PROJECT_DIR}/.env")" || secret=""
    fi
    if [ -z "$secret" ]; then
        echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET not found in /etc/kaia/secret.env or project .env." >&2
        exit 1
    fi
    export KAIA_CAPABILITY_TOKEN_SECRET="$secret"
fi

# The installer validates this path and writes it without shell evaluation.
# Parse the value instead of sourcing the file as root.
if [ -z "${KAIA_STATE_DIR:-}" ] && [ -r /etc/kaia/state.env ]; then
    state_line="$(grep -E '^KAIA_STATE_DIR=/' /etc/kaia/state.env | head -n 1 || true)"
    state_value="${state_line#KAIA_STATE_DIR=}"
    if [[ "$state_value" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
        export KAIA_STATE_DIR="$state_value"
    fi
fi

STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"
BASELINE="${STATE_DIR}/baseline.sha256"

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

# config.py exits if KAIA_CAPABILITY_TOKEN_SECRET is unset, so load .env the
# same way the daemon does. Never echo it.
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "${PROJECT_DIR}/.env"
    set +a
fi

# Ask the detector itself which files it protects, so this script and the
# runtime watchlist cannot drift apart.
mapfile -t FILES < <(
    cd "$PROJECT_DIR" && PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/core" "$PY" - <<'PYEOF'
from security.tamper_detection import TamperDetector
import os
for f in sorted(TamperDetector().immutable_files):
    if os.path.exists(f):
        print(f)
PYEOF
)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo -e "${RED}Could not enumerate protected files. Is the project importable?${NC}"
    exit 1
fi

echo "  Hashing ${#FILES[@]} protected files..."
mkdir -p "$STATE_DIR" && chmod 700 "$STATE_DIR"

tmp="$(mktemp "${STATE_DIR}/.baseline.XXXXXX")" || exit 1
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

mv "$tmp" "$BASELINE"
chmod 600 "$BASELINE"
chown root:root "$BASELINE" 2>/dev/null || true

echo
echo -e "  ${GREEN}Baseline written: ${BASELINE}${NC}"
echo "  entries: $(grep -vc '^#' "$BASELINE")"
echo
echo -e "  ${YELLOW}Restart the Policy Gate to adopt it:${NC}"
echo "      systemctl restart kaia-policy-gate"
