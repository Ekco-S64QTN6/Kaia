#!/bin/bash
# ============================================================
# ROTATE THE AUDIT LEDGER
#
# INV-003 makes the ledger append-only and gives the agent zero write,
# update or delete capability over it, so rotation is deliberately an
# OPERATOR action and is not performed by the daemon. Tamper detection
# also prefix-hashes the ledger, so rotating behind its back would look
# exactly like log truncation by an intruder -- which is the point.
#
#     sudo systemctl stop kaia-policy-gate
#     sudo ./scripts/kaia-rotate-ledger.sh
#     sudo ./scripts/kaia-baseline.sh
#     sudo systemctl start kaia-policy-gate
#
# Exit: 0 rotated, 1 error
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; NC='\033[0m'

STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"
LEDGER="${STATE_DIR}/audit_ledger.json"
KEEP="${KAIA_LEDGER_KEEP:-6}"

echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${CYAN}  Kaia audit ledger rotation${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}\n"

[ "$EUID" -eq 0 ] || { echo -e "${RED}Must run as root: sudo $0${NC}"; exit 1; }

if systemctl is-active --quiet kaia-policy-gate 2>/dev/null; then
    echo -e "${RED}The Policy Gate is running.${NC}"
    echo "Stop it first, or rotation will race the writer and trip tamper detection:"
    echo "    sudo systemctl stop kaia-policy-gate"
    exit 1
fi

if [ ! -f "$LEDGER" ]; then
    echo -e "${YELLOW}No ledger at $LEDGER — nothing to rotate.${NC}"
    exit 0
fi

size=$(stat -c%s "$LEDGER")
lines=$(grep -c . "$LEDGER" 2>/dev/null || echo 0)
echo "  current: $LEDGER"
echo "           $(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes"), $lines entries"

stamp=$(date +%Y%m%d-%H%M%S)
archive="${LEDGER}.${stamp}"
mv "$LEDGER" "$archive"
gzip -9 "$archive" 2>/dev/null && archive="${archive}.gz"
chmod 600 "$archive" 2>/dev/null

# Start a fresh ledger with the same ownership/permissions the daemon expects.
: > "$LEDGER"
chown --reference="$archive" "$LEDGER" 2>/dev/null || true
chmod 640 "$LEDGER"

echo -e "  ${GREEN}archived: $archive${NC}"
echo -e "  ${GREEN}new empty ledger created${NC}"

# Prune old archives beyond KEEP, oldest first.
mapfile -t old < <(find "$STATE_DIR" -maxdepth 1 -name 'audit_ledger.json.*' -printf '%T@ %p\n' \
                   | sort -n | head -n -"$KEEP" | cut -d' ' -f2-)
if [ "${#old[@]}" -gt 0 ]; then
    echo "  pruning $((${#old[@]})) archive(s) beyond the newest $KEEP:"
    for f in "${old[@]}"; do rm -f "$f" && echo "    removed $(basename "$f")"; done
fi

echo
echo -e "  ${YELLOW}The ledger hash has changed. Re-baseline before restarting:${NC}"
echo "      sudo ./scripts/kaia-baseline.sh"
echo "      sudo systemctl start kaia-policy-gate"
