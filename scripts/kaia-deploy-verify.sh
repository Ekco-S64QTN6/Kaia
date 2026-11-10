#!/bin/bash
# Deploy this checkout to /opt/kaia and run the host acceptance checks. Run as root:
#   sudo scripts/kaia-deploy-verify.sh   (log: /var/log/kaia/deploy-<time>.log, group kaiacord)
# Everything root writes goes to root-owned locations: never into a user-writable
# directory, where a planted symlink would redirect the write.
# Order matters (CLAUDE.md): install -> baseline -> start. If the baseline cannot
# be recorded, the gate is disabled so the next boot cannot lock the host down.
set -uo pipefail
[ "$EUID" -eq 0 ] || { echo "run with sudo"; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-}"
[ -n "$USER_NAME" ] && [ "$USER_NAME" != root ] || { echo "run via sudo from your own account"; exit 1; }
install -d -m 0750 -o root -g kaiacord /var/log/kaia
OUT="/var/log/kaia/deploy-$(date +%Y%m%d-%H%M%S).log"
( umask 027; : > "$OUT" ) && chgrp kaiacord "$OUT"
exec > >(tee "$OUT") 2>&1
step() { echo; echo "=== $* ($(date -Is))"; }

step "1. install (stops the gate, deploys /opt/kaia, units, timer, canaries, decoys)"
SUDO_USER="$USER_NAME" bash "$REPO/scripts/install_services.sh" || { echo "INSTALL FAILED"; exit 1; }
git -C "$REPO" log --oneline -1

step "2. trusted baseline"
if ! /opt/kaia/scripts/kaia-baseline.sh; then
    echo "BASELINE FAILED: disabling the gate so a stale baseline cannot lock the host at boot"
    systemctl disable kaia-policy-gate.service
    exit 1
fi

step "3. start the gate"
START=$(date '+%Y-%m-%d %H:%M:%S')
systemctl start kaia-policy-gate.service
sleep 20
systemctl --no-pager status kaia-policy-gate.service | head -5
journalctl -u kaia-policy-gate --since "$START" --no-pager | grep -v "Security event recorded" | head -60

step "4. privilege-gain probe: sudo exec (expect privilege_setuid_exec, no CRITICAL)"
sudo -u "$USER_NAME" sudo -n /usr/bin/ss -tlnp >/dev/null 2>&1 || true
step "4b. unexpected-gain path: cap_setuid interpreter copy (expect 'Unexpected root gain')"
# Root-owned private directory on a filesystem without nosuid (file caps need it).
TDIR="$(mktemp -d /var/lib/kaia-privtest.XXXXXX)" && chmod 0711 "$TDIR"
T="$TDIR/kaia-priv-test"
cp /usr/bin/python3 "$T" && chmod 0755 "$T" && setcap cap_setuid+ep "$T"
sudo -u "$USER_NAME" "$T" -c 'import os; os.setuid(0); print("uid now", os.getuid())'
rm -rf "$TDIR"
sleep 3
journalctl -u kaia-policy-gate --since "$START" --no-pager | grep -E "privilege_|Unexpected root gain|not attached" | tail -10

step "5. control health (as the unprivileged user)"
sleep 5
sudo -u "$USER_NAME" python3 -I -s /opt/kaia/scripts/kaia_health_check.py; echo "health exit $?"
systemctl list-timers kaia-health-check.timer --no-pager | head -3
systemctl start kaia-health-check.service; journalctl -u kaia-health-check -n 15 --no-pager

step "5b. honeypot coverage (expect 5 of 5 decoy/canary files)"
journalctl -u kaia-policy-gate --since "$START" --no-pager | grep -E "Honeypot inode watches|Could not create filesystem decoy" | tail -3

step "6. installed-content attestation (as root)"
python3 -I -s /opt/kaia/scripts/kaia_package_attest.py; echo "attest exit $?"
step "6b. against the signed package archives (as the unprivileged timer runs it)"
systemctl start kaia-package-verify.service; echo "signed verify exit $?"
journalctl -u kaia-package-verify -n 8 --no-pager

step "6c. security advisories"
systemctl start kaia-vuln-check.service; echo "vuln check exit $?"
journalctl -u kaia-vuln-check -n 12 --no-pager

step "7. versions"
uname -r; systemctl --version | head -1
echo; echo "DONE. Log: $OUT"
