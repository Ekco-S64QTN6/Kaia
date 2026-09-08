#!/bin/bash
# Install Kaia systemd services with the correct project path.
# Run with sudo from the repository root: sudo scripts/install_services.sh
set -e
KAIA_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SERVICE in kaia-policy-gate kaia-lockdown; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed "s|/home/ekco/github/Kaia|${KAIA_PROJECT_DIR}|g" \
        "$SRC" > "$DEST"
    echo "Installed $DEST"
done
# Pre-create filesystem honeypot files/folders so systemd can bind-mount them
mkdir -p /var/backups /root/.ssh
touch /etc/api_keys.json /var/backups/credentials.txt /root/.ssh/authorized_keys.bak
chmod 600 /etc/api_keys.json /var/backups/credentials.txt /root/.ssh/authorized_keys.bak

# Provision root-only secret environment file for systemd.
#
# There is deliberately NO fallback default here. A previous version wrote a
# literal signing key when .env was absent -- a value committed to this public
# repository, which meant a fresh install silently ran with a publicly known
# secret and forgeable capability tokens. Per the fail-closed axiom, a missing
# or empty secret aborts the install instead.
mkdir -p /etc/kaia
chmod 700 /etc/kaia

if [ ! -f "$KAIA_PROJECT_DIR/.env" ]; then
    echo "ERROR: $KAIA_PROJECT_DIR/.env not found." >&2
    echo "       KAIA_CAPABILITY_TOKEN_SECRET must be set before installing." >&2
    echo "       Generate one with:  openssl rand -hex 32" >&2
    exit 1
fi

secret_line="$(grep -E '^(export )?KAIA_CAPABILITY_TOKEN_SECRET=' "$KAIA_PROJECT_DIR/.env" | sed 's/^export //' | head -n1)"
secret_val="${secret_line#*=}"
secret_val="${secret_val%\"}"; secret_val="${secret_val#\"}"

if [ -z "$secret_val" ]; then
    echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET is missing or empty in .env." >&2
    echo "       Generate one with:  openssl rand -hex 32" >&2
    exit 1
fi

# Refuse the value that leaked into this repository's git history.
if [ "$secret_val" = "kaia_secure_signing_secret_key_2026" ]; then
    echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET is set to the value published in this" >&2
    echo "       repository's git history. Capability tokens signed with it can be" >&2
    echo "       forged by anyone who has read the source. Rotate it:" >&2
    echo "           openssl rand -hex 32" >&2
    exit 1
fi

printf 'KAIA_CAPABILITY_TOKEN_SECRET=%s\n' "$secret_val" > /etc/kaia/secret.env
chmod 600 /etc/kaia/secret.env

systemctl daemon-reload
systemctl enable --now kaia-policy-gate.service
echo "Done. kaia-lockdown.service installed but not enabled (start manually on breach)."
