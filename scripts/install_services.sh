#!/bin/bash
# Install Kaia systemd services with the correct project path.
# Run with sudo from the repository root: sudo scripts/install_services.sh
set -e
KAIA_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${KAIA_DEPLOY_DIR:-/opt/kaia}"
STATE_DIR="${KAIA_STATE_DIR:-/var/lib/kaia}"

[ "$EUID" -eq 0 ] || { echo "Run with sudo." >&2; exit 1; }

# These paths are substituted into systemd unit files and used by recursive
# chown/chmod/rsync operations. Canonicalize and reject unsafe/root/overlapping
# destinations before any privileged filesystem changes.
DEPLOY_DIR="$(realpath -m -- "$DEPLOY_DIR")"
STATE_DIR="$(realpath -m -- "$STATE_DIR")"
for path_name in DEPLOY_DIR STATE_DIR; do
    path_value="${!path_name}"
    case "$path_value" in
        /*) ;;
        *) echo "$path_name must be an absolute path." >&2; exit 1 ;;
    esac
    case "$path_value" in
        /) echo "$path_name must not be the filesystem root." >&2; exit 1 ;;
        *[!A-Za-z0-9_./-]*)
            echo "$path_name contains unsupported characters: $path_value" >&2
            exit 1
            ;;
    esac
done
case "$STATE_DIR/" in
    "$DEPLOY_DIR/"*) echo "STATE_DIR must not be inside DEPLOY_DIR." >&2; exit 1 ;;
esac
case "$DEPLOY_DIR/" in
    "$STATE_DIR/"*) echo "DEPLOY_DIR must not be inside STATE_DIR." >&2; exit 1 ;;
esac

# --- Deploy a root-owned copy of the code ------------------------------------
#
# The daemon runs as root. Running it directly out of a checkout under $HOME
# means root executes code that an unprivileged user can rewrite at any time --
# the single largest privilege-escalation path on the host, and one that tamper
# detection can only report after the fact, not prevent.
#
# Deploy to a root-owned tree instead. Develop in the checkout, install to run.
echo "==> Deploying code to $DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
rsync -a --delete \
    --exclude '.git/' --exclude '.venv/' --exclude 'storage/' --exclude 'logs/' \
    --exclude '.env' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.pytest_cache/' --exclude 'tests/' \
    "$KAIA_PROJECT_DIR/" "$DEPLOY_DIR/"

# rsync protects excluded destination files from --delete by default. Explicitly
# remove secrets and VCS metadata left by older deployments before normalizing
# permissions; otherwise an old .env can survive the sync and become mode 0644.
rm -f "$DEPLOY_DIR/.env"
rm -rf "$DEPLOY_DIR/.git" "$DEPLOY_DIR/.venv"
find "$DEPLOY_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$DEPLOY_DIR" -type f -name '*.pyc' -delete

# The systemd unit explicitly grants this log directory write access; create it
# during installation so config import never has to mkdir under ProtectSystem.
mkdir -p "$DEPLOY_DIR/logs"

# The excluded storage/ tree is runtime data, not deployed code. Preserve its
# ownership and modes rather than recursively making old databases world-readable.
find "$DEPLOY_DIR" -path "$DEPLOY_DIR/storage" -prune -o -exec chown root:root {} +
find "$DEPLOY_DIR" -path "$DEPLOY_DIR/storage" -prune -o -type d -exec chmod 755 {} +
find "$DEPLOY_DIR" -path "$DEPLOY_DIR/storage" -prune -o -type f -exec chmod 644 {} +
chmod 755 "$DEPLOY_DIR"/scripts/*.sh "$DEPLOY_DIR"/scripts/*.py 2>/dev/null || true
echo "    $(find "$DEPLOY_DIR" -type f | wc -l) files, root:root, not writable by any other user"

# --- System state directory (INV-008) ----------------------------------------
# The audit ledger must live outside the agent's working tree. root owns it;
# the kaiacord group gets read access so the dashboard can poll it unprivileged.
echo "==> Provisioning $STATE_DIR"
getent group kaiacord >/dev/null || groupadd --system kaiacord
mkdir -p "$STATE_DIR/threat_intel"
chown root:kaiacord "$STATE_DIR" "$STATE_DIR/threat_intel"
chmod 750 "$STATE_DIR"
chmod 750 "$STATE_DIR/threat_intel"

# Migrate any existing in-repo ledger/database exactly once.
for f in audit_ledger.json security_events.db; do
    src="$KAIA_PROJECT_DIR/storage/security/$f"
    if [ -f "$src" ] && [ ! -f "$STATE_DIR/$f" ]; then
        cp -a "$src" "$STATE_DIR/$f"
        chown root:kaiacord "$STATE_DIR/$f"
        chmod 640 "$STATE_DIR/$f"
        echo "    migrated $f ($(stat -c%s "$src") bytes)"
    fi
done

# FIM history used to live outside the configured state directory. Preserve it
# during the first upgrade to the unified state layout.
legacy_fim_db="/var/lib/secdaemon/fim_audit.db"
if [ -f "$legacy_fim_db" ] && [ ! -e "$STATE_DIR/fim_audit.db" ]; then
    cp -a "$legacy_fim_db" "$STATE_DIR/fim_audit.db"
    chown root:kaiacord "$STATE_DIR/fim_audit.db"
    chmod 640 "$STATE_DIR/fim_audit.db"
    echo "    migrated legacy FIM history ($(stat -c%s "$legacy_fim_db") bytes)"
fi

for SERVICE in kaia-policy-gate kaia-lockdown kaia-restore-blocks; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    [ -f "$SRC" ] || continue
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed -e "s|/home/ekco/github/Kaia|${DEPLOY_DIR}|g" \
        -e "s|__DEPLOY_DIR__|${DEPLOY_DIR}|g" \
        -e "s|__STATE_DIR__|${STATE_DIR}|g" \
        "$SRC" > "$DEST"
    chmod 644 "$DEST"
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

# The interactive unlock script is not launched by systemd, so persist the
# configured state path separately for manual recovery commands.
printf 'KAIA_STATE_DIR=%q\n' "$STATE_DIR" > /etc/kaia/state.env
chmod 600 /etc/kaia/state.env

# The secret file is read by the daemon at startup; root-only is correct since
# the daemon runs as root.
chmod 600 /etc/kaia/secret.env

systemctl daemon-reload
systemctl enable kaia-restore-blocks.service 2>/dev/null || true
systemctl enable --now kaia-policy-gate.service

echo
echo "==> Record the trusted integrity baseline for the DEPLOYED tree:"
echo "        sudo ${DEPLOY_DIR}/scripts/kaia-baseline.sh"
echo "    then:  sudo systemctl restart kaia-policy-gate"
echo "Done. kaia-lockdown.service installed but not enabled (start manually on breach)."
