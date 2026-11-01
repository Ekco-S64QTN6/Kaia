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

# These scripts perform privileged recursive sync/chown/chmod operations.
# Constrain targets to dedicated application-data roots rather than only rejecting '/'.
case "$DEPLOY_DIR" in
    /opt/*) ;;
    *) echo "DEPLOY_DIR must be a dedicated directory below /opt." >&2; exit 1 ;;
esac
case "$STATE_DIR" in
    /var/lib/*|/srv/*|/mnt/*) ;;
    *) echo "STATE_DIR must be a dedicated directory below /var/lib, /srv, or /mnt." >&2; exit 1 ;;
esac

validate_root_controlled_path() {
    local target="$1" label="$2" current="$1" owner mode
    while :; do
        if [ -L "$current" ]; then
            echo "$label has a symlinked path component: $current" >&2
            return 1
        fi
        if [ -e "$current" ]; then
            if [ ! -d "$current" ]; then
                echo "$label path component is not a directory: $current" >&2
                return 1
            fi
            owner="$(stat -c '%u' -- "$current")" || return 1
            mode="$(stat -c '%a' -- "$current")" || return 1
            if [ "$owner" != "0" ] || (( (8#$mode & 0022) != 0 )); then
                echo "$label path component must be root-owned and not group/world-writable: $current" >&2
                return 1
            fi
        fi
        [ "$current" = "/" ] && break
        current="$(dirname -- "$current")"
    done
}

# Root later syncs code and writes integrity/audit state to these destinations.
# Every existing path component must be root-controlled, including custom /srv
# or /mnt ancestors, so a non-root user cannot swap the destination after checks.
if ! validate_root_controlled_path "$DEPLOY_DIR" DEPLOY_DIR; then
    exit 1
fi
if ! validate_root_controlled_path "$STATE_DIR" STATE_DIR; then
    exit 1
fi

PROJECT_REAL="$(realpath -e -- "$KAIA_PROJECT_DIR")"
case "$DEPLOY_DIR/" in
    "$PROJECT_REAL/"*) echo "DEPLOY_DIR must not be inside the source checkout." >&2; exit 1 ;;
esac
case "$PROJECT_REAL/" in
    "$DEPLOY_DIR/"*) echo "DEPLOY_DIR must not contain the source checkout." >&2; exit 1 ;;
esac
case "$STATE_DIR/" in
    "$PROJECT_REAL/"*) echo "STATE_DIR must not be inside the source checkout." >&2; exit 1 ;;
esac
case "$PROJECT_REAL/" in
    "$STATE_DIR/"*) echo "STATE_DIR must not contain the source checkout." >&2; exit 1 ;;
esac
case "$STATE_DIR/" in
    "$DEPLOY_DIR/"*) echo "STATE_DIR must not be inside DEPLOY_DIR." >&2; exit 1 ;;
esac
case "$DEPLOY_DIR/" in
    "$STATE_DIR/"*) echo "DEPLOY_DIR must not be inside STATE_DIR." >&2; exit 1 ;;
esac

# Preflight the signing secret before stopping services or changing deployed code.
# Parse only the intended key assignment; never source the project .env as root.
if [ ! -f "$KAIA_PROJECT_DIR/.env" ]; then
    echo "ERROR: $KAIA_PROJECT_DIR/.env not found." >&2
    echo "       KAIA_CAPABILITY_TOKEN_SECRET must be set before installing." >&2
    echo "       Generate one with: openssl rand -hex 32" >&2
    exit 1
fi
secret_line="$(grep -E '^(export )?KAIA_CAPABILITY_TOKEN_SECRET=' "$KAIA_PROJECT_DIR/.env" | sed 's/^export //' | head -n1)"
secret_val="${secret_line#*=}"
case "$secret_val" in
    \"*\") secret_val="${secret_val:1:${#secret_val}-2}" ;;
    \'*\') secret_val="${secret_val:1:${#secret_val}-2}" ;;
esac
if [ -z "$secret_val" ]; then
    echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET is missing or empty in .env." >&2
    echo "       Generate one with: openssl rand -hex 32" >&2
    exit 1
fi
if [ "$secret_val" = "kaia_secure_signing_secret_key_2026" ]; then
    echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET is the value published in git history." >&2
    echo "       Rotate it before installing; capability tokens can otherwise be forged." >&2
    exit 1
fi

# The signing key is written directly to systemd's EnvironmentFile and protects
# HMAC-SHA256 capability tokens. Require 32 random bytes encoded as 64 hex chars:
# this also prevents whitespace/newline/comment syntax from altering EnvironmentFile.
if [[ ! "$secret_val" =~ ^[A-Fa-f0-9]{64}$ ]]; then
    echo "ERROR: KAIA_CAPABILITY_TOKEN_SECRET must be exactly 64 hexadecimal characters." >&2
    echo "       Generate a 32-byte key with: openssl rand -hex 32" >&2
    exit 1
fi

# Root runs the deployed code. Reject unexpected links before any recursive
# ownership normalization. The tracked dashboard link is permitted only as its
# exact relative target and only when it resolves inside the tree being checked.
validate_deployment_symlinks() {
    local tree_root="$1" mode="$2" root_real link relative target resolved
    root_real="$(realpath -e -- "$tree_root")" || return 1

    if [ "$mode" = "source" ]; then
        while IFS= read -r -d '' link; do
            relative="${link#"$tree_root"/}"
            target="$(readlink -- "$link")" || return 1
            if [[ "$relative" != "static/dashboard" || "$target" != "../core/web/static" ]]; then
                echo "ERROR: refusing unexpected symlink in source checkout: $relative" >&2
                return 1
            fi
            resolved="$(realpath -e -- "$link")" || {
                echo "ERROR: dangling dashboard symlink: $relative" >&2
                return 1
            }
            case "$resolved/" in
                "$root_real/"*) ;;
                *) echo "ERROR: dashboard symlink escapes source checkout." >&2; return 1 ;;
            esac
        done < <(find "$tree_root" \
            \( -path "$tree_root/.git" -o -path "$tree_root/.venv" -o \
               -path "$tree_root/storage" -o -path "$tree_root/logs" -o \
               -path "$tree_root/.pytest_cache" -o -path "$tree_root/tests" -o \
               -name '__pycache__' \
            \) -prune -o -type l -print0)
    else
        while IFS= read -r -d '' link; do
            relative="${link#"$tree_root"/}"
            target="$(readlink -- "$link")" || return 1
            if [[ "$relative" != "static/dashboard" || "$target" != "../core/web/static" ]]; then
                echo "ERROR: refusing unexpected symlink in deployed tree: $relative" >&2
                return 1
            fi
            resolved="$(realpath -e -- "$link")" || {
                echo "ERROR: dangling dashboard symlink in deployed tree: $relative" >&2
                return 1
            }
            case "$resolved/" in
                "$root_real/"*) ;;
                *) echo "ERROR: dashboard symlink escapes deployment tree." >&2; return 1 ;;
            esac
        done < <(find "$tree_root" -path "$tree_root/storage" -prune -o -type l -print0)
    fi
}
if ! validate_deployment_symlinks "$PROJECT_REAL" source; then
    exit 1
fi

# --- Deploy a root-owned copy of the code ------------------------------------
#
# The daemon runs as root. Running it directly out of a checkout under $HOME
# means root executes code that an unprivileged user can rewrite at any time --
# the single largest privilege-escalation path on the host, and one that tamper
# detection can only report after the fact, not prevent.
#
# Deploy to a root-owned tree instead. Develop in the checkout, install to run.
if systemctl is-active --quiet kaia-policy-gate.service; then
    echo "==> Stopping kaia-policy-gate before changing watched code"
    systemctl stop kaia-policy-gate.service
fi
if systemctl is-active --quiet kaia-restore-blocks.service; then
    echo "==> Stopping block replay before changing deployed scripts"
    systemctl stop kaia-restore-blocks.service
fi

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
if ! validate_deployment_symlinks "$DEPLOY_DIR" deployed; then
    echo "ERROR: refusing to chown or execute a deployment containing unexpected symlinks." >&2
    exit 1
fi

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
    dst="$STATE_DIR/$f"
    # Never follow a symlink while running as root; it could redirect ownership changes.
    if [ -L "$src" ]; then
        echo "ERROR: refusing to migrate symlinked runtime data: $src" >&2
        exit 1
    fi
    if [ -L "$dst" ] || { [ -e "$dst" ] && [ ! -f "$dst" ]; }; then
        echo "ERROR: runtime data destination is not a regular file: $dst" >&2
        exit 1
    fi
    if [ -f "$src" ] && [ ! -e "$dst" ]; then
        cp -- "$src" "$dst"
        chown root:kaiacord "$dst"
        chmod 640 "$dst"
        echo "    migrated $f ($(stat -c%s "$src") bytes)"
    elif [ -f "$dst" ]; then
        chown root:kaiacord "$dst"
        chmod 640 "$dst"
    fi
done

# FIM history used to live outside the configured state directory. Preserve it
# during the first upgrade to the unified state layout.
legacy_fim_db="/var/lib/secdaemon/fim_audit.db"
fim_db_destination="$STATE_DIR/fim_audit.db"
if [ -L "$fim_db_destination" ] || { [ -e "$fim_db_destination" ] && [ ! -f "$fim_db_destination" ]; }; then
    echo "ERROR: FIM DB destination exists but is not a regular file: $fim_db_destination" >&2
    exit 1
fi
if [ -L "$legacy_fim_db" ]; then
    echo "ERROR: refusing to migrate legacy FIM history through a symlink: $legacy_fim_db" >&2
    exit 1
fi
if [ -f "$legacy_fim_db" ] && [ ! -e "$fim_db_destination" ]; then
    cp -- "$legacy_fim_db" "$fim_db_destination"
    echo "    migrated legacy FIM history ($(stat -c%s "$legacy_fim_db") bytes)"
fi
if [ -L "$fim_db_destination" ] || { [ -e "$fim_db_destination" ] && [ ! -f "$fim_db_destination" ]; }; then
    echo "ERROR: FIM DB is not a regular file: $fim_db_destination" >&2
    exit 1
fi
if [ -f "$fim_db_destination" ]; then
    chown root:kaiacord "$fim_db_destination"
    chmod 640 "$fim_db_destination"
fi

# The home-directory canary belongs to the login that ran sudo.
CANARY_OWNER="${SUDO_USER:-}"
OWNER_HOME="/nonexistent"
if [ -n "$CANARY_OWNER" ] && [ "$CANARY_OWNER" != "root" ]; then
    OWNER_HOME="$(getent passwd "$CANARY_OWNER" | cut -d: -f6)"
    [ -n "$OWNER_HOME" ] || OWNER_HOME="/nonexistent"
fi

for SERVICE in kaia-policy-gate kaia-lockdown kaia-restore-blocks; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    [ -f "$SRC" ] || continue
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed -e "s|/home/ekco/github/Kaia|${DEPLOY_DIR}|g" \
        -e "s|__DEPLOY_DIR__|${DEPLOY_DIR}|g" \
        -e "s|__STATE_DIR__|${STATE_DIR}|g" \
        -e "s|__OWNER_HOME__|${OWNER_HOME}|g" \
        "$SRC" > "$DEST"
    chmod 644 "$DEST"
    echo "Installed $DEST"
done
# Canary flag files (~/flag.txt, /root/root.txt) must exist before the gate
# starts: the honeypot registers them by inode at startup. The helper never
# overwrites a file it cannot prove is a canary.
echo "==> Installing canary flag files"
python3 "$DEPLOY_DIR/scripts/install_canaries.py" \
    ${CANARY_OWNER:+--owner "$CANARY_OWNER"} \
    --template-dir "$DEPLOY_DIR/security/canaries"

# Filesystem decoys are created by HoneypotCoordinator with exclusive, no-follow
# creation only when the target path is vacant. Never touch, truncate, or chmod a
# potentially real API-key file, backup credential file, or SSH backup.
# Provision root-only environment files for systemd and manual recovery.
mkdir -p /etc/kaia
chmod 700 /etc/kaia
printf '%s\n' "KAIA_CAPABILITY_TOKEN_SECRET=$secret_val" > /etc/kaia/secret.env
chmod 600 /etc/kaia/secret.env
printf '%s\n' "KAIA_STATE_DIR=$STATE_DIR" > /etc/kaia/state.env
chmod 600 /etc/kaia/state.env

systemctl daemon-reload
systemctl enable kaia-restore-blocks.service 2>/dev/null || true
systemctl enable kaia-policy-gate.service

echo
echo "==> Deployment installed; the Policy Gate is NOT started automatically."
echo "    Review the deployed source and baseline diff, then run:"
echo "        sudo ${DEPLOY_DIR}/scripts/kaia-baseline.sh"
echo "        sudo systemctl start kaia-policy-gate.service"
echo "The service stays stopped until the reviewed baseline is recorded."
echo "kaia-lockdown.service is installed but not enabled (start manually on breach)."
