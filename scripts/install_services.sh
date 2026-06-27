#!/bin/bash
# Install Kaia systemd services with the correct project path.
# Run with sudo from the repository root: sudo scripts/install_services.sh
set -e
KAIA_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SERVICE in kaia-policy-gate kaia-lockdown; do
    SRC="$KAIA_PROJECT_DIR/scripts/${SERVICE}.service"
    DEST="/etc/systemd/system/${SERVICE}.service"
    sed "s|KAIA_PROJECT_DIR=/home/ekco/github/Kaia|KAIA_PROJECT_DIR=${KAIA_PROJECT_DIR}|g" \
        "$SRC" > "$DEST"
    echo "Installed $DEST"
done
systemctl daemon-reload
systemctl enable --now kaia-policy-gate.service
echo "Done. kaia-lockdown.service installed but not enabled (start manually on breach)."
