#!/bin/bash
# Download and update script for MaxMind GeoLite2 City database.
# Requires the MAXMIND_LICENSE_KEY and MAXMIND_ACCOUNT_ID environment variables.
#
# Usage:
#   export MAXMIND_LICENSE_KEY="your_license_key_here"
#   export MAXMIND_ACCOUNT_ID="your_account_id_here"
#   ./scripts/update_geoip.sh

set -euo pipefail
umask 077

COLOR_GREEN="\033[92m"; COLOR_BLUE="\033[94m"; COLOR_YELLOW="\033[93m"; COLOR_RED="\033[91m"; COLOR_RESET="\033[0m"

if [ -z "${MAXMIND_LICENSE_KEY:-}" ]; then
    echo -e "${COLOR_RED}Error: MAXMIND_LICENSE_KEY environment variable is not set.${COLOR_RESET}" >&2
    echo -e "${COLOR_YELLOW}Please obtain a free license key from maxmind.com and export it before running this script.${COLOR_RESET}" >&2
    exit 1
fi

if [ -z "${MAXMIND_ACCOUNT_ID:-}" ]; then
    echo -e "${COLOR_RED}Error: MAXMIND_ACCOUNT_ID environment variable is not set.${COLOR_RESET}" >&2
    echo -e "${COLOR_YELLOW}Please set your MaxMind Account ID (found on your maxmind.com account page).${COLOR_RESET}" >&2
    exit 1
fi

# Keep credentials out of curl's command line and prevent curl-config injection.
if [[ ! "$MAXMIND_ACCOUNT_ID" =~ ^[0-9]+$ ]] || [[ ! "$MAXMIND_LICENSE_KEY" =~ ^[A-Za-z0-9]+$ ]]; then
    echo "Error: MaxMind account ID and license key must contain only their documented alphanumeric characters." >&2
    exit 1
fi

GEOIP_DIR="storage/threat_intel/geoip"
mkdir -p -- "$GEOIP_DIR"

URL="https://download.maxmind.com/geoip/databases/GeoLite2-City/download?suffix=tar.gz"
# Never use predictable shared /tmp filenames: a symlink there could redirect
# a privileged download or make cleanup remove files that this run did not create.
TEMP_ROOT="$(mktemp -d /tmp/kaia-geoip.XXXXXX)"
TEMP_TAR="${TEMP_ROOT}/download.tar.gz"
TEMP_DIR="${TEMP_ROOT}/extracted"
CURL_CONFIG="${TEMP_ROOT}/curl.conf"
MMDB_FILE="${TEMP_DIR}/GeoLite2-City.mmdb"

cleanup() {
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT

mkdir -- "$TEMP_DIR"
printf 'user = "%s:%s"\n' "$MAXMIND_ACCOUNT_ID" "$MAXMIND_LICENSE_KEY" > "$CURL_CONFIG"
chmod 600 -- "$CURL_CONFIG"

echo -e "${COLOR_BLUE}Downloading GeoLite2 City database...${COLOR_RESET}"
if curl --config "$CURL_CONFIG" --fail --silent --show-error --location --output "$TEMP_TAR" "$URL"; then
    echo -e "${COLOR_GREEN}Download complete.${COLOR_RESET}"
else
    echo -e "${COLOR_RED}Error: Failed to download database.${COLOR_RESET}" >&2
    exit 1
fi

# Do not run tar's general-purpose extractor on a downloaded archive. Validate
# every member and copy only the single expected regular MMDB member to a new
# file inside our private temporary directory; paths, links, devices, and other
# archive members are never materialized on disk.
echo -e "${COLOR_BLUE}Validating and extracting database...${COLOR_RESET}"
if ! python3 - "$TEMP_TAR" "$MMDB_FILE" <<'PYEOF'
import os
import sys
import tarfile
from pathlib import PurePosixPath

archive_path, output_path = sys.argv[1:3]
max_size = 512 * 1024 * 1024

try:
    with tarfile.open(archive_path, "r:gz") as archive:
        candidates = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("archive contains an absolute or parent-traversal path")
            if not (member.isdir() or member.isfile()):
                raise ValueError("archive contains a link or special file")
            if member.isfile() and path.name == "GeoLite2-City.mmdb":
                if len(path.parts) != 2 or not path.parts[0].startswith("GeoLite2-City_"):
                    raise ValueError("MMDB file is not at the expected archive path")
                candidates.append(member)

        if len(candidates) != 1:
            raise ValueError("archive must contain exactly one expected GeoLite2-City.mmdb file")

        source = archive.extractfile(candidates[0])
        if source is None:
            raise ValueError("MMDB archive entry could not be read")
        total = 0
        with source, open(output_path, "xb") as destination:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_size:
                    raise ValueError("MMDB database exceeds the 512 MiB safety limit")
                destination.write(chunk)
            if total == 0:
                raise ValueError("MMDB database is empty")
            destination.flush()
            os.fsync(destination.fileno())
except Exception as exc:
    print(f"Error: rejected GeoLite2 archive: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
then
    echo -e "${COLOR_RED}Error: archive validation or extraction failed.${COLOR_RESET}" >&2
    exit 1
fi

if [ -f "$MMDB_FILE" ] && [ ! -L "$MMDB_FILE" ] && [ -s "$MMDB_FILE" ]; then
    mv -- "$MMDB_FILE" "${GEOIP_DIR}/GeoLite2-City.mmdb"
    echo -e "${COLOR_GREEN}GeoLite2 City database updated successfully at ${GEOIP_DIR}/GeoLite2-City.mmdb${COLOR_RESET}"
else
    echo -e "${COLOR_RED}Error: validated .mmdb file not found.${COLOR_RESET}" >&2
    exit 1
fi

# TEMP_ROOT is removed on success, failure, and signals via the EXIT trap.
echo -e "${COLOR_GREEN}Cleanup complete.${COLOR_RESET}"
