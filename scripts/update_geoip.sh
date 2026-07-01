#!/bin/bash
# Download and update script for MaxMind GeoLite2 City database.
# Requires the MAXMIND_LICENSE_KEY environment variable.
# 
# Usage:
#   export MAXMIND_LICENSE_KEY="your_license_key_here"
#   ./scripts/update_geoip.sh

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

if [ -z "$MAXMIND_LICENSE_KEY" ]; then
    echo -e "${COLOR_RED}Error: MAXMIND_LICENSE_KEY environment variable is not set.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Please obtain a free license key from maxmind.com and export it before running this script.${COLOR_RESET}"
    exit 1
fi

if [ -z "$MAXMIND_ACCOUNT_ID" ]; then
    echo -e "${COLOR_RED}Error: MAXMIND_ACCOUNT_ID environment variable is not set.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Please set your MaxMind Account ID (found on your maxmind.com account page).${COLOR_RESET}"
    exit 1
fi

GEOIP_DIR="storage/threat_intel/geoip"
mkdir -p "$GEOIP_DIR"

URL="https://download.maxmind.com/geoip/databases/GeoLite2-City/download?suffix=tar.gz"
TEMP_TAR="/tmp/GeoLite2-City.tar.gz"
TEMP_DIR="/tmp/GeoLite2-City-extracted"

echo -e "${COLOR_BLUE}Downloading GeoLite2 City database...${COLOR_RESET}"
if curl -sSL -u "${MAXMIND_ACCOUNT_ID}:${MAXMIND_LICENSE_KEY}" -o "$TEMP_TAR" "$URL"; then
    echo -e "${COLOR_GREEN}Download complete.${COLOR_RESET}"
else
    echo -e "${COLOR_RED}Error: Failed to download database.${COLOR_RESET}"
    exit 1
fi

echo -e "${COLOR_BLUE}Extracting database...${COLOR_RESET}"
mkdir -p "$TEMP_DIR"
if tar -xzf "$TEMP_TAR" -C "$TEMP_DIR"; then
    MMDB_FILE=$(find "$TEMP_DIR" -name "*.mmdb" | head -n 1)
    if [ -n "$MMDB_FILE" ] && [ -f "$MMDB_FILE" ]; then
        mv "$MMDB_FILE" "${GEOIP_DIR}/GeoLite2-City.mmdb"
        echo -e "${COLOR_GREEN}GeoLite2 City database updated successfully at ${GEOIP_DIR}/GeoLite2-City.mmdb${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: .mmdb file not found in download archive.${COLOR_RESET}"
        exit 1
    fi
else
    echo -e "${COLOR_RED}Error: Extraction failed.${COLOR_RESET}"
    exit 1
fi

# Cleanup
rm -f "$TEMP_TAR"
rm -rf "$TEMP_DIR"

echo -e "${COLOR_GREEN}Cleanup complete.${COLOR_RESET}"
