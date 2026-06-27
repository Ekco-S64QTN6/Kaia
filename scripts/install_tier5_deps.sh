#!/bin/bash
# Install script for Kaia Tier 5 Dependencies

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

echo -e "${COLOR_BLUE}Checking running kernel headers...${COLOR_RESET}"
KERNEL_VERSION=$(uname -r)
HEADERS_DIR="/usr/lib/modules/${KERNEL_VERSION}/build"

if [ ! -d "${HEADERS_DIR}" ] || [ ! -f "${HEADERS_DIR}/Makefile" ]; then
    echo -e "${COLOR_RED}Error: Kernel headers for running kernel ${KERNEL_VERSION} are not installed or mismatched.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Please install the appropriate headers (e.g., linux-headers, linux-lts-headers) for your kernel version.${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_GREEN}Kernel headers verified for ${KERNEL_VERSION}.${COLOR_RESET}"
fi

if command -v pacman &> /dev/null; then
    echo -e "${COLOR_BLUE}Arch Linux detected. Installing dependencies via pacman...${COLOR_RESET}"
    # bcc includes bcc/bpf tools and python bindings.
    # yara is the security analysis tool.
    # maxminddb is for GeoIPMMDB.
    # nftables is for firewall control.
    sudo pacman -S --needed --noconfirm \
        bcc \
        yara \
        python-yara \
        python-maxminddb \
        maxminddb \
        nftables
else
    echo -e "${COLOR_YELLOW}Warning: pacman not found. Please install BCC, Yara, MaxMindDB, and nftables manually.${COLOR_RESET}"
fi

echo -e "${COLOR_GREEN}Tier 5 system dependencies check and installation complete.${COLOR_RESET}"
