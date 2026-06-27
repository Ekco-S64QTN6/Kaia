#!/bin/bash
# One-time setup script for Kaia Honeypot Network Decoys
# Run this script with root privileges to establish the ns_decoy namespace.

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: Please run this script with sudo or as root.${COLOR_RESET}"
    exit 1
fi

echo -e "${COLOR_BLUE}Creating network namespace ns_decoy...${COLOR_RESET}"
ip netns add ns_decoy 2>/dev/null || true

echo -e "${COLOR_BLUE}Setting up veth interfaces...${COLOR_RESET}"
ip link add veth_host type veth peer name veth_decoy 2>/dev/null || true
ip link set veth_decoy netns ns_decoy 2>/dev/null || true

echo -e "${COLOR_BLUE}Assigning IP addresses...${COLOR_RESET}"
ip addr add 10.254.0.1/30 dev veth_host 2>/dev/null || true
ip link set veth_host up 2>/dev/null || true

ip netns exec ns_decoy ip addr add 10.254.0.2/30 dev veth_decoy 2>/dev/null || true
ip netns exec ns_decoy ip link set veth_decoy up 2>/dev/null || true
ip netns exec ns_decoy ip link set lo up 2>/dev/null || true

echo -e "${COLOR_BLUE}Configuring routing...${COLOR_RESET}"
# Default route inside namespace via host veth
ip netns exec ns_decoy ip route add default via 10.254.0.1 2>/dev/null || true

# Enable forwarding on host
echo 1 > /proc/sys/net/ipv4/ip_forward

echo -e "${COLOR_GREEN}Decoy network setup complete.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}Decoy port listeners can now be run inside namespace 'ns_decoy'.${COLOR_RESET}"
