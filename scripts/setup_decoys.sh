#!/bin/bash
# One-time setup script for Kaia honeypot network decoys.
# Creates an isolated namespace and a host-only veth link.
# This script intentionally leaves host-wide forwarding and NAT unchanged.

set -euo pipefail
umask 077

COLOR_GREEN="\033[92m"; COLOR_BLUE="\033[94m"; COLOR_YELLOW="\033[93m"; COLOR_RED="\033[91m"; COLOR_RESET="\033[0m"

if [ "$EUID" -ne 0 ]; then
    echo -e "${COLOR_RED}Error: Please run this script with sudo or as root.${COLOR_RESET}" >&2
    exit 1
fi

if ! command -v ip >/dev/null 2>&1; then
    echo "ERROR: iproute2's ip command is required." >&2
    exit 1
fi

# Validate existing state before making changes. A complete configuration is
# accepted only when all expected links, addresses, and UP states match exactly;
# a partial or unrelated pre-existing resource is never adopted or overwritten.
namespace_exists=0
if ip netns list | awk '$1 == "ns_decoy" { found = 1 } END { exit !found }'; then
    namespace_exists=1
fi

host_link_exists=0
if ip link show dev veth_host >/dev/null 2>&1; then
    host_link_exists=1
fi

host_peer_exists=0
if ip link show dev veth_decoy >/dev/null 2>&1; then
    host_peer_exists=1
fi

namespace_peer_exists=0
if [ "$namespace_exists" -eq 1 ] && ip netns exec ns_decoy ip link show dev veth_decoy >/dev/null 2>&1; then
    namespace_peer_exists=1
fi

verify_link_up() {
    local output
    output="$(ip -o link show dev "$1")" || return 1
    [[ "$output" == *",UP,"* || "$output" == *",UP>"* ]]
}

verify_namespace_link_up() {
    local output
    output="$(ip netns exec ns_decoy ip -o link show dev veth_decoy)" || return 1
    [[ "$output" == *",UP,"* || "$output" == *",UP>"* ]]
}

verify_namespace_loopback_up() {
    local output
    output="$(ip netns exec ns_decoy ip -o link show dev lo)" || return 1
    [[ "$output" == *",UP,"* || "$output" == *",UP>"* ]]
}

verify_host_peer_binding() {
    local output
    output="$(ip -d link show dev veth_host)" || return 1
    [[ "$output" == *"veth"* && "$output" == *"link-netns ns_decoy"* ]]
}

expected_host_addr() {
    [ "$(ip -o -4 addr show dev veth_host | awk '{print $4}')" = "10.254.0.1/30" ]
}

expected_namespace_addr() {
    [ "$(ip netns exec ns_decoy ip -o -4 addr show dev veth_decoy | awk '{print $4}')" = "10.254.0.2/30" ]
}

if [ "$namespace_exists" -eq 1 ] && [ "$host_link_exists" -eq 1 ] && \
   [ "$host_peer_exists" -eq 0 ] && [ "$namespace_peer_exists" -eq 1 ]; then
    if expected_host_addr && expected_namespace_addr && verify_host_peer_binding && \
       verify_link_up veth_host && verify_namespace_link_up && verify_namespace_loopback_up; then
        echo -e "${COLOR_GREEN}Verified existing ns_decoy host-only network; no changes needed.${COLOR_RESET}"
        exit 0
    fi
fi

if [ "$namespace_exists" -eq 1 ] || [ "$host_link_exists" -eq 1 ] || \
   [ "$host_peer_exists" -eq 1 ] || [ "$namespace_peer_exists" -eq 1 ]; then
    echo -e "${COLOR_RED}ERROR: existing or partial decoy namespace/interface state detected. Inspect it manually; refusing to mutate it.${COLOR_RESET}" >&2
    exit 1
fi

NETNS_CREATED=0
HOST_LINK_CREATED=0
SETUP_COMPLETE=0

cleanup_on_exit() {
    local status=$?
    if [ "$status" -ne 0 ] && [ "$SETUP_COMPLETE" -eq 0 ]; then
        echo "ERROR: decoy network setup failed; rolling back resources created by this invocation." >&2
        if [ "$HOST_LINK_CREATED" -eq 1 ]; then
            if ! ip link del dev veth_host; then
                echo "WARNING: failed to remove newly created host veth; inspect veth_host manually." >&2
            fi
        fi
        if [ "$NETNS_CREATED" -eq 1 ]; then
            if ! ip netns del ns_decoy; then
                echo "WARNING: failed to remove newly created ns_decoy namespace; inspect it manually." >&2
            fi
        fi
    fi
    return "$status"
}
trap cleanup_on_exit EXIT

echo -e "${COLOR_BLUE}Creating network namespace ns_decoy...${COLOR_RESET}"
ip netns add ns_decoy
NETNS_CREATED=1

echo -e "${COLOR_BLUE}Creating host/decoy veth pair...${COLOR_RESET}"
ip link add veth_host type veth peer name veth_decoy
HOST_LINK_CREATED=1
ip link set dev veth_decoy netns ns_decoy

echo -e "${COLOR_BLUE}Assigning isolated link addresses...${COLOR_RESET}"
ip addr add 10.254.0.1/30 dev veth_host
ip link set dev veth_host up
ip netns exec ns_decoy ip addr add 10.254.0.2/30 dev veth_decoy
ip netns exec ns_decoy ip link set dev veth_decoy up
ip netns exec ns_decoy ip link set dev lo up

# Deliberately do not create a default route, enable global IPv4 forwarding, or
# add NAT rules. The namespace remains host-reachable on 10.254.0.2; exposing it
# externally requires a separate, explicitly reviewed firewall/NAT configuration.
if ! expected_host_addr || ! expected_namespace_addr || ! verify_host_peer_binding || \
   ! verify_link_up veth_host || ! verify_namespace_link_up || \
   ! verify_namespace_loopback_up; then
    echo "ERROR: post-setup verification failed; refusing to report success." >&2
    exit 1
fi

SETUP_COMPLETE=1
echo -e "${COLOR_GREEN}Decoy network setup verified successfully.${COLOR_RESET}"
echo -e "${COLOR_YELLOW}Listeners can run inside ns_decoy at 10.254.0.2; external exposure is not configured.${COLOR_RESET}"
