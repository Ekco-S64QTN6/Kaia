# Linux Network Issues Troubleshooting Guide

## Initial Checks
* **Physical Connection:**
    * Verify Ethernet cable is plugged in firmly.
    * Check router/modem status lights.
* **Network Manager Status:**
    * `sudo systemctl status NetworkManager`: Check if NetworkManager service is running.
    * `sudo systemctl restart NetworkManager`: Restart the service if it's not working correctly.
* **Interface Status:**
    * `ip link show`: Lists all network interfaces and their states (UP/DOWN).
    * `ip a`: Shows IP addresses assigned to interfaces.
    * `ifconfig -a` (if `net-tools` is installed): Alternative to `ip a`.

## IP Addressing & Routing
* **DHCP/Static IP Check:**
    * `cat /etc/netplan/*.yaml` (for Netplan) or `cat /etc/network/interfaces` (Debian/Ubuntu) or `cat /etc/systemd/network/*.network` (Systemd-networkd).
    * `ip r`: View routing table. Default route (0.0.0.0) should point to your gateway.
* **Ping Gateway:** `ping -c 4 <your_gateway_ip>`: Test connectivity to your router.
* **Ping External Host:** `ping -c 4 8.8.8.8`: Test connectivity to the internet (Google DNS).

## DNS Resolution
* **Check DNS Servers:**
    * `cat /etc/resolv.conf`: Lists DNS servers being used.
    * `resolvectl status` (for systemd-resolved): Detailed DNS resolver status.
* **Test DNS Resolution:**
    * `dig google.com` or `nslookup google.com`: Test if DNS names are resolving to IP addresses.
* **Clear DNS Cache:**
    * `sudo systemctl restart systemd-resolved` (if using systemd-resolved).
    * `sudo /etc/init.d/nscd restart` (if using nscd).

## Firewall & Proxy
* **Firewall Status:**
    * `sudo ufw status` (if using UFW).
    * `sudo firewall-cmd --state` (if using Firewalld).
    * Ensure necessary ports (e.g., 80, 443, 22) are open.
* **Proxy Settings:**
    * Check environment variables: `echo $http_proxy`, `echo $https_proxy`.
    * Check browser/application-specific proxy settings.

## Wireless (Wi-Fi) Specific Issues
* **Hardware Switch:** Ensure Wi-Fi is enabled via physical switch or Fn key.
* **Driver Check:**
    * `lspci -k | grep -EA3 'Network|Wireless|Ethernet'`: Check loaded kernel modules for your network card.
    * `dmesg | grep wifi`: Look for driver-related errors.
* **Network Manager Applet:** Ensure it's running and connected to the correct SSID.
* **Reassociate/Reconnect:** Disconnect and reconnect to the Wi-Fi network.
