# Wi-Fi Card Compatibility and Drivers on Linux

## Identifying Your Wi-Fi Chipset
* **PCIe/USB Cards:**
    * `lspci -k | grep -EA3 'Network|Wireless'` (for PCIe cards): Shows PCI devices and their kernel modules.
    * `lsusb -v | grep -EA3 'Wireless|Network'` (for USB dongles): Shows USB devices and relevant drivers.
* **`dmesg` Output:**
    * `dmesg | grep -i wifi`: Look for messages during boot related to your wireless adapter.

## Common Chipsets and Driver Status
* **Intel:** Generally very well supported out-of-the-box with in-kernel `iwlwifi` drivers. Requires firmware (often installed automatically).
* **Realtek:** Mixed support. Some work well, others require specific out-of-tree drivers (e.g., `rtl88x2bu`, `rtl8821ce`), often available in AUR for Arch Linux. May require manual compilation.
* **Broadcom:** Historically problematic. Many require proprietary `wl` driver.
    * For `wl` driver: `sudo pacman -S broadcom-wl` (Arch Linux).
* **Atheros:** Good support with `ath9k`, `ath10k`, `ath11k` in-kernel drivers. Requires firmware.

## Firmware Requirements
* Most modern Wi-Fi chipsets require proprietary firmware blobs to function correctly.
* **Firmware Location:** Typically in `/lib/firmware/`.
* **Installation:** Often installed automatically with the kernel or through firmware packages (e.g., `linux-firmware` on Arch Linux). If using an out-of-tree driver, firmware might be bundled or need separate installation.
* **Missing Firmware:** Look for `firmware missing` or `failed to load firmware` errors in `dmesg` or `journalctl -f`.

## Troubleshooting Common Wi-Fi Issues
* **No Wi-Fi Adapters Detected:**
    * Ensure Wi-Fi is enabled by a physical switch or Function (Fn) key on laptops.
    * Check `rfkill list all`: Ensure Wi-Fi is not hard-blocked or soft-blocked. Use `sudo rfkill unblock all`.
    * Verify correct driver is loaded (`lsmod | grep <driver_name>`).
* **Cannot Connect to Network:**
    * Incorrect password.
    * Weak signal strength.
    * Incorrect security protocol (WPA2/WPA3).
    * MAC address filtering on router (unlikely for home use but check).
    * DNS issues (see `network_issues.md`).
* **Intermittent Connectivity:**
    * Interference from other devices or Wi-Fi networks.
    * Power saving features: Try disabling power management for the Wi-Fi card (`iwconfig <interface> power off`).
    * Outdated driver/firmware.
* **Slow Speeds:**
    * Channel congestion (use `iw dev <interface> scan` or `linssid` to find less congested channels).
    * Driver/firmware issues.
    * Router configuration (e.g., 2.4GHz vs 5GHz, channel width).

## Network Management Tools
* **NetworkManager:** Most common daemon for managing network connections (GUI frontends available).
* **`iwctl` (iwd):** Modern alternative to `wpa_supplicant`, often preferred for its simplicity with Wayland.
* **`wpa_supplicant`:** Command-line tool for WPA/WPA2/WPA3 authentication.
