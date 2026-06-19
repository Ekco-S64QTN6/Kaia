# System Configuration and Troubleshooting

This document provides a comprehensive guide to your system's hardware, software, and configuration, consolidating information from Arch Linux installation, hardware component details, and troubleshooting steps.

## 1. Hardware Components and BIOS Settings

### 1.1. PC Build Components

| Component | Model |
| :----------- | :-------------------------------------------------------- |
| CPU | AMD Ryzen 5 9600X 6-Core Processor |
| Motherboard | ASUS ROG Strix B650-A Gaming WiFi 6E AM5 (LGA1718) Ryzen 7000 Motherboard(12+2 Power Stages,DDR5,3xM.2 Slots,PCIe® 4.0, 2.5G LAN,WiFi 6E,USB 3.2 Gen 2x2 Type-C® Port) |
| RAM | CORSAIR VENGEANCE RGB DDR5 32GB (2x16GB) 6000MHz |
| Storage | WD_BLACK 1TB SN850X NVMe SSD (Operating System) & 2x KingSpec 1TB 2.5 SSD (Storage) |
| CPU Cooler | Thermalright Peerless Assassin 120 SE ARGB White CPU Air Cooler, 6 Heat Pipes CPU Cooler, Dual 120mm TL-C12CW-S PWM Fan, AGHP Technology |
| GPU | NVIDIA GeForce RTX 3060 |
| Power Supply | CORSAIR RM750e (2025) Fully Modular Low-Noise ATX Power Supply |
| Case | MUSETEX ATX PC Case Pre-Install 6 PWM ARGB Fans, Polygonal Mesh Computer Gaming Case, Opening Tempered Glass Side Panel Mid-Tower Case, USB 3.0 x 2, Black, NN8 |
| Keyboard | Owpkeenthy Wired 65% Mechanical Gaming Keyboard with Blue Switch 60% Ultra Compact RGB Gaming Keyboard Backlit Keys N-Key Rollover for PC PS5 Xbox(Blue/ 68 Blue Switch) |
| Mouse | XVX Wired Gaming Mouse, 12000 DPI RGB Gaming Mouse with 12 Backlit Modes & 7 Macro Buttons |
| Monitor | Acer KB272 EBI 27" IPS Full HD (1920 x 1080) Zero-Frame Gaming Office Monitor (Primary Monitor, Left,HDMI) | Acer G246HL Abd 24-Inch Screen LED-Lit Monitor (Secondary Monitor,Right,DVI)|

### 1.2. Critical BIOS Settings

* **Disable Secure Boot:** Necessary for NVIDIA driver installation.
    * *Path: Settings > Security > Secure Boot > Disable*
* **Enable EXPO for RAM:**
    * *Path: OC > A-XMP/EXPO Profile > Profile 1 (6000MHz)*
* **Set PCIe Slot to Gen 3 (Recommended for GTX 950):**
    * *Path: Settings > Advanced > PCI Subsystem Settings > PCIe x16 Slot Speed > Gen 3*
* **Check Boot Mode:** Ensure UEFI (not Legacy/CSM) is enabled.
* **Enable Q-Fan Control and set to PWM Mode.**
* **Increase CPU Current Capability to 140% for potentially higher boost clocks.**

## 2. Arch Linux Installation

### 2.1. Pre-Installation

* **Connect to Wi-Fi:**
    ```bash
    iwctl
    device list
    device name set-property Powered on
    station name scan
    station name get-networks
    station name connect SSID
    ```
* **Verify Network and Prepare:**
    ```bash
    ip -4 addr show wlan0
    dhcpcd -4 wlan0 # If no IP
    ping -4 archlinux.org
    reflector --country "United States" --protocol https --latest 10 --save /etc/pacman.d/mirrorlist
    timedatectl set-ntp true
    ```

### 2.2. Installation (`archinstall`)

* **Run the guided installer:**
    ```bash
    archinstall
    ```
    * **Disk Configuration:** Btrfs (with /boot partition).
    * **Network:** NetworkManager.
    * **Desktop:** plasma (KDE).
    * **Additional Packages:** `nvidia-470xx-dkms`, `nvidia-470xx-settings`, `networkmanager`, `bluez-utils`.

### 2.3. Post-Installation

* **Initial Setup:**
    ```bash
    sudo pacman -Syu
    hostnamectl
    groups
    sudo usermod -aG wheel,video,network $USER
    ```
* **Enable Services:**
    ```bash
    sudo systemctl enable --now NetworkManager
    sudo systemctl enable --now bluetooth
    sudo systemctl enable fstrim.timer
    ```

## 3. Hardware and System Configuration

### 3.1. NVIDIA Drivers

* **Installation:**
    ```bash
    sudo pacman -S nvidia nvidia-utils
    sudo mkinitcpio -P
    nvidia-smi
    ```
* **Wayland Configuration:**
    ```bash
    kwriteconfig6 --file kwinrc --group Xwayland --key Scale 1.25
    kwriteconfig6 --file kwinrc --group Compositing --key GLCoreProfile false
    kwin_wayland --replace &
    ```
* **X11 Screen Tearing Fix:** Open NVIDIA X Server Settings > Display Configuration > Advanced > Force Composition Pipeline.

### 3.2. CoolerControl (Fan Control)

* **Installation:**
    ```bash
    yay -S coolercontrol coolercontrol-liqctld
    sudo systemctl enable --now coolercontrol-liqctld.service
    sudo systemctl enable --now coolercontrol.service
    sudo usermod -aG coolercontrol $USER
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```
* **ASUS Motherboard Sensor Driver:**
    ```bash
    sudo modprobe it87 force_id=0x8628
    echo "it87 force_id=0x8628" | sudo tee /etc/modprobe.d/it87.conf
    ```

### 3.3. Display and Font Configuration

* **KDE Display Scaling (X11):**
    ```bash
    kwriteconfig6 --file kcmfonts --group General --key forceFontDPI 100
    kwriteconfig6 --file kcmfonts --group General --key fontDPI 100
    kwriteconfig6 --file kcmfonts --group General --key antialiasing 1
    kwriteconfig6 --file kcmfonts --group General --key hinting 1
    kwriteconfig6 --file kcmfonts --group General --key subpixelRendering 1
    ```

### 3.4. Conky (System Monitor)

* **Conky Configuration:** The provided Conky setup includes system information such as CPU, Memory, Storage, GPU, and Network.
    * Example section from `.conkyrc`:
        ```lua
        conky.config = {
            alignment = 'top_right',
            background = false,
            border_width = 1,
            cpu_avg_samples = 2,
            gap_x = 10,
            gap_y = 10,
            minimum_width = 250,
            net_avg_samples = 2,
            override_utf8_locale = true,
            own_window = true,
            own_window_transparent = true,
            own_window_type = 'desktop',
            text_buffer_size = 2048,
            update_interval = 1.0,
            use_xft = true,
            font = 'sans-serif:bold:size=9',
            color1 = 'FFFFFF', -- White
            color2 = 'CCCCCC', -- Light Gray
            template = [[
            ${font sans-serif:bold:size=10}${color1}System${hr 2}
            ${font sans-serif:size=9}${color2}Hostname: ${nodename}
            ${color2}Uptime: ${uptime}
            ${color2}Kernel: ${kernel}
            ${color2}CPU: ${cpu cpu0}% @ ${freq_g} GHz
            Temperature: ${exec sensors | grep 'Tctl' | awk '{print $2}'}

            ${font sans-serif:bold:size=10}${color1}Memory${hr 2}
            ${font sans-serif:size=9}${color2}RAM Usage: ${mem} / ${memmax} (${memperc}%)
            ${font sans-serif:size=9}${color2}Swap Usage: ${swap} / ${swapmax} (${swapperc}%)

            ${font sans-serif:bold:size=10}${color1}Storage${hr 2}
            ${font sans-serif:size=9}${color2}Root: ${fs_used /} / ${fs_size /} (${fs_used_perc /}% used)
            ${font sans-serif:size=9}${color2}Home: ${fs_used /home} / ${fs_size /home} (${fs_used_perc /home}% used)
            ${font sans-serif:size=9}${color2}Windows11: ${fs_used /run/media/ekco/Windows11} / ${fs_size /run/media/ekco/Windows11} (${fs_used_perc /run/media/ekco/Windows11}% used)
            ${font sans-serif:size=9}${color2}Steam: ${fs_used /run/media/ekco/Steam} / ${fs_size /run/media/ekco/Steam} (${fs_used_perc /run/media/ekco/Steam}% used)

            ${font sans-serif:bold:size=10}${color1}GPU ${hr 2}
            ${font sans-serif:size=9}${color2}GPU Temp: ${execi 5 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader}C
            ${color2}GPU Usage: ${execi 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader}%
            ${color2}GPU Mem: ${execi 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader}MiB / ${execi 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader}MiB

            ${font sans-serif:bold:size=10}${color1}Network${hr 2}
            ${font sans-serif:size=9}${color2}IP: ${addr wlan0}
            ${color2}Download: ${downspeed wlan0} / ${totaldown wlan0}
            ${color2}Upload: ${upspeed wlan0} / ${totalup wlan0}
            ]]
        }
        ```
