# 🛡️ Kaia System Security & Diagnostics Audit

**Generated on:** June 12, 2026 00:09 | **Host:** `kaia` | **Status:** Completed

---

## 📊 Quick Health Summary
| Subsystem | Status | Details / Issues |
| :--- | :---: | :--- |
| **System Load & Kernel** | 🟢 PASS (Load: 0.18 / Cores: 12) | Load average vs. CPU cores |
| **Hardware Thermals** | 🟢 PASS (37°C) | Temperature threshold monitoring |
| **Storage & FTL Integrity** | 🟢 PASS (Max disk usage: 59%) | Free space partition check |
| **Drive SMART Health** | 🟢 PASS (SMART Health OK) | Structural health check of sda/sdb/nvme0n1 |
| **GPU & VRAM Compute** | 🟢 PASS (VRAM: 88% - 10908/12288 MiB) | NVIDIA hardware & VRAM load state |
| **Local AI (Ollama)** | 🟢 PASS (Running on port 11434) | Ollama system service & socket validation |
| **Database (PostgreSQL)** | 🟢 PASS (Running on port 5432) | Datastore accessibility check |
| **Network & Firewall (UFW)** | 🟢 PASS (Active) | Host level firewall daemon |
| **Systemd Services** | 🟢 PASS (0 failed services) | Failed service sweep |
| **User Space Limits** | 🟢 PASS (ulimit -n: 524288) | Dynamic file handle ceilings |
| **Package Database** | 🟢 PASS (No db.lck) | DB transaction locks status |

---

## 💻 Hardware & Host Specifications
| Component | Details |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 9600X 6-Core Processor (12 threads) |
| **GPU** | NVIDIA GeForce RTX 3060 |
| **Memory (RAM)** | 30Gi |
| **Display Server / Session** | Wayland (wayland session) |

### Active Compositor Topology
```
KWin support info unavailable
```

---

## 🗄️ Storage & Partition Layout
### Storage Space Utilization (df -h)
| Filesystem | Size | Used | Avail | Use% | Mounted on |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dev` | 16G | 0 | 16G | 0% | `/dev` |
| `run` | 16G | 2.1M | 16G | 1% | `/run` |
| `efivarfs` | 128K | 66K | 58K | 54% | `/sys/firmware/efi/efivars` |
| `/dev/nvme0n1p1` | 817G | 376G | 400G | 49% | `/` |
| `none` | 1.0M | 0 | 1.0M | 0% | `/run/credentials/systemd-journald.service` |
| `/dev/nvme0n1p4` | 1022M | 186M | 837M | 19% | `/boot` |
| `/dev/sda1` | 954G | 560G | 395G | 59% | `/run/media/ekco/Steam` |
| `kaia@192.168.1.212:/home/kaia/kaia_share` | 586G | 95G | 462G | 17% | `/mnt/Satellite` |
| `/dev/sdb2` | 937G | 291G | 599G | 33% | `/run/media/ekco/Backup` |

### SMART Telemetry Summary
- **nvme0n1 (OS):** `SMART overall-health self-assessment test result: PASSED`
- **sda (Steam):** `SMART overall-health self-assessment test result: PASSED`
- **sdb (Backup):** `SMART overall-health self-assessment test result: PASSED`

---

## 🧠 GPU & AI Inference Stack
- **Primary GPU:** NVIDIA GeForce RTX 3060
- **Active VRAM Compute Processes:**
  | PID | Process Name | VRAM Used |
  | :--- | :--- | :--- |
  | `1860262` | python | 104 MiB |
| `2408022` | /usr/local/bin/ollama | 9630 MiB |


---

## ⚙️ Service & Network Topology
### Core Services
| Service | Purpose | Status |
| :--- | :--- | :--- |
| `ollama` | Ollama Service | 🟢 Active |
| `postgresql` | PostgreSQL database server | 🟢 Active |
| `coolercontrold` | CoolerControl Daemon | 🟢 Active |
| `NetworkManager` | Network Manager | 🟢 Active |
| `wpa_supplicant` | WPA supplicant | 🟢 Active |
| `bluetooth` | Bluetooth service | 🟢 Active |
| `sddm` | Simple Desktop Display Manager | 🟢 Active |
| `systemd-timesyncd` | Network Time Synchronization | 🟢 Active |
| `ufw` | CLI Netfilter Manager | 🟢 Active |

### Open Listening Ports
| Port | Process | PID | Bind Address |
| :--- | :--- | :--- | :--- |
| `58261` | firedragon | 2591 | `127.0.0.1:58261` |
| `11988` | coolercontrold | 863 | `127.0.0.1:11988` |
| `11987` | coolercontrold | 863 | `127.0.0.1:11987` |
| `11434` | ollama | 1436 | `127.0.0.1:11434` |
| `44533` | ollama | 2408022 | `127.0.0.1:44533` |
| `34429` | ollama | 1868289 | `127.0.0.1:34429` |
| `5432` | postgres | 1439 | `127.0.0.1:5432` |
| `6463` | vesktop.bin | 7218 | `127.0.0.1:6463` |
| `5432` | postgres | 1439 | `[::1]:5432` |
| `11987` | coolercontrold | 863 | `[::1]:11987` |
| `11988` | coolercontrold | 863 | `[::1]:11988` |

### SSHFS Network Storage Mounts
- **Active Mounts (mount | grep sshfs):**
```
kaia@192.168.1.212:/home/kaia/kaia_share on /mnt/Satellite type fuse.sshfs (rw,nosuid,nodev,noexec,relatime,user_id=0,group_id=0,allow_other,_netdev,x-systemd.automount)
```
- **Systemd Automount Status:**
```
  mnt-Satellite.mount                  loaded    active   mounted /mnt/Satellite
● mnt-Wintermute.mount                 loaded    failed   failed  /mnt/Wintermute
```

---

## 📦 Package Management & Virtual Envs
- **System Package Counts:**
  - `pacman` native packages: **1174**
  - `AUR` (foreign) packages: **13**
  - `Flatpak` application profiles: **12**
  - `Orphans` detected: **0**

### Flatpak Applications
| Application | Application ID | Version |
| :--- | :--- | :--- |
| Pinta | `com.github.PintaProject.Pinta` | 3.1.2 |
| Google Chrome | `com.google.Chrome` | 149.0.7827.53-1 |
| Heroic | `com.heroicgameslauncher.hgl` | v2.22.0 |
| OBS Studio | `com.obsproject.Studio` | 32.1.2 |
| ProtonPlus | `com.vysp3r.ProtonPlus` | 0.5.20 |
| Vesktop | `dev.vencord.Vesktop` | 1.6.5 |
| Kooha | `io.github.seadve.Kooha` | 2.3.2 |
| mpv | `io.mpv.Mpv` | v0.41.0 |
| ProtonUp-Qt | `net.davidotek.pupgui2` | 2.15.0 |
| Lutris | `net.lutris.Lutris` | 0.5.22 |
| FireDragon | `org.garudalinux.firedragon` | v12.9.1 |
| Kdenlive | `org.kde.kdenlive` | 26.04.2 |

### Python virtualenv (~/venv)
- **Python Version:** `Python 3.14.5`
- **Packages:**
```
Package Version
------- -------
pip     25.3
```

---

## 🐚 Scripts & Home Directories
### Large Home Folders
| Target Directory | Size |
| :--- | :--- |
| /home/ekco/Videos | 61G |
| /home/ekco/Games | 50G |
| /home/ekco/github | 43G |
| /home/ekco/Music | 18G |
| /home/ekco/paru | 654M |
| /home/ekco/Pictures | 413M |
| /home/ekco/nltk_data | 68M |
| /home/ekco/auto-heroic-categories | 56M |
| /home/ekco/paru-bin | 25M |
| /home/ekco/Documents | 25M |
| /home/ekco/venv | 13M |
| /home/ekco/aura-bin | 6.9M |
| /home/ekco/Downloads | 2.2M |
| /home/ekco/fgmod | 36K |
| /home/ekco/generate_audit.sh | 20K |

### Scripts in Home Directory
| Script | First execution line | Description hint |
| :--- | :--- | :--- |
| `backup_home.sh` | `STEAM="/run/media/ekco/Steam"` | MOUNT POINTS |
| `bpytop.sh` | `  clear` | Check if bpytop is installed, and install it if it's not |
| `clean_system.sh` | `  clear` | --- Kaia's Comprehensive System Cleanup Script --- |
| `epub-to-md.sh` | `  clear` | --- Configuration --- |
| `fastfetch-lolcat.sh` | `fastfetch \| lolcat` |  |
| `find-file.sh` | `  clear` | Find a file by name (case-insensitive) |
| `free-space.sh` | `  clear` | Display disk space usage |
| `generate_audit.sh` | `OUT="/home/ekco/system_audit.md"` | KAIA SYSTEM AUDIT - generates ~/system_audit.md |
| `mp4-to-gif.sh` | `MAX_SIZE_MB=200` | Cap absurd FPS |
| `music.sh` | `nohup mpv /home/ekco/Music/my_playlist.m3u8 > /dev/null 2>&1 &` | This script launches my playlist in mpv |
| `restart_speech_dispatcher.sh` | `echo "Attempting to stop speech-dispatcher service..."` |  |
| `run_kaia.sh` | `cd ~/github/Kaiacord/` |  |
| `start_dashboard.sh` | `pkill -9 -x kitty` | 2. Launch the 4-window grid with auto-commands |
| `stop_dashboard.sh` | `pkill -9 -f "terminal-1"` | -9 forces an immediate close, -f matches the full launch command |
| `update-system.sh` | `if ! command -v lolcat &>/dev/null; then` | Check if lolcat is installed, and install it if it's not |

### Startup Applications (KDE Autostart)
```
total 8
drwxr-xr-x  2 ekco ekco 4096 Nov 24  2025 .
drwxr-xr-x 47 ekco ekco 4096 Jun 12 00:01 ..
```

### Shell Configuration Traces (~/.bashrc / ~/.zshrc)
```
# Path & Libraries
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/var/lib/flatpak/exports/bin:$HOME/.local/share/flatpak/exports/bin:$PATH"
export PATH="/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export KAIA_DB_USER="kaiauser"
export KAIA_DB_PASS="kaia123"
export KAIA_DB_HOST="localhost"
export KAIA_DB_NAME="kaiadb"
```

---

## 🪵 Log & Diagnostics Interpretation Table
| Log Warning / Message | Mechanical Explanation | Impact |
| :--- | :--- | :--- |
| *None of the known warnings found in recent logs.* | | |


---

## 📝 Verbose Diagnostics & Logs

### 🔍 System Journal Errors & Warnings (Priority 0-3)
```
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:58:39 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 03:27:22 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-91) running -> error (Received error event)
Jun 11 03:27:22 kaia bluetoothd[806]: src/profile.c:ext_io_disconnected() Unable to get io data for Hands-Free Voice gateway: getpeername: Transport endpoint is not connected (107)
Jun 11 03:27:22 kaia pipewire-pulse[1209]: mod.protocol-pulse: 0x55cb762f7f60: card 134 port 0 profiles inconsistent (1 < 2)
Jun 11 03:27:22 kaia pipewire-pulse[1209]: mod.protocol-pulse: 0x55cb762f7f60: card 134 port 1 profiles inconsistent (1 < 2)
Jun 11 03:27:22 kaia pipewire-pulse[1209]: mod.protocol-pulse: 0x55cb762f7f60: card 134 port 0 profiles inconsistent (1 < 2)
Jun 11 03:27:22 kaia pipewire-pulse[1209]: mod.protocol-pulse: 0x55cb762f7f60: card 134 port 1 profiles inconsistent (1 < 2)
Jun 11 04:15:28 kaia bluetoothd[806]: src/service.c:btd_service_connect() a2dp-sink profile connect failed for A1:A7:CB:F6:22:7C: Device or resource busy
Jun 11 04:16:49 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 04:34:24 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 04:47:56 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 04:48:56 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 07:13:57 kaia bluetoothd[806]: src/profile.c:ext_io_disconnected() Unable to get io data for Hands-Free Voice gateway: getpeername: Transport endpoint is not connected (107)
Jun 11 07:13:57 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-88) idle -> error (Received error event)
Jun 11 09:21:07 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 09:22:07 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 11:30:20 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 11:31:20 kaia kernel: Bluetooth: hci0: ACL packet for unknown connection handle 3837
Jun 11 17:28:47 kaia bluetoothd[806]: src/profile.c:ext_io_disconnected() Unable to get io data for Hands-Free Voice gateway: getpeername: Transport endpoint is not connected (107)
Jun 11 17:28:47 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-151) running -> error (Received error event)
Jun 11 17:48:28 kaia bluetoothd[806]: src/profile.c:ext_io_disconnected() Unable to get io data for Hands-Free Voice gateway: getpeername: Transport endpoint is not connected (107)
Jun 11 17:48:28 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-159) running -> error (Received error event)
Jun 11 22:11:33 kaia systemd-coredump[2578059]: Process 2576342 (antigravity-ide) of user 1000 dumped core.
                                                
                                                Stack trace of thread 2576342:
                                                #0  0x000055d877b0de23 n/a (n/a + 0x0)
                                                ELF object binary architecture: AMD x86-64
Jun 11 23:38:46 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 11 23:38:49 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 11 23:38:53 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 11 23:38:56 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 11 23:38:59 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 11 23:39:08 kaia systemd-coredump[2645115]: Process 2644408 (antigravity-ide) of user 1000 dumped core.
                                                
                                                Stack trace of thread 2644408:
                                                #0  0x00005599f4e7de23 n/a (n/a + 0x0)
                                                ELF object binary architecture: AMD x86-64
Jun 11 23:42:46 kaia sudo[2647710]: pam_unix(sudo:auth): conversation failed
Jun 11 23:42:46 kaia sudo[2647710]: pam_unix(sudo:auth): auth could not identify password for [ekco]
Jun 11 23:42:50 kaia sudo[2647761]:     ekco : a password is required ; PWD=/home/ekco/github/System Audit ; USER=root ; COMMAND=/usr/bin/true
Jun 12 00:01:49 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 12 00:01:52 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 12 00:01:55 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 12 00:01:58 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 12 00:02:27 kaia systemd-coredump[2666217]: Process 2645125 (antigravity-ide) of user 1000 dumped core.
                                                
                                                Stack trace of thread 2645125:
                                                #0  0x00007fb86a5c4775 n/a (n/a + 0x0)
                                                ELF object binary architecture: AMD x86-64
Jun 12 00:08:21 kaia systemd[1]: Failed to mount /mnt/Wintermute.
Jun 12 00:08:24 kaia systemd[1]: Failed to mount /mnt/Wintermute.
```

### 🐧 Kernel Warning Messages (dmesg)
```

```

### 💚 NVIDIA Kernel Logs
```
Jun 08 21:33:18 kaia kernel: nvidia: module verification failed: signature and/or required key missing - tainting kernel
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
Jun 11 02:15:35 kaia kernel: [drm:nv_drm_gem_alloc_nvkms_memory_ioctl [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to allocate NVKMS memory for GEM object
```

### 🪵 Ollama Logs (Errors & Warnings)
```
● ollama.service - Ollama Service
     Loaded: loaded (/etc/systemd/system/ollama.service; enabled; preset: disabled)
    Drop-In: /etc/systemd/system/ollama.service.d
             └─override.conf
     Active: active (running) since Mon 2026-06-08 21:33:34 CDT; 3 days ago
 Invocation: bfc86945fa8e4fa89832d86e75dcb0a5
    Process: 1432 ExecStartPre=/usr/bin/nvidia-modprobe -u -c=0 (code=exited, status=0/SUCCESS)
   Main PID: 1436 (ollama)
      Tasks: 58 (limit: 37297)
     Memory: 11.2G (peak: 14.1G, swap: 200.8M, swap peak: 375.6M)
        CPU: 1h 59min 17.274s
     CGroup: /system.slice/ollama.service
             ├─   1436 /usr/local/bin/ollama serve
             ├─1868289 /usr/local/bin/ollama runner --model /var/lib/ollama/.ollama/models/blobs/sha256-7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b --port 34429
             └─2408022 /usr/local/bin/ollama runner --ollama-engine --model /var/lib/ollama/.ollama/models/blobs/sha256-e8ad13eff07a78d89926e9e8b882317d082ef5bf9768ad7b50fcdbbcd63748de --port 44533

Jun 12 00:09:02 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:02 | 200 |       42.37µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:04 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:04 | 200 |       38.47µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:05 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:05 | 200 |       54.44µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:07 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:07 | 200 |      38.609µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:08 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:08 | 200 |       39.24µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:10 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:10 | 200 |      66.909µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:11 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:11 | 200 |       54.89µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:13 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:13 | 200 |       42.51µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:14 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:14 | 200 |        46.6µs |       127.0.0.1 | GET      "/api/ps"
Jun 12 00:09:16 kaia ollama[1436]: [GIN] 2026/06/12 - 00:09:16 | 200 |        36.4µs |       127.0.0.1 | GET      "/api/ps"

Jun 11 11:31:23 kaia ollama[1436]: time=2026-06-11T11:31:23.958-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 11:31:23 kaia ollama[1436]: time=2026-06-11T11:31:23.964-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 12:32:02 kaia ollama[1436]: time=2026-06-11T12:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 12:32:02 kaia ollama[1436]: time=2026-06-11T12:32:02.268-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 13:32:02 kaia ollama[1436]: time=2026-06-11T13:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 13:32:02 kaia ollama[1436]: time=2026-06-11T13:32:02.268-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 13:46:08 kaia ollama[1436]: time=2026-06-11T13:46:08.105-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 13:46:08 kaia ollama[1436]: time=2026-06-11T13:46:08.111-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 14:32:02 kaia ollama[1436]: time=2026-06-11T14:32:02.260-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 14:32:02 kaia ollama[1436]: time=2026-06-11T14:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 15:32:02 kaia ollama[1436]: time=2026-06-11T15:32:02.259-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 15:32:02 kaia ollama[1436]: time=2026-06-11T15:32:02.263-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 15:42:46 kaia ollama[1436]: time=2026-06-11T15:42:46.888-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 15:42:46 kaia ollama[1436]: time=2026-06-11T15:42:46.891-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 16:32:02 kaia ollama[1436]: time=2026-06-11T16:32:02.261-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 16:32:02 kaia ollama[1436]: time=2026-06-11T16:32:02.265-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 17:32:02 kaia ollama[1436]: time=2026-06-11T17:32:02.259-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 17:32:02 kaia ollama[1436]: time=2026-06-11T17:32:02.263-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 18:32:02 kaia ollama[1436]: time=2026-06-11T18:32:02.260-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 18:32:02 kaia ollama[1436]: time=2026-06-11T18:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 19:32:02 kaia ollama[1436]: time=2026-06-11T19:32:02.265-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 19:32:02 kaia ollama[1436]: time=2026-06-11T19:32:02.269-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 20:32:02 kaia ollama[1436]: time=2026-06-11T20:32:02.260-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 20:32:02 kaia ollama[1436]: time=2026-06-11T20:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 21:32:02 kaia ollama[1436]: time=2026-06-11T21:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 21:32:02 kaia ollama[1436]: time=2026-06-11T21:32:02.268-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 22:32:02 kaia ollama[1436]: time=2026-06-11T22:32:02.265-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 22:32:02 kaia ollama[1436]: time=2026-06-11T22:32:02.269-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 23:32:02 kaia ollama[1436]: time=2026-06-11T23:32:02.264-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
Jun 11 23:32:02 kaia ollama[1436]: time=2026-06-11T23:32:02.267-05:00 level=WARN source=runner.go:1219 msg="model does not support caching, setting batch size to context length" batch_size=2048
```

### 🐘 PostgreSQL Logs
```
● postgresql.service - PostgreSQL database server
     Loaded: loaded (/usr/lib/systemd/system/postgresql.service; enabled; preset: disabled)
     Active: active (running) since Mon 2026-06-08 21:33:34 CDT; 3 days ago
 Invocation: bd943f89fc024945b01af8c70f88fc6d
       Docs: man:postgres(1)
    Process: 1433 ExecStartPre=/usr/bin/postgresql-check-db-dir ${PGROOT}/data (code=exited, status=0/SUCCESS)
   Main PID: 1439 (postgres)
      Tasks: 9 (limit: 37297)
     Memory: 73M (peak: 241.3M, swap: 161.1M, swap peak: 161.1M)
        CPU: 18.487s
     CGroup: /system.slice/postgresql.service
             ├─1439 /usr/bin/postgres -D /var/lib/postgres/data
             ├─1464 "postgres: io worker 0"
             ├─1465 "postgres: io worker 1"
             ├─1466 "postgres: io worker 2"
             ├─1467 "postgres: checkpointer "
             ├─1468 "postgres: background writer "
             ├─1470 "postgres: walwriter "
             ├─1471 "postgres: autovacuum launcher "
             └─1472 "postgres: logical replication launcher "

Jun 08 21:33:34 kaia systemd[1]: Starting PostgreSQL database server...
Jun 08 21:33:34 kaia postgres[1439]: 2026-06-08 21:33:34.956 CDT [1439] LOG:  starting PostgreSQL 18.4 on x86_64-pc-linux-gnu, compiled by gcc (GCC) 16.1.1 20260430, 64-bit
Jun 08 21:33:34 kaia postgres[1439]: 2026-06-08 21:33:34.956 CDT [1439] LOG:  listening on IPv6 address "::1", port 5432
Jun 08 21:33:34 kaia postgres[1439]: 2026-06-08 21:33:34.956 CDT [1439] LOG:  listening on IPv4 address "127.0.0.1", port 5432
Jun 08 21:33:34 kaia postgres[1439]: 2026-06-08 21:33:34.961 CDT [1439] LOG:  listening on Unix socket "/run/postgresql/.s.PGSQL.5432"
Jun 08 21:33:34 kaia postgres[1469]: 2026-06-08 21:33:34.973 CDT [1469] LOG:  database system was shut down at 2026-06-08 21:32:15 CDT
Jun 08 21:33:34 kaia postgres[1439]: 2026-06-08 21:33:34.980 CDT [1439] LOG:  database system is ready to accept connections
Jun 08 21:33:34 kaia systemd[1]: Started PostgreSQL database server.
Jun 08 21:38:35 kaia postgres[1467]: 2026-06-08 21:38:35.749 CDT [1467] LOG:  checkpoint starting: time
Jun 08 21:38:35 kaia postgres[1467]: 2026-06-08 21:38:35.797 CDT [1467] LOG:  checkpoint complete: wrote 0 buffers (0.0%), wrote 3 SLRU buffers; 0 WAL file(s) added, 0 removed, 0 recycled; write=0.019 s, sync=0.003 s, total=0.049 s; sync files=2, longest=0.002 s, average=0.002 s; distance=0 kB, estimate=0 kB; lsn=0/1C8B080, redo lsn=0/1C8B028


```

### 💨 CoolerControl Daemon Logs
```
● coolercontrold.service - CoolerControl Daemon
     Loaded: loaded (/usr/lib/systemd/system/coolercontrold.service; enabled; preset: disabled)
     Active: active (running) since Mon 2026-06-08 21:33:24 CDT; 3 days ago
 Invocation: acd7511f23474b3d84e47c182e436410
   Main PID: 863 (coolercontrold)
      Tasks: 7 (limit: 37297)
     Memory: 38.2M (peak: 95.5M, swap: 26.4M, swap peak: 26.4M)
        CPU: 1h 50min 24.002s
     CGroup: /system.slice/coolercontrold.service
             └─863 /usr/bin/coolercontrold

Jun 08 21:33:26 kaia coolercontrold[863]: Initialized Hwmon Devices: {"nct6799":{"driver name":["nct6775"],"temps":["temp1","temp10","temp11","temp12","temp13","temp2","temp3","temp4","temp5","temp7","temp8","temp9"],"locations":["/sys/class/hwmon/hwmon2","/sys/devices/platform/nct6775.656","platform:nct6775"],"driver version":["7.0.10-arch1-1"],"channels":["fan1","fan2","fan3","fan4","fan5","fan6","fan7"]},"mt7921_phy0":{"temps":["temp1"],"channels":[],"locations":["/sys/class/hwmon/hwmon7","/sys/devices/pci0000:00/0000:00:02.1/0000:03:00.0/0000:04:0b.0/0000:09:00.0/ieee80211/phy0"],"driver name":[""],"driver version":["7.0.10-arch1-1"]},"nvme":{"channels":[],"driver version":["7.0.10-arch1-1"],"temps":["temp1","temp2","temp3"],"locations":["/sys/class/hwmon/hwmon0","/sys/devices/pci0000:00/0000:00:01.2/0000:02:00.0/nvme/nvme0"],"driver name":[""]},"spd5118":{"channels":[],"temps":["temp1"],"locations":["/sys/class/hwmon/hwmon6","/sys/devices/pci0000:00/0000:00:14.0/i2c-4/4-0053","i2c:spd5118"],"driver name":["spd5118"],"driver version":["7.0.10-arch1-1"]}}
Jun 08 21:33:26 kaia coolercontrold[863]: Initialized Custom Sensors: []
Jun 08 21:33:26 kaia coolercontrold[863]: Applying all saved device settings
Jun 08 21:33:26 kaia coolercontrold[863]: Successfully applied:: NVIDIA GeForce RTX 3060 | fan1 | Profile: Unmanaged
Jun 08 21:33:26 kaia coolercontrold[863]: stress-ng is not installed. Install it for additional stress test backends.
Jun 08 21:33:26 kaia coolercontrold[863]: Using existing TLS certificates
Jun 08 21:33:26 kaia coolercontrold[863]: Serving HTTP and HTTPS API on 127.0.0.1:11987
Jun 08 21:33:26 kaia coolercontrold[863]: Serving HTTP and HTTPS API on [::1]:11987
Jun 08 21:33:26 kaia coolercontrold[863]: Initialization Complete
Jun 08 21:33:26 kaia coolercontrold[863]: DBUS sleep listener connected.

Jun 08 21:33:26 kaia coolercontrold[863]: Python Environment Error: Python liquidctl system package not detected. If you want liquidctl device support, please make sure the liquidctl package is installed with your distribution's package manager. If not, you may disable liquidctl support to no longer see this message. liqctld exited with a non-zero exit code: 1
```

### 📶 NetworkManager & WiFi Logs
```
● NetworkManager.service - Network Manager
     Loaded: loaded (/usr/lib/systemd/system/NetworkManager.service; enabled; preset: disabled)
     Active: active (running) since Mon 2026-06-08 21:33:24 CDT; 3 days ago
 Invocation: 7460f8d9378a4d9da3544674bc0a6ea5
       Docs: man:NetworkManager(8)
   Main PID: 804 (NetworkManager)
      Tasks: 4 (limit: 37297)
     Memory: 13.7M (peak: 20.4M)
        CPU: 11.232s
     CGroup: /system.slice/NetworkManager.service
             └─804 /usr/bin/NetworkManager --no-daemon

Jun 11 20:24:33 kaia NetworkManager[804]: <info>  [1781227473.7972] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 20:54:06 kaia NetworkManager[804]: <info>  [1781229246.4430] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 21:21:21 kaia NetworkManager[804]: <info>  [1781230881.7971] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 21:33:35 kaia NetworkManager[804]: <info>  [1781231615.4257] dhcp4 (wlan0): state changed new lease, address=192.168.1.209
Jun 11 21:48:51 kaia NetworkManager[804]: <info>  [1781232531.7957] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 22:13:57 kaia NetworkManager[804]: <info>  [1781234037.7965] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 22:39:21 kaia NetworkManager[804]: <info>  [1781235561.7927] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 23:06:08 kaia NetworkManager[804]: <info>  [1781237168.8077] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 11 23:30:51 kaia NetworkManager[804]: <info>  [1781238651.7971] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38
Jun 12 00:00:03 kaia NetworkManager[804]: <info>  [1781240403.7924] dhcp6 (wlan0): state changed new lease, address=2600:1702:6f20:56b0::38

● wpa_supplicant.service - WPA supplicant
     Loaded: loaded (/usr/lib/systemd/system/wpa_supplicant.service; disabled; preset: disabled)
     Active: active (running) since Mon 2026-06-08 21:33:24 CDT; 3 days ago
 Invocation: 82cc282ba56b4b37bb3ae5415c84c313
   Main PID: 858 (wpa_supplicant)
      Tasks: 1 (limit: 37297)
     Memory: 6.6M (peak: 7.2M)
        CPU: 501ms
     CGroup: /system.slice/wpa_supplicant.service
             └─858 /usr/bin/wpa_supplicant -u -s -O /run/wpa_supplicant

Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: SME: Trying to authenticate with 28:74:f5:a3:a7:5c (SSID='2WIRE302' freq=5805 MHz)
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: Trying to associate with 28:74:f5:a3:a7:5c (SSID='2WIRE302' freq=5805 MHz)
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: Associated with 28:74:f5:a3:a7:5c
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: CTRL-EVENT-SUBNET-STATUS-UPDATE status=0
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: CTRL-EVENT-REGDOM-CHANGE init=COUNTRY_IE type=COUNTRY alpha2=US
Jun 08 21:33:33 kaia wpa_supplicant[858]: p2p-dev-wlan0: Channel list changed: 6 GHz was enabled
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: Channel list changed: 6 GHz was enabled
Jun 08 21:33:33 kaia wpa_supplicant[858]: wlan0: CTRL-EVENT-REGDOM-CHANGE init=COUNTRY_IE type=COUNTRY alpha2=US
Jun 08 21:33:34 kaia wpa_supplicant[858]: wlan0: WPA: Key negotiation completed with 28:74:f5:a3:a7:5c [PTK=CCMP GTK=CCMP]
Jun 08 21:33:34 kaia wpa_supplicant[858]: wlan0: CTRL-EVENT-CONNECTED - Connection to 28:74:f5:a3:a7:5c completed [id=0 id_str=]

Jun 08 21:33:24 kaia NetworkManager[804]: <warn>  [1780972404.5063] device (p2p-dev-wlan0): error setting IPv4 forwarding to '0': Resource temporarily unavailable
```

### 🔊 PipeWire / WirePlumber Audio Logs
```
Jun 09 23:15:09 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a55b95e0> failed: failed to activate item: Object activation aborted: proxy destroyed
Jun 09 23:17:28 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a55b95e0> failed: failed to activate item: Object activation aborted: proxy destroyed
Jun 09 23:17:58 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a5595530> failed: <WpSiStandardLink:0x55b0a584f730> link failed: some node was destroyed before the link was created
Jun 10 03:48:05 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-93) running -> error (Received error event)
Jun 10 03:48:05 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 10 03:48:05 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-93) running -> error (Received error event)
Jun 10 06:34:07 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-94) running -> error (Received error event)
Jun 10 06:34:07 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 10 06:34:07 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-94) running -> error (Received error event)
Jun 10 07:17:32 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a55b95e0> failed: failed to activate item: Object activation aborted: proxy destroyed
Jun 10 07:19:29 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a5595530> failed: <WpSiStandardLink:0x55b0a5cc5c20> link failed: 1 of 1 PipeWire links failed to activate
Jun 10 08:07:30 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-106) running -> error (Received error event)
Jun 10 08:07:30 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 10 08:07:30 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-106) running -> error (Received error event)
Jun 10 09:26:30 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-133) running -> error (Received error event)
Jun 10 09:26:30 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 10 09:26:30 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-133) running -> error (Received error event)
Jun 11 03:27:22 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-91) running -> error (Received error event)
Jun 11 03:27:22 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 11 03:27:22 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-91) running -> error (Received error event)
Jun 11 07:13:57 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-88) running -> error (Received error event)
Jun 11 07:13:57 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 11 07:13:57 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-88) idle -> error (Received error event)
Jun 11 17:28:47 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-151) running -> error (Received error event)
Jun 11 17:28:47 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 11 17:28:47 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-151) running -> error (Received error event)
Jun 11 17:48:28 kaia wireplumber[899]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-159) running -> error (Received error event)
Jun 11 17:48:28 kaia wireplumber[899]: spa.bluez5: Failure in Bluetooth audio transport /org/bluez/hci0/dev_A1_A7_CB_F6_22_7C/sep2/fd0
Jun 11 17:48:28 kaia pipewire[897]: pw.node: (bluez_output.A1_A7_CB_F6_22_7C.1-159) running -> error (Received error event)
Jun 11 23:43:53 kaia wireplumber[899]: wp-event-dispatcher: <WpAsyncEventHook:0x55b0a5595530> failed: <WpSiStandardLink:0x55b0a5ca7a50> link failed: 1 of 1 PipeWire links failed to activate
```

### 🔗 SSHFS Storage Logs
```

```

### 🧱 Out-Of-Memory (OOM) Kernel Traces
```
Jun 11 02:14:08 kaia kernel: NVRM: GPU0 nvAssertOkFailedNoLog: Assertion failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from pmaAllocatePages(pMemReserveInfo->pPma, pageSize / PMA_CHUNK_SIZE_64K, PMA_CHUNK_SIZE_64K, &allocOptions, &pageBegin) @ pool_alloc.c:270
Jun 11 02:14:08 kaia kernel: NVRM: GPU0 nvAssertOkFailedNoLog: Assertion failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from rmMemPoolReserve(pCtxBufPool->pMemPool[i], totalSize[i], 0) @ ctx_buf_pool.c:315
Jun 11 02:14:08 kaia kernel: NVRM: GPU0 nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from ctxBufPoolReserve(pGpu, pKernelChannelGroup->pCtxBufPool, bufInfoList, bufCount) @ kernel_channel_group_api.c:558
Jun 11 02:14:23 kaia kernel: NVRM: GPU0 nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1338
Jun 11 02:14:23 kaia kernel: NVRM: GPU0 nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from rmStatus @ system_mem.c:353
```

### 💾 ZRAM / Swap Block Compression Telemetry
```
NAME       ALGORITHM DISKSIZE DATA  COMPR  TOTAL STREAMS MOUNTPOINT
/dev/zram0 zstd            4G   2G 302.3M 322.5M         [SWAP]

NAME       TYPE      SIZE USED PRIO
/dev/zram0 partition   4G 2.5G  100
```

### 📊 Top Memory Consuming Processes
```
    PID    PPID CMD                         %MEM %CPU
1868289    1436 /usr/local/bin/ollama runne  6.1  0.3
1860262  882863 python Kaiacord.py           5.8  0.8
2408022    1436 /usr/local/bin/ollama runne  4.6  0.1
 882813     881 kitty --title terminal-3 -e  4.5  4.6
   2591    2590 /app/lib/firedragon/firedra  2.6  5.1
 808364    2666 /app/lib/firedragon/firedra  2.4  2.4
1860335 1860262 python Kaiacord.py           2.4  0.7
1860330 1860262 python Kaiacord.py           2.4  0.1
2411378    2666 /app/lib/firedragon/firedra  2.1  0.7
   7336    7242 /app/bin/vesktop/vesktop.bi  1.6  3.5
2624484    2666 /app/lib/firedragon/firedra  1.3  1.1
   2789    2666 /app/lib/firedragon/firedra  1.1  0.1
   1080     881 /usr/bin/plasmashell --no-r  1.0  0.2
    966     960 /usr/bin/kwin_wayland --way  0.7  3.8
```

### 🔥 UFW Firewall Block Logs
```
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), disabled (routed)
New profiles: skip

To                         Action      From
--                         ------      ----
22/tcp                     LIMIT IN    Anywhere                  
22/tcp (v6)                LIMIT IN    Anywhere (v6)             

Jun 11 23:26:38 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=48790 DF PROTO=2 
Jun 11 23:28:43 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=20432 DF PROTO=2 
Jun 11 23:30:48 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=44030 DF PROTO=2 
Jun 11 23:32:53 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=24313 DF PROTO=2 
Jun 11 23:34:58 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=26317 DF PROTO=2 
Jun 11 23:37:03 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=32538 DF PROTO=2 
Jun 11 23:39:08 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=36985 DF PROTO=2 
Jun 11 23:41:13 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=62086 DF PROTO=2 
Jun 11 23:43:18 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=22026 DF PROTO=2 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:86:dd SRC=2600:1702:6f20:56b0:0000:0000:0000:0001 DST=2600:1702:6f20:56b0:0000:0000:0000:0038 LEN=84 TC=40 HOPLIMIT=64 FLOWLBL=0 PROTO=UDP SPT=53 DPT=45674 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=192.168.1.209 LEN=64 TOS=0x08 PREC=0x20 TTL=64 ID=6268 PROTO=UDP SPT=53 DPT=57263 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:86:dd SRC=2600:1702:6f20:56b0:0000:0000:0000:0001 DST=2600:1702:6f20:56b0:0000:0000:0000:0038 LEN=84 TC=40 HOPLIMIT=64 FLOWLBL=0 PROTO=UDP SPT=53 DPT=45674 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=192.168.1.209 LEN=64 TOS=0x08 PREC=0x20 TTL=64 ID=6269 PROTO=UDP SPT=53 DPT=57263 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:86:dd SRC=2600:1702:6f20:56b0:0000:0000:0000:0001 DST=2600:1702:6f20:56b0:0000:0000:0000:0038 LEN=84 TC=40 HOPLIMIT=64 FLOWLBL=0 PROTO=UDP SPT=53 DPT=50856 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=192.168.1.209 LEN=64 TOS=0x08 PREC=0x20 TTL=64 ID=6270 PROTO=UDP SPT=53 DPT=44159 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:86:dd SRC=2600:1702:6f20:56b0:0000:0000:0000:0001 DST=2600:1702:6f20:56b0:0000:0000:0000:0038 LEN=84 TC=40 HOPLIMIT=64 FLOWLBL=0 PROTO=UDP SPT=53 DPT=50856 LEN=44 
Jun 11 23:45:21 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=192.168.1.209 LEN=64 TOS=0x08 PREC=0x20 TTL=64 ID=6271 PROTO=UDP SPT=53 DPT=44159 LEN=44 
Jun 11 23:45:23 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=4578 DF PROTO=2 
Jun 11 23:47:28 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=58631 DF PROTO=2 
Jun 11 23:49:33 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=54656 DF PROTO=2 
Jun 11 23:51:38 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=32400 DF PROTO=2 
Jun 11 23:53:42 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=18421 DF PROTO=2 
Jun 11 23:55:48 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=47234 DF PROTO=2 
Jun 11 23:57:53 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=23538 DF PROTO=2 
Jun 11 23:59:58 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=2295 DF PROTO=2 
Jun 12 00:02:03 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=64198 DF PROTO=2 
Jun 12 00:02:19 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=3c:0a:f3:32:13:05:28:74:f5:a3:a7:52:08:00 SRC=34.107.243.93 DST=192.168.1.209 LEN=76 TOS=0x00 PREC=0x00 TTL=122 ID=30913 PROTO=TCP SPT=443 DPT=47556 WINDOW=1044 RES=0x00 ACK PSH URGP=0 
Jun 12 00:04:08 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=12440 DF PROTO=2 
Jun 12 00:06:13 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=28718 DF PROTO=2 
Jun 12 00:08:18 kaia kernel: [UFW BLOCK] IN=wlan0 OUT= MAC=01:00:5e:00:00:01:28:74:f5:a3:a7:52:08:00 SRC=192.168.1.254 DST=224.0.0.1 LEN=32 TOS=0x00 PREC=0xC0 TTL=1 ID=20215 DF PROTO=2 
```

### 🚨 Package File Integrity Verification
```
All packages verified intact.
```

### ⏱️ Systemd Timer Execution Schedules
```
NEXT                                 LEFT LAST                              PASSED UNIT                             ACTIVATES
Fri 2026-06-12 00:41:03 CDT         31min Thu 2026-06-11 04:36:19 CDT      19h ago man-db.timer                     man-db.service
Fri 2026-06-12 21:49:01 CDT           21h Thu 2026-06-11 21:49:01 CDT 2h 20min ago systemd-tmpfiles-clean.timer     systemd-tmpfiles-clean.service
Sat 2026-06-13 00:00:00 CDT           23h Fri 2026-06-12 00:00:03 CDT     9min ago shadow.timer                     shadow.service
Mon 2026-06-15 00:48:32 CDT        3 days Mon 2026-06-08 00:44:23 CDT            - fstrim.timer                     fstrim.service
Fri 2026-06-19 16:21:02 CDT 1 week 0 days Tue 2026-06-09 14:46:43 CDT   2 days ago archlinux-keyring-wkd-sync.timer archlinux-keyring-wkd-sync.service

5 timers listed.
```

### ⚙️ Pacman Mirrorlist & Cache Pruning Dry-Run
**Active Pacman Mirrors (Top 10):**
```


Server = https://frankfurt.mirror.pkgbuild.com/$repo/os/$arch
Server = https://at.arch.mirror.kescher.at/$repo/os/$arch
Server = https://mirror.lcarilla.de/archlinux/$repo/os/$arch
Server = https://mirror.ubrco.de/archlinux/$repo/os/$arch
Server = https://nl.arch.niranjan.co/$repo/os/$arch
```

**Reclaimable Cache Space (paccache -dk 3):**
```
==> no candidate packages found for pruning
```

### 🎛️ Hardware Thermal Sensor Baseline (lm_sensors)
```
spd5118-i2c-4-53
Adapter: SMBus PIIX4 adapter port 0 at 0b00
temp1:        +37.0°C  (low  =  +0.0°C, high = +55.0°C)
                       (crit low =  +0.0°C, crit = +85.0°C)

nct6799-isa-0290
Adapter: ISA adapter
in0:                            1.24 V  (min =  +0.00 V, max =  +1.74 V)
in1:                          984.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in2:                            3.39 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in3:                            3.26 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in4:                          1000.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in5:                            1.03 V  (min =  +0.00 V, max =  +0.00 V)
in6:                          736.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in7:                            3.39 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in8:                            3.28 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in9:                            3.26 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in10:                           1.38 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in11:                           1.12 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in12:                           1.03 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in13:                         464.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in14:                           2.04 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in15:                         872.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in16:                           1.82 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in17:                           1.28 V  (min =  +0.00 V, max =  +0.00 V)
fan1:                          634 RPM  (min =    0 RPM)
fan2:                          765 RPM  (min =    0 RPM)
fan3:                            0 RPM  (min =    0 RPM)
fan4:                            0 RPM  (min =    0 RPM)
fan5:                            0 RPM  (min =    0 RPM)
fan6:                            0 RPM  (min =    0 RPM)
fan7:                            0 RPM  (min =    0 RPM)
SYSTIN:                        +33.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +125.0°C)  sensor = thermistor
CPUTIN:                        +37.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +125.0°C)  sensor = thermistor
AUXTIN0:                       +41.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +125.0°C)  sensor = thermistor
AUXTIN1:                        +6.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +125.0°C)  sensor = thermistor
AUXTIN2:                       +19.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +100.0°C)  sensor = thermistor
AUXTIN3:                       -62.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +125.0°C)  sensor = thermistor
AUXTIN4:                       +24.0°C  (high = +80.0°C, hyst = +75.0°C)
                                        (crit = +100.0°C)
PECI/TSI Agent 0 Calibration:  +51.0°C  (high = +80.0°C, hyst = +75.0°C)
AUXTIN5:                       +12.0°C  
PCH_CHIP_CPU_MAX_TEMP:          +0.0°C  
PCH_CHIP_TEMP:                  +0.0°C  
PCH_CPU_TEMP:                   +0.0°C  
TSI0_TEMP:                     +62.0°C  
pwm1:                              62%  (mode = pwm)
pwm2:                              62%  (mode = pwm)
pwm3:                              96%  (mode = pwm)
pwm4:                              96%  (mode = pwm)
pwm6:                              96%  (mode = pwm)
pwm7:                             128%  (mode = pwm)
intrusion0:                   ALARM
intrusion1:                   OK
beep_enable:                  disabled

nvme-pci-0200
Adapter: PCI adapter
Composite:    +40.9°C  (low  = -273.1°C, high = +81.8°C)
                       (crit = +84.8°C)
Sensor 1:     +40.9°C  (low  = -273.1°C, high = +65261.8°C)
Sensor 2:     +42.9°C  (low  = -273.1°C, high = +65261.8°C)

mt7921_phy0-pci-0900
Adapter: PCI adapter
temp1:        +46.0°C  

spd5118-i2c-4-51
Adapter: SMBus PIIX4 adapter port 0 at 0b00
temp1:        +37.0°C  (low  =  +0.0°C, high = +55.0°C)
                       (crit low =  +0.0°C, crit = +85.0°C)

k10temp-pci-00c3
Adapter: PCI adapter
Tctl:         +63.5°C  
Tccd1:        +40.0°C  

amdgpu-pci-0c00
Adapter: PCI adapter
vddgfx:        1.24 V  
vddnb:         1.24 V  
edge:         +47.0°C  
PPT:           8.00 mW 
sclk:         600 MHz 
```

### 💽 High-Resolution I/O Saturation (5-second snapshot)
```
Total DISK READ :       0.00 B/s | Total DISK WRITE :       0.00 B/s
Actual DISK READ:       0.00 B/s | Actual DISK WRITE:       0.00 B/s
    PID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN      IO    COMMAND
Total DISK READ :       0.00 B/s | Total DISK WRITE :       3.96 K/s
Actual DISK READ:       0.00 B/s | Actual DISK WRITE:       0.00 B/s
    PID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN      IO    COMMAND
    362 be/4 root        0.00 B/s    3.96 K/s  ?unavailable?  systemd-journald
Total DISK READ :       0.00 B/s | Total DISK WRITE :       0.00 B/s
Actual DISK READ:       0.00 B/s | Actual DISK WRITE:       0.00 B/s
    PID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN      IO    COMMAND
Total DISK READ :       0.00 B/s | Total DISK WRITE :       3.94 K/s
Actual DISK READ:       0.00 B/s | Actual DISK WRITE:     725.86 K/s
    PID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN      IO    COMMAND
    362 be/4 root        0.00 B/s    3.94 K/s  ?unavailable?  systemd-journald
Total DISK READ :       0.00 B/s | Total DISK WRITE :     130.30 K/s
Actual DISK READ:       0.00 B/s | Actual DISK WRITE:       0.00 B/s
    PID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN      IO    COMMAND
    362 be/4 root        0.00 B/s  130.30 K/s  ?unavailable?  systemd-journald
```

### 📦 Explicit AUR Package Footprints
```
Name            : antigravity-ide
Description     : An agentic development platform from Google, evolving the IDE into the agent-first era
Name            : coolercontrol-bin
Description     : A program to monitor and control your cooling devices (binary release)
Name            : coolercontrold-bin
Description     : A program to monitor and control your cooling devices: daemon (binary release)
Name            : endcord
Description     : Feature rich Discord TUI client.
Name            : gtk2
Description     : GObject-based multi-platform GUI toolkit (legacy)
Name            : lib32-gst-plugins-base-libs
Description     : Multimedia graph framework (32-bit) - base
Name            : lib32-gstreamer
Description     : Multimedia graph framework (32-bit) - core
Name            : lib32-gtk2
Description     : GObject-based multi-platform GUI toolkit (legacy, 32-bit)
Name            : maplemono-ttf
Description     : Open source monospace font with round corner, ligatures and Nerd-Font for IDE and command line
Name            : openrgb-git
Description     : Open source RGB lighting control that doesn't depend on manufacturer software
Name            : paru
Description     : Feature packed AUR helper
Name            : python312
Description     : Major release 3.12 of the Python high-level programming language
Name            : unimatrix-git
Description     : Python script to simulate the display from "The Matrix" in terminal. Uses half-width katakana unicode characters by default, but can use custom character sets.
```

### 🐧 Loaded Kernel Modules (lsmod)
```
842_compress           24576  1 zram
842_decompress         20480  1 zram
aesni_intel           106496  3
af_alg                 32768  6 algif_hash,algif_skcipher
algif_hash             16384  1
algif_skcipher         12288  1
amd_atl                61440  1
amdgpu              16883712  1
amdxcp                 12288  1 amdgpu
asus_armoury          118784  0
asus_wmi              118784  2 asus_armoury,eeepc_wmi
bluetooth            1196032  34 btrtl,btmtk,btintel,btbcm,bnep,btusb,rfcomm
bnep                   36864  2
btbcm                  24576  1 btusb
btintel                73728  1 btusb
btmtk                  32768  1 btusb
btrtl                  32768  1 btusb
btusb                  86016  0
ccm                    24576  6
ccp                   221184  1 kvm_amd
cec                    98304  2 drm_display_helper,amdgpu
cfg80211             1507328  4 mt76,mac80211,mt7921_common,mt76_connac_lib
cmac                   12288  4
crypto_user            16384  0
drm_buddy              28672  1 amdgpu
drm_display_helper    290816  1 amdgpu
drm_exec               12288  1 amdgpu
drm_panel_backlight_quirks    12288  1 amdgpu
drm_suballoc_helper    16384  1 amdgpu
drm_ttm_helper         20480  3 amdgpu,nvidia_drm
eeepc_wmi              12288  0
fat                   114688  1 vfat
ff_memless             24576  1 hid_microsoft
firmware_attributes_class    12288  1 asus_armoury
ghash_clmulni_intel    12288  0
gpio_amdpt             16384  0
gpio_generic           24576  1 gpio_amdpt
gpu_sched              73728  1 amdgpu
hid_microsoft          16384  0
hkdf                   12288  1 nvme_auth
hwmon_vid              12288  1 nct6775
i2c_algo_bit           24576  1 amdgpu
i2c_dev                28672  0
i2c_piix4              40960  0
i2c_smbus              20480  1 i2c_piix4
igc                   221184  0
inet_diag              20480  2 tcp_diag,udp_diag
intel_rapl_common      53248  1 intel_rapl_msr
intel_rapl_msr         20480  0
ip6t_REJECT            12288  1
ip6t_rt                16384  3
ipt_REJECT             12288  1
irqbypass              16384  1 kvm
joydev                 28672  0
k10temp                12288  0
kvm                  1474560  1 kvm_amd
kvm_amd               258048  0
libarc4                12288  1 mac80211
lz4_compress           24576  1 zram
lz4hc_compress         20480  1 zram
mac80211             1757184  4 mt792x_lib,mt76,mt7921_common,mt76_connac_lib
mac_hid                12288  0
mc                     94208  1 snd_usb_audio
Module                  Size  Used by
mousedev               28672  0
mt76                  155648  4 mt792x_lib,mt7921e,mt7921_common,mt76_connac_lib
mt76_connac_lib        98304  3 mt792x_lib,mt7921e,mt7921_common
mt7921_common          94208  1 mt7921e
mt7921e                28672  0
mt792x_lib             65536  2 mt7921e,mt7921_common
nct6775                40960  0
nct6775_core           86016  1 nct6775
nf_conntrack          200704  1 xt_conntrack
nf_defrag_ipv4         12288  1 nf_conntrack
nf_defrag_ipv6         24576  1 nf_conntrack
nf_log_syslog          20480  10
nfnetlink              20480  2 nft_compat,nf_tables
nf_reject_ipv4         12288  1 ipt_REJECT
nf_reject_ipv6         20480  1 ip6t_REJECT
nf_tables             401408  628 nft_compat,nft_limit
nft_compat             20480  125
nft_limit              16384  13
ntsync                 20480  0
nvidia              18190336  1388 nvidia_uvm,nvidia_modeset
nvidia_drm            167936  158
nvidia_modeset       1929216  66 nvidia_drm
nvidia_uvm           2449408  12
nvme                   77824  2
nvme_auth              32768  1 nvme_core
nvme_core             278528  3 nvme
nvme_keyring           20480  1 nvme_core
pcspkr                 12288  0
pkcs8_key_parser       12288  0
platform_profile       20480  1 asus_wmi
pps_core               32768  1 ptp
ptp                    53248  1 igc
rapl                   20480  0
rfcomm                110592  4
rfkill                 45056  10 mt7921e,asus_wmi,bluetooth,cfg80211
snd                   159744  27 snd_seq,snd_seq_device,snd_hda_codec_hdmi,snd_hwdep,snd_hda_intel,snd_usb_audio,snd_usbmidi_lib,snd_hda_codec,snd_timer,snd_ump,snd_pcm,snd_rawmidi
snd_hda_codec         221184  4 snd_hda_codec_hdmi,snd_hda_intel,snd_hda_codec_nvhdmi,snd_hda_codec_atihdmi
snd_hda_codec_atihdmi    20480  1
snd_hda_codec_hdmi     61440  2 snd_hda_codec_nvhdmi,snd_hda_codec_atihdmi
snd_hda_codec_nvhdmi    16384  1
snd_hda_core          147456  4 snd_hda_codec_hdmi,snd_hda_intel,snd_hda_codec,snd_hda_codec_atihdmi
snd_hda_intel          73728  2
snd_hrtimer            12288  1
snd_hwdep              24576  2 snd_usb_audio,snd_hda_codec
snd_intel_dspcfg       49152  1 snd_hda_intel
snd_intel_sdw_acpi     16384  1 snd_intel_dspcfg
snd_pcm               221184  6 snd_hda_codec_hdmi,snd_hda_intel,snd_usb_audio,snd_hda_codec,snd_hda_core
snd_rawmidi            57344  2 snd_usbmidi_lib,snd_ump
snd_seq               135168  7 snd_seq_dummy
snd_seq_device         16384  3 snd_seq,snd_ump,snd_rawmidi
snd_seq_dummy          12288  0
snd_timer              57344  3 snd_seq,snd_hrtimer,snd_pcm
snd_ump                40960  1 snd_usb_audio
snd_usb_audio         610304  5
snd_usbmidi_lib        53248  1 snd_usb_audio
soundcore              16384  1 snd
sp5100_tco             20480  0
sparse_keymap          12288  1 asus_wmi
spd5118                16384  0
tcp_diag               20480  0
ttm                   126976  2 amdgpu,drm_ttm_helper
udp_diag               12288  0
uhid                   24576  1
uinput                 28672  0
vfat                   28672  1
video                  81920  3 asus_wmi,amdgpu,nvidia_modeset
wmi                    36864  4 video,asus_armoury,asus_wmi,wmi_bmof
wmi_bmof               12288  0
x_tables               65536  11 xt_conntrack,nft_compat,xt_LOG,xt_tcpudp,xt_addrtype,xt_recent,ip6t_rt,ipt_REJECT,xt_limit,xt_hl,ip6t_REJECT
xt_addrtype            12288  4
xt_conntrack           12288  20
xt_hl                  12288  22
xt_limit               12288  0
xt_LOG                 16384  10
xt_recent              24576  4
xt_tcpudp              20480  60
zram                   73728  1
```

### 🕒 Recently Installed Packages (last 30 days)
```
[2026-04-13T09:23:23-0500] [ALPM] installed bpytop (1.0.68-2)
[2026-04-14T05:30:07-0500] [ALPM] installed nftables (1:1.1.6-3)
[2026-05-01T09:13:31-0500] [ALPM] installed appstream-glib (0.8.3-4)
[2026-05-06T22:46:45-0500] [ALPM] installed gcc15-libs (15.2.1+r934+gbcd3e3ff5aa7-1)
[2026-05-06T22:46:45-0500] [ALPM] installed gcc15 (15.2.1+r934+gbcd3e3ff5aa7-1)
[2026-05-09T16:58:02-0500] [ALPM] installed cccl (3.3.3-3)
[2026-05-11T14:26:52-0500] [ALPM] installed cpio (2.15-3)
[2026-05-11T14:26:52-0500] [ALPM] installed libcbor (0.14.0-1)
[2026-05-11T14:26:52-0500] [ALPM] installed libfido2 (1.17.0-1)
[2026-05-14T12:25:00-0500] [ALPM] installed appstream-glib (0.8.3-4)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-gpm (1.20.7.r38.ge82d1a6-2)
[2026-05-14T12:25:01-0500] [ALPM] installed aalib (1.4rc5-19)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-aalib (1.4rc5-5)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-cdparanoia (10.2-5)
[2026-05-14T12:25:01-0500] [ALPM] installed jack2 (1.9.22-2)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libsamplerate (0.2.2-3)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-jack2 (1.9.22-2)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libraw1394 (2.1.2-5)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libavc1394 (0.5.4-5)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-imlib2 (1.12.6-1)
[2026-05-14T12:25:01-0500] [ALPM] installed libcaca (0.99.beta20-7)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libcaca (0.99.beta20-2)
[2026-05-14T12:25:01-0500] [ALPM] installed libdv (1.0.0-12)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-popt (1.19-2)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libdv (1.0.0-9)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libiec61883 (1.2.0-5)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libtheora (1.2.0-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-speex (1.2.1-2)
[2026-05-14T12:25:01-0500] [ALPM] installed libshout (1:2.4.6-5)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libshout (1:2.4.6-4)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-duktape (2.7.0-7)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libproxy (0.5.12-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-glib-networking (1:2.80.1-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libsoup3 (3.6.6-2)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-libvpx (1.16.0-2)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-mpg123 (1.33.5-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-rust-libs (1:1.95.0-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-taglib (2.1.1-1)
[2026-05-14T12:25:01-0500] [ALPM] installed twolame (0.4.0-4)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-twolame (0.4.0-3)
[2026-05-14T12:25:01-0500] [ALPM] installed wavpack (5.9.0-1)
[2026-05-14T12:25:01-0500] [ALPM] installed lib32-wavpack (5.9.0-1)
[2026-05-14T12:25:01-0500] [ALPM] installed ninja (1.13.2-3)
[2026-05-14T12:25:01-0500] [ALPM] installed python-tqdm (4.67.3-1)
[2026-05-14T12:25:01-0500] [ALPM] installed meson (1.11.1-1)
[2026-05-14T12:25:01-0500] [ALPM] installed wayland-protocols (1.48-1)
[2026-05-14T12:25:01-0500] [ALPM] installed xorg-server-xvfb (21.1.22-2)
[2026-05-31T18:59:01-0500] [ALPM] installed antigravity-ide (2.0.3-1)
[2026-06-07T23:20:35-0500] [ALPM] installed sshfs (3.7.6-1)
[2026-06-08T20:14:34-0500] [ALPM] installed uv (0.11.19-1)
```

### 🗝️ Pacman Keyring & Database Transaction Locks
```
No stale transaction locks found.
```

### 📼 Full Hardware Block Layout (lsblk)
```

```

### 📂 Full Disk Partition Table (fdisk)
```
/dev/sda1   2048 2000408575 2000406528 953.9G Microsoft basic data
/dev/sdb1     2048    2099199    2097152     1G EFI System
/dev/sdb2  2099200 2000406527 1998307328 952.9G Linux root (x86-64)
/dev/nvme0n1p1 211812352 1953523711 1741711360 830.5G Linux root (x86-64)
/dev/nvme0n1p4 209715200  211812351    2097152     1G EFI System
```

### 📟 Full NVIDIA System Status (nvidia-smi)
```
Fri Jun 12 00:09:15 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        On  |   00000000:01:00.0  On |                  N/A |
|  0%   49C    P5             40W /  170W |   10908MiB /  12288MiB |      7%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             966      G   /usr/bin/kwin_wayland                    59MiB |
|    0   N/A  N/A            1029      G   /usr/bin/Xwayland                         2MiB |
|    0   N/A  N/A            1063      G   /usr/bin/ksmserver                        2MiB |
|    0   N/A  N/A            1065      G   /usr/bin/kded6                            2MiB |
|    0   N/A  N/A            1080      G   /usr/bin/plasmashell                     82MiB |
|    0   N/A  N/A            1122      G   /usr/bin/kaccess                          2MiB |
|    0   N/A  N/A            1123      G   ...it-kde-authentication-agent-1          2MiB |
|    0   N/A  N/A            1273      G   /usr/bin/kwalletd6                        2MiB |
|    0   N/A  N/A            1296      G   /usr/bin/ksecretd                         2MiB |
|    0   N/A  N/A            1318      G   /usr/bin/kwalletmanager5                  2MiB |
|    0   N/A  N/A            1374      G   /usr/lib/DiscoverNotifier                 2MiB |
|    0   N/A  N/A            1391      G   /usr/lib/xdg-desktop-portal-kde           2MiB |
|    0   N/A  N/A            2591      G   /app/lib/firedragon/firedragon          215MiB |
|    0   N/A  N/A            7275      G   /app/bin/vesktop/vesktop.bin            235MiB |
|    0   N/A  N/A           12120      G   /usr/bin/krunner                          7MiB |
|    0   N/A  N/A           13638      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A           13639      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A          208230      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A          831466      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A          837325      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A          882811      G   kitty                                    13MiB |
|    0   N/A  N/A          882812      G   kitty                                    16MiB |
|    0   N/A  N/A          882813      G   kitty                                    13MiB |
|    0   N/A  N/A          882814      G   kitty                                    16MiB |
|    0   N/A  N/A         1047362      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         1047363      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         1179728      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         1705620      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         1860262      C   python                                  104MiB |
|    0   N/A  N/A         2408022      C   /usr/local/bin/ollama                  9630MiB |
|    0   N/A  N/A         2578531      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         2645392      G   ...lib/drkonqi-coredump-launcher          2MiB |
|    0   N/A  N/A         2666535      G   ...lib/drkonqi-coredump-launcher          2MiB |
+-----------------------------------------------------------------------------------------+
```

### ⏱️ SSD TRIM Timer Detail
```
● fstrim.timer - Discard unused filesystem blocks once a week
     Loaded: loaded (/usr/lib/systemd/system/fstrim.timer; enabled; preset: disabled)
     Active: active (waiting) since Mon 2026-06-08 21:33:23 CDT; 3 days ago
 Invocation: 80a95f2ddac646af81ef6ff43ba7609f
    Trigger: Mon 2026-06-15 00:48:32 CDT; 3 days left
   Triggers: ● fstrim.service
       Docs: man:fstrim

Jun 08 21:33:23 kaia systemd[1]: Started Discard unused filesystem blocks once a week.
```

### 💀 Failed Systemd Units Detail
```
  UNIT                 LOAD   ACTIVE SUB    DESCRIPTION
● mnt-Wintermute.mount loaded failed failed /mnt/Wintermute

Legend: LOAD   → Reflects whether the unit definition was properly loaded.
        ACTIVE → The high-level unit activation state, i.e. generalization of SUB.
        SUB    → The low-level unit activation state, values depend on unit type.

1 loaded units listed.
```

---
*Report generated automatically by `generate_audit.sh` on June 12, 2026 00:09.*
