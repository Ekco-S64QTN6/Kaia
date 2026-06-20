# The Ultimate Linux Command and Development Handbook

This document provides a comprehensive reference for system administration, development workflows, and advanced shell configurations on Linux, tailored for a developer's needs.

---

## Table of Contents
1.  [Package Management](#1-package-management)
2.  [System Information & Monitoring](#2-system-information--monitoring)
3.  [File & Directory Management](#3-file--directory-management)
4.  [Text Processing & Manipulation](#4-text-processing--manipulation)
5.  [User & Permission Management](#5-user--permission-management)
6.  [Process Management](#6-process-management)
7.  [Networking](#7-networking)
8.  [Development Workflows: Git & Node.js](#8-development-workflows-git--nodejs)
9.  [Multimedia with FFmpeg](#9-multimedia-with-ffmpeg)
10. [Shell Customization & Scripting](#10-shell-customization--scripting)

---

## 1. Package Management

Your system uses `pacman` (Arch) and `apt` (Debian/Ubuntu).

### **Pacman (Arch Linux)**
*   **Full System Upgrade:**
    ```bash
    sudo pacman -Syu
    ```
*   **Install a Package:**
    ```bash
    sudo pacman -S <package_name>
    ```
*   **Remove a Package and its Dependencies:**
    ```bash
    sudo pacman -Rns <package_name>
    ```
*   **Search for a Package:**
    ```bash
    pacman -Ss <query>
    ```
*   **Query Installed Packages:**
    ```bash
    pacman -Qs <query>
    ```

### **Yay (AUR Helper)**
*   **Install an AUR Package:**
    ```bash
    yay -S <package_name>
    ```
*   **Combined System and AUR Upgrade:**
    ```bash
    yay -Syu
    ```

### **APT (Debian/Ubuntu)**
*   **Full Update and Cleanup:**
    ```bash
    sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y && sudo apt autoclean
    ```
*   **Install/Remove a Package:**
    ```bash
    sudo apt install <package_name>
    sudo apt remove <package_name>
    ```

---

## 2. System Information & Monitoring

*   **System Details (Kernel, OS):**
    ```bash
    uname -a
    ```
*   **CPU Information:**
    ```bash
    lscpu
    ```
*   **Disk Space Usage (Human-readable):**
    ```bash
    df -h
    ```
*   **Directory Space Usage (Summarized, Human-readable):**
    ```bash
    du -sh /path/to/directory
    ```
*   **Real-time Process Monitoring (Recommended):**
    ```bash
    btop
    ```
*   **List Block Devices (Disks & Partitions):**
    ```bash
    lsblk
    ```
*   **View System Logs (systemd):**
    ```bash
    journalctl -f # Follow new logs in real-time
    journalctl -u nginx.service # View logs for a specific service
    ```

---

## 3. File & Directory Management

*   **List Contents (Long format, Human-readable, All files):**
    ```bash
    ls -lah
    ```
*   **Create Nested Directories:**
    ```bash
    mkdir -p /path/to/new/directory
    ```
*   **Copy a Directory Recursively:**
    ```bash
    cp -r /path/to/source /path/to/destination
    ```
*   **Find Files (Advanced):**
    *   Find files by name (case-insensitive):
        ```bash
        find . -iname "MyFile.txt"
        ```
    *   Find directories only:
        ```bash
        find . -type d -name "config"
        ```
    *   Find files larger than 100MB:
        ```bash
        find . -type f -size +100M
        ```
    *   Execute a command on found files:
        ```bash
        find . -type f -name "*.log" -exec rm -i {} \;
        ```

---

## 4. Text Processing & Manipulation

*   **Grep (Global Regular Expression Print):**
    *   Search recursively and show line numbers:
        ```bash
        grep -rn "search_pattern" /path/to/dir
        ```
    *   Invert search (show lines that DON'T match):
        ```bash
        grep -v "exclude_pattern" file.txt
        ```
*   **Sed (Stream Editor):**
    *   Find and replace in a file (in-place backup):
        ```bash
        sed -i.bak 's/old_text/new_text/g' file.txt
        ```
*   **Awk (Pattern Scanning and Processing Language):**
    *   Print the first and third columns of a space-delimited file:
        ```bash
        awk '{print $1, $3}' data.txt
        ```
*   **Combining Tools with Pipes (`|`):**
    *   Find the 5 largest files on the system:
        ```bash
        sudo find / -type f -exec du -h {} + | sort -rh | head -n 5
        ```
    *   Count the number of active SSH sessions:
        ```bash
        ps aux | grep sshd | wc -l
        ```

---

## 5. User & Permission Management

*   **File Permissions (chmod):**
    *   Make a script executable:
        ```bash
        chmod +x script.sh
        ```
    *   Set permissions using octal notation (e.g., `rwx r-x r--`):
        ```bash
        chmod 754 file.txt
        ```
*   **File Ownership (chown):**
    *   Change owner and group of a file:
        ```bash
        sudo chown user:group /path/to/file
        ```
    *   Change ownership recursively:
        ```bash
        sudo chown -R user:group /path/to/directory
        ```
*   **Add a User to a Group:**
    ```bash
    sudo usermod -aG <groupname> <username>
    ```

---

## 6. Process Management

*   **List Processes with Full Details:**
    ```bash
    ps aux
    ```
*   **Find a Specific Process:**
    ```bash
    ps aux | grep "nginx"
    ```
*   **Kill a Process:**
    *   By Process ID (PID), gracefully:
        ```bash
        kill <PID>
        ```
    *   By PID, forcefully:
        ```bash
        kill -9 <PID>
        ```
    *   By name:
        ```bash
        pkill -f "process_name"
        ```
*   **Backgrounding a Process:**
    *   Start a process in the background:
        ```bash
        my_long_running_command &
        ```
    *   Move a running foreground process to the background:
        1.  Press `Ctrl+Z` to stop the process.
        2.  Type `bg` to resume it in the background.

---

## 7. Networking

*   **Show IP Addresses and Interfaces:**
    ```bash
    ip addr show
    ```
*   **Show Routing Table:**
    ```bash
    ip route
    ```
*   **Check Network Connectivity:**
    ```bash
    ping <hostname_or_ip>
    ```
*   **Trace Network Path:**
    ```bash
    traceroute <hostname_or_ip>
    ```
*   **Check for Listening Ports:**
    ```bash
    ss -tuln
    ```
*   **DNS Lookup:**
    ```bash
    dig example.com
    ```

---

## 8. Development Workflows: Git & Node.js

### **Git Version Control**
*   **View Commit History:**
    ```bash
    git log --oneline --graph --decorate
    ```
*   **Create and Switch to a New Branch:**
    ```bash
    git checkout -b new-feature-branch
    ```
*   **Stage All Changes and Commit:**
    ```bash
    git add . && git commit -m "Your message"
    ```
*   **Stash Changes:**
    *   Temporarily save uncommitted changes:
        ```bash
        git stash
        ```
    *   Re-apply stashed changes:
        ```bash
        git stash pop
        ```

### **Node.js (pnpm)**
*   **Install/Update/Build/Start:**
    ```bash
    pnpm install
    pnpm update
    pnpm build
    pnpm start
    ```
*   **Add a New Dependency:**
    ```bash
    pnpm add <package_name>
    pnpm add -D <package_name> # Add as a dev dependency
    ```

---

## 9. Multimedia with FFmpeg

*   **High-Quality MP4 to GIF:**
    ```bash
    ffmpeg -i input.mp4 -y -vf 'fps=30,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle' -loop 0 output.gif
    ```
*   **Extract Audio from Video:**
    ```bash
    ffmpeg -i video.mp4 -vn -acodec copy audio.aac
    ```
*   **Resize a Video:**
    ```bash
    ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
    ```

---

## 10. Shell Customization & Scripting

### **.bashrc / .zshrc**
*   **Aliases:** Create shortcuts for long commands.
    ```bash
    alias ll='ls -lah'
    alias update='sudo pacman -Syu'
    ```
*   **Environment Variables:** Set variables for your shell session.
    ```bash
    export EDITOR="vim"
    export PATH="$HOME/.local/bin:$PATH"
    ```

### **Basic Bash Scripting**
*   **Example Script (`backup.sh`):**
    ```bash
    #!/bin/bash
    # A simple backup script

    # Variables
    SOURCE_DIR="/home/user/documents"
    DEST_DIR="/mnt/backups"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_FILE="$DEST_DIR/backup_$TIMESTAMP.tar.gz"

    # Logic
    echo "Starting backup of $SOURCE_DIR..."
    tar -czf "$BACKUP_FILE" "$SOURCE_DIR"
    echo "Backup complete: $BACKUP_FILE"
    ```
*   **Making it Executable:**
    ```bash
    chmod +x backup.sh
    ./backup.sh
    ```