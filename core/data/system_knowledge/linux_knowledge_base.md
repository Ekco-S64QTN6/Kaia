# Linux Knowledge Base for AI Assistant

This document contains general Linux commands, concepts, and troubleshooting tips. It is designed to provide foundational knowledge for a Linux desktop assistant without including any personally identifiable information (PII) or user-specific configurations.

## 1. Basic Linux Commands & Navigation

### File System
-   `pwd`: Print working directory (show current directory path)
-   `ls [options]`: List directory contents (`-l` for long format, `-a` for all files including hidden)
-   `cd [directory]`: Change directory
-   `mkdir [directory_name]`: Create a new directory
-   `rm [options] [file/directory]`: Remove files or directories (`-r` for recursive, `-f` for force)
-   `cp [source] [destination]`: Copy files or directories
-   `mv [source] [destination]`: Move (rename) files or directories
-   `touch [file_name]`: Create an empty file or update timestamp

### File Content
-   `cat [file]`: Display file content
-   `less [file]`: View file content page by page
-   `head [options] [file]`: Display the beginning of a file (`-n <lines>`)
-   `tail [options] [file]`: Display the end of a file (`-n <lines>`, `-f` for follow)

### Permissions & Ownership
-   `chmod [permissions] [file]`: Change file permissions (e.g., `chmod +x script.sh`, `chmod 755 file`)
-   `chown [user]:[group] [file]`: Change file ownership

## 2. System Management & Troubleshooting

### Systemd (Service Management)
-   `systemctl start [service]`: Start a service
-   `systemctl stop [service]`: Stop a service
-   `systemctl restart [service]`: Restart a service
-   `systemctl enable [service]`: Enable service to start on boot
-   `systemctl disable [service]`: Disable service from starting on boot
-   `systemctl status [service]`: Check service status
-   `journalctl -u [service]`: View logs for a specific systemd service
-   `journalctl -f`: Follow system logs in real-time

### Process Management
-   `ps aux`: Display running processes
-   `top`: Real-time view of system processes
-   `htop`: Interactive process viewer (often needs installation: `sudo pacman -S htop`)
-   `kill [PID]`: Terminate a process by ID
-   `killall [process_name]`: Terminate processes by name

### Disk Usage & Storage
-   `df -h`: Display free disk space in human-readable format
-   `du -sh [directory]`: Summarize disk usage of a directory
-   `fdisk -l`: List disk partitions (requires root)
-   `lsblk`: List block devices

## 3. Networking Basics

### Network Configuration
-   `ip a`: Display network interface information (IP addresses, etc.)
-   `ping [hostname/IP]`: Test network connectivity
-   `ss -tuln`: List listening sockets (ports)
-   `netstat -tuln`: (Deprecated, but still common) List listening sockets

### Remote Access
-   `ssh [user]@[host]`: Secure Shell for remote login
-   `scp [source] [destination]`: Secure copy files over SSH

## 4. Package Management (Arch Linux Focus)

-   **Pacman:**
    -   `sudo pacman -Syu`: Synchronize package databases and update system
    -   `sudo pacman -S [package]`: Install a package
    -   `sudo pacman -R [package]`: Remove a package
    -   `sudo pacman -Qs [keyword]`: Query local database for installed packages matching keyword
    -   `sudo pacman -Ss [keyword]`: Search remote repositories for packages
-   **AUR Helpers (e.g., Yay):**
    -   `yay -S [package]`: Install/build package from AUR
    -   `yay -Syu`: Update system including AUR packages

## 5. Text Processing & Utilities

-   `grep [pattern] [file]`: Search for patterns in files
-   `find [path] [expression]`: Search for files and directories
-   `sed`: Stream editor for text transformations
-   `awk`: Pattern-scanning and processing language
-   `tar [options] [archive] [files]`: Create or extract archive files (e.g., `tar -cvf`, `tar -xvf`)
-   `rsync [options] [source] [destination]`: Efficiently copy and synchronize files remotely or locally

## 6. General Best Practices & Concepts

-   **Security:** Keep your system updated, use strong unique passwords, understand firewall basics.
-   **Backup Strategy:** Emphasize regular backups of important data.
-   **The Unix Philosophy:** "Do one thing and do it well."
-   **Dotfiles:** Mention the concept of managing configuration files (dotfiles) for system personalization.

## 7. User and Group Management

-   `sudo [command]`: Execute a command as the superuser.
-   `useradd [options] [username]`: Create a new user account.
-   `userdel [options] [username]`: Delete a user account.
-   `usermod [options] [username]`: Modify user account properties.
-   `passwd [username]`: Change a user's password.
-   `groupadd [groupname]`: Create a new group.
-   `groupdel [groupname]`: Delete a group.
-   `groups [username]`: Display the groups a user belongs to.
-   `id [username]`: Print user and group IDs.

## 8. Environmental Variables and Shell Customization

-   `printenv` or `env`: Display all environment variables.
-   `echo $[VARIABLE_NAME]`: Display the value of a specific environment variable (e.g., `echo $PATH`).
-   `export [VARIABLE_NAME]=[value]`: Set an environment variable for the current session and child processes.
-   `~/.bashrc`, `~/.zshrc`: Shell configuration files for interactive shells.
-   `~/.profile`, `~/.bash_profile`: Login shell configuration files.
-   `PATH`: Environment variable listing directories where executables are searched.

## 9. I/O Redirection and Pipes

-   `command > file`: Redirect standard output to a file (overwrite).
-   `command >> file`: Redirect standard output to a file (append).
-   `command < file`: Redirect standard input from a file.
-   `command 2> error_file`: Redirect standard error to a file.
-   `command &> file`: Redirect both standard output and standard error to a file.
-   `command1 | command2`: Pipe the standard output of `command1` as standard input to `command2`.
-   `tee [file]`: Read from standard input and write to both standard output and one or more files.

## 10. Scheduling Tasks (Cron)

-   `crontab -e`: Edit the current user's crontab file.
-   `crontab -l`: List the current user's crontab entries.
-   `crontab -r`: Remove the current user's crontab file.
-   Cron syntax: `* * * * * command_to_execute` (minute, hour, day of month, month, day of week).

## 11. Advanced Troubleshooting & Diagnostics

### Network Diagnostics
-   `ss -tunlp`: List all listening TCP/UDP sockets with associated processes (requires root for process info).
-   `netstat -tulnp`: (Older, but common) Same as above.
-   `traceroute [hostname/IP]`: Trace the route packets take to a network host.
-   `nslookup [hostname/IP]` or `dig [hostname/IP]`: Query DNS servers.
-   `nmap [options] [target]`: Network scanner for discovery and security auditing (often needs installation: `sudo pacman -S nmap`).

### System Logs
-   `/var/log/`: Directory containing various system logs (e.g., `syslog`, `auth.log`, `kern.log`).
-   `dmesg`: Display kernel ring buffer messages (boot messages, device drivers).

### Hardware & Kernel Modules
-   `lsmod`: List loaded kernel modules.
-   `modprobe [module_name]`: Add or remove a kernel module.
-   `lspci`: List all PCI devices.
-   `lsusb`: List USB devices.
-   `lshw -short`: Summarize hardware configuration (often needs installation: `sudo pacman -S lshw`).

## 12. Basic Text Editors

-   **Nano:**
    -   `nano [file]`: Open a file in the Nano editor.
    -   **Usage:** Simple, user-friendly text editor. Commands are displayed at the bottom of the screen (e.g., `^X` to exit, `^O` to save).

-   **Vim (Very basic overview):**
    -   `vim [file]`: Open a file in the Vim editor.
    -   **Modes:**
        -   **Normal Mode (default):** For navigation and commands.
        -   **Insert Mode (press `i`):** For typing text.
    -   **Saving & Quitting:**
        -   Press `Esc` to return to Normal Mode.
        -   `:w` (write/save), `:q` (quit), `:wq` (write and quit), `:q!` (quit without saving).

## 13. More on File Archiving and Compression

-   **Gzip/Gunzip:**
    -   `gzip [file]`: Compress file into `.gz` format (replaces original).
    -   `gunzip [file.gz]`: Decompress `.gz` file (replaces original).
    -   `gzip -d [file.gz]`: Same as `gunzip`.
    -   `zcat [file.gz]`: View compressed file content without decompressing.

-   **Bzip2/Bunzip2:**
    -   `bzip2 [file]`: Compress file into `.bz2` format.
    -   `bunzip2 [file.bz2]`: Decompress `.bz2` file.
    -   `bzcat [file.bz2]`: View compressed file content without decompressing.

-   **Xz/Unxz:**
    -   `xz [file]`: Compress file into `.xz` format.
    -   `unxz [file.xz]`: Decompress `.xz` file.
    -   `xzcat [file.xz]`: View compressed file content without decompressing.

-   **Zip/Unzip:** (For cross-platform archives)
    -   `zip [archive_name].zip [files/directories]`: Create a zip archive.
    -   `unzip [archive_name].zip`: Extract files from a zip archive.

## 14. System Information and Uptime

-   `uptime`: Show how long the system has been running, number of users, and load averages.
-   `hostname`: Display the system's hostname.
-   `whoami`: Print the effective username of the current user.
-   `w`: Display who is logged on and what they are doing.
-   `history`: Display the shell command history.

# Advanced Command Combinations
- `journalctl -xe | grep -i error`: Filter system logs for errors
- `rsync -avz --progress /source user@remote:/dest`: Detailed progress during transfer
- `find / -type f -size +100M -exec ls -lh {} +`: Find large files

## Common Troubleshooting Scenarios

### Network Issues
1. `ip link show` → Check interface status
2. `sudo systemctl restart NetworkManager`
3. Check `/etc/resolv.conf` for DNS settings

### Performance Diagnostics
- `iotop` → Disk I/O monitoring
- `nethogs` → Per-process network usage
