# Linux Disk Error Troubleshooting Guide

## Common Symptoms
* Filesystem errors during boot (e.g., `fsck` prompts).
* Slow system performance, especially during file operations.
* Read/write errors reported by applications.
* Unusual grinding or clicking noises from physical drives.
* Partitions becoming read-only unexpectedly.

## Initial Diagnostics
* **Check Disk Usage:**
    * `df -h`: View free and used space on mounted filesystems.
    * `du -sh /path/to/directory`: Check size of specific directories.
* **Identify Drive:**
    * `lsblk`: List block devices (disks and partitions).
    * `fdisk -l` or `parted -l`: Detailed partition information.
* **Monitor Disk I/O:**
    * `iotop`: Real-time disk I/O monitoring (install if not present).
    * `iostat -xz 1`: Report CPU and I/O statistics.

## Filesystem Checks (fsck)
* **Unmount Filesystem First:** `sudo umount /dev/sdXN` (replace `sdXN` with your partition). Cannot run `fsck` on a mounted filesystem.
* **Run Check:** `sudo fsck -f /dev/sdXN` (The `-f` option forces a check even if the filesystem seems clean).
* **For ext4:** `sudo e2fsck -f /dev/sdXN`
* **Force Check on Next Boot:** `sudo touch /forcefsck` (for older sysvinit/some systemd configurations).

## S.M.A.R.T. Health Check (for HDDs/SSDs)
* **Install `smartmontools`:** `sudo pacman -S smartmontools`
* **Check Drive Health:** `sudo smartctl -H /dev/sdX` (replace `sdX` with your drive, e.g., `sda`).
* **View Full Report:** `sudo smartctl -a /dev/sdX | less`

## Common Solutions & Notes
* **Clear Space:** If disk is full, remove unnecessary files.
* **Bad Blocks:** `sudo badblocks -sv /dev/sdXN` (use with extreme caution, can destroy data if used incorrectly).
* **Hardware Failure:** If SMART tests indicate failure or persistent issues, consider drive replacement.
* **Backups:** Always ensure recent backups before attempting significant disk repairs.
