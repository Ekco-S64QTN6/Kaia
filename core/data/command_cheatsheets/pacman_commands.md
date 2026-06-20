# Pacman Commands Quick Reference (Arch Linux)

## System Update
* `sudo pacman -Syu`: Synchronize package databases and update all installed packages.
* `sudo pacman -Syyu`: Force database synchronization before updating (useful if `pacman -Syu` fails or databases are out of sync).
* `sudo pacman -Syuu`: Upgrade a system from a testing or unstable branch (downgrades packages if newer versions exist in stable).

## Package Installation & Removal
* `sudo pacman -S <package_name>`: Install a new package.
    * Example: `sudo pacman -S neovim`
* `sudo pacman -R <package_name>`: Remove a package, leaving all of its dependencies.
* `sudo pacman -Rs <package_name>`: Remove a package and its dependencies that are no longer required by any other installed packages.
* `sudo pacman -Rns <package_name>`: Remove a package, its unneeded dependencies, and configuration files.

## Package Query & Search
* `pacman -Ss <keyword>`: Search for packages in the repositories (by name and description).
    * Example: `pacman -Ss firefox`
* `pacman -Qs <keyword>`: Search for installed packages.
* `pacman -Si <package_name>`: Display information about a package in the repositories.
* `pacman -Qi <package_name>`: Display information about an installed package.
* `pacman -Ql <package_name>`: List all files owned by an installed package.
* `pacman -Qo <file_path>`: Show which package owns a specific file.
    * Example: `pacman -Qo /usr/bin/ls`

## Package Management & Maintenance
* `sudo pacman -Sc`: Clean the package cache by removing old versions of packages.
* `sudo pacman -Scc`: Clean the entire package cache (removes all cached packages, use with caution).
* `pacman -Qdt`: List orphaned packages (dependencies no longer required by any installed package).
* `sudo pacman -Rns $(pacman -Qdt)`: Remove all orphaned packages.
* `pacman -Qk`: Check for broken installed packages.

## Troubleshooting
* **Database Lock:** If `pacman` complains about a database lock, remove the lock file: `sudo rm /var/lib/pacman/db.lck` (use only if no `pacman` process is running).
* **Keyring Issues:** If signature verification fails, refresh Arch Linux keyring: `sudo pacman -Sy archlinux-keyring && sudo pacman -Syyu`.
