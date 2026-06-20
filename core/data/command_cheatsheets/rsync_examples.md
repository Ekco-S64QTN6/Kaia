# rsync Examples Cheat Sheet

`rsync` is a powerful and versatile command-line utility for synchronizing files and directories, locally or remotely.

## Basic Syntax
`rsync [options] source destination`

## Common Options
* `-a`, `--archive`: Archive mode; equates to `-rlptgoD` (recursive, links, perms, times, group, owner, devices, specials). Most commonly used.
* `-v`, `--verbose`: Increase verbosity.
* `-h`, `--human-readable`: Output numbers in a human-readable format.
* `-z`, `--compress`: Compress file data during transfer (useful for remote sync).
* `-P`: Combination of `--progress` (show progress) and `--partial` (keep partially transferred files).
* `-n`, `--dry-run`: Perform a trial run with no changes made. Highly recommended for complex commands.
* `--delete`: Delete extra files at the destination that are not present at the source. Use with extreme caution.
* `-e <ssh_command>`: Specify the remote shell to use (e.g., `-e ssh`).
* `--exclude=<pattern>`: Exclude files/directories matching a pattern.
* `--include=<pattern>`: Don't exclude files/directories matching a pattern (used with `--exclude`).

## Local Synchronization
* **Copy a directory recursively:**
    `rsync -av /path/to/source_dir/ /path/to/destination_dir/`
    * Note the trailing slash `/` on `source_dir/`: Copies *contents* of `source_dir` into `destination_dir`.
    * Without trailing slash (`/path/to/source_dir`): Copies `source_dir` *itself* into `destination_dir` (i.e., `destination_dir/source_dir/`).

* **Mirror a directory (delete extraneous files at destination):**
    `rsync -av --delete /path/to/source_dir/ /path/to/destination_dir/`

* **Copy specific files with dry run:**
    `rsync -av --dry-run /path/to/source_dir/*.txt /path/to/destination_dir/`

## Remote Synchronization (via SSH)
* **Local to Remote:**
    `rsync -avzP -e ssh /path/to/local_dir/ user@remote_host:/path/to/remote_dir/`
    * Or simply: `rsync -avzP /path/to/local_dir/ user@remote_host:/path/to/remote_dir/` (SSH is default).

* **Remote to Local:**
    `rsync -avzP user@remote_host:/path/to/remote_dir/ /path/to/local_dir/`

* **Mirror Remote to Local (backup):**
    `rsync -avzP --delete user@remote_host:/path/to/remote_dir/ /path/to/local_backup_dir/`

## Excluding and Including Files
* **Exclude a specific directory:**
    `rsync -av --exclude 'cache/' /path/to/source/ /path/to/destination/`
* **Exclude multiple patterns:**
    `rsync -av --exclude={'*.log','tmp/','cache/'} /path/to/source/ /path/to/destination/`
* **Exclude everything but include specific files (order matters!):**
    `rsync -av --exclude '*' --include '*.md' --include 'documents/' /path/to/source/ /path/to/destination/`
    * Exclude all, then include all `.md` files, then include the `documents` directory.
