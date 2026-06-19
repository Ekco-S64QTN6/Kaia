# My Linux System Directories and Data Organization

This document details the structure and purpose of my common Linux system directories and outlines my general approach to data organization and backups. This information helps Kaia understand my overall file system and assist with file management, project context, and backup strategies across my environment.

## Primary Directory Structure

- **Root Project Directory:** `~/projects/`
- **Work Projects:** `~/projects/work/[project_name]/`
  - Each work project typically contains: `src/` (source code), `docs/` (documentation), `venv/` (Python virtual environment).
- **Personal Coding Projects:** `~/github/`
  - These are personal repositories and development tools.
  - **Note:** The `Kaia` project (Hardened AI Admin Agent) is located at `~/github/Kaia/`. Its detailed internal structure is documented in `~/github/Kaia/personal_context/my_project_filesystem_map.md`.
- **Configuration Files (Dotfiles):** `~/dotfiles/`
  - Managed using Git for version control and synchronization across machines.
- **Temporary Scratchpad:** `~/tmp/`
  - Used for quick tests, temporary files, and miscellaneous items. Content is cleared periodically.
- **Media Storage:**
  - `~/media/photos/`
  - `~/media/videos/`
- **Documents:**
  - `~/documents/work/` (for work-related documents, reports, etc.)
  - `~/documents/personal/` (for personal documents, notes, etc.)

## Backup Strategy

- **Daily:** `rsync` to an external hard drive for frequently updated, critical data.
- **Weekly:** Cloud synchronization for critical data, ensuring off-site redundancy.
