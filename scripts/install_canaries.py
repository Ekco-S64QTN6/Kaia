#!/usr/bin/env python3
"""Install the canary "flag" files the honeypot watches.

  <owner home>/flag.txt  owner-readable: where an agent running as the user lands
  /root/root.txt         root-only: reaching it means a privilege escalation

CTF-trained agents go looking for flag.txt / root.txt, so these are files a
hijacked agent is likely to open. The policy gate watches them by inode
(security/honeypot.py); this script only places them.

Safety rules match HoneypotCoordinator's decoys: never follow symlinks, never
overwrite a file that is not provably a canary. An existing canary keeps its
ID (so logs stay traceable) and gets the current text; the owner's original
hand-written flag.txt is backed up before being replaced.
"""

import argparse
import errno
import os
import pwd
import re
import secrets
import stat
import sys
from pathlib import Path

ID_RE = re.compile(rb"KAIA-CANARY-[0-9a-f]{16}")
# The owner's original hand-written flag.txt; safe to supersede (backed up).
LEGACY_MARKERS = (b"AUTONOMOUS AGENT \xe2\x80\x94 STOP AND REPORT",)
MAX_BYTES = 64 * 1024


def read_no_follow(path: Path):
    """(bytes, stat) of a small regular file, or (None, stat/None) otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        return None, None
    except OSError as e:
        if e.errno == errno.ELOOP:  # a symlink sits at the target: never follow it
            return None, os.lstat(path)
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > MAX_BYTES:
            return None, st
        return os.read(fd, MAX_BYTES + 1), st
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes, uid: int, gid: int, mode: int) -> None:
    """Write via an exclusively-created temp file in the same directory, then rename."""
    tmp = path.with_name(f".{path.name}.kaia-tmp-{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, data)
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise
    os.close(fd)
    os.replace(tmp, path)


def install(target: Path, template: Path, uid: int, gid: int, mode: int) -> str:
    """Install one canary; returns its canary ID, or '' if left untouched."""
    text = template.read_bytes()
    existing, st = read_no_follow(target)

    if st is None:
        canary_id = f"KAIA-CANARY-{secrets.token_hex(8)}"
        action = "created"
    elif existing is None:
        print(f"  ! {target}: not a small regular single-link file; left untouched", file=sys.stderr)
        return ""
    elif st.st_uid != uid:
        print(f"  ! {target}: owned by uid {st.st_uid}, expected {uid}; left untouched", file=sys.stderr)
        return ""
    elif ID_RE.search(existing):
        canary_id = ID_RE.search(existing).group(0).decode()
        action = "updated (ID kept)"
    elif any(m in existing for m in LEGACY_MARKERS):
        backup = target.with_name(target.name + ".pre-kaia")
        if not backup.exists():
            write_atomic(backup, existing, uid, gid, st.st_mode & 0o777)
        canary_id = f"KAIA-CANARY-{secrets.token_hex(8)}"
        action = f"replaced your original (backup: {backup})"
    else:
        print(f"  ! {target}: exists and is not a Kaia canary; left untouched", file=sys.stderr)
        return ""

    data = text.replace(b"__CANARY_ID__", canary_id.encode())
    if existing == data:
        print(f"  = {target}: already current ({canary_id})")
        return canary_id
    write_atomic(target, data, uid, gid, mode)
    print(f"  + {target}: {action} ({canary_id})")
    return canary_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--owner", help="login that owns the home-directory canary")
    ap.add_argument("--template-dir", required=True, type=Path)
    ap.add_argument("--ids-file", type=Path, default=Path("/etc/kaia/canary_ids"))
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("install_canaries.py must run as root", file=sys.stderr)
        return 1

    ids = []
    if args.owner and args.owner != "root":
        pw = pwd.getpwnam(args.owner)
        cid = install(Path(pw.pw_dir) / "flag.txt", args.template_dir / "user_flag.txt",
                      pw.pw_uid, pw.pw_gid, 0o644)
        if cid:
            ids.append(f"{cid} {pw.pw_dir}/flag.txt")
    cid = install(Path("/root/root.txt"), args.template_dir / "root_flag.txt", 0, 0, 0o600)
    if cid:
        ids.append(f"{cid} /root/root.txt")

    # Root-only record of the IDs, so they can be searched for in logs or leaks.
    if ids:
        args.ids_file.parent.mkdir(mode=0o700, exist_ok=True)
        write_atomic(args.ids_file, ("\n".join(ids) + "\n").encode(), 0, 0, 0o600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
