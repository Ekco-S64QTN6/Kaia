#!/usr/bin/env python3
"""Probe which fanotify mount-mark masks this kernel accepts.

fanotify mount marks require CAP_SYS_ADMIN, and several event types are only
permitted when fanotify_init() was given FAN_REPORT_FID. Rather than guess which
combination a given kernel objects to, ask it.

Use this to decide whether the Policy Gate unit needs CAP_SYS_ADMIN: if no mask
tier succeeds, mount-wide FIM cannot work and the capability buys nothing.

    sudo python3 scripts/fanotify_probe.py /opt/kaia
"""
import ctypes
import os
import sys

FAN_CLASS_NOTIF, FAN_CLOEXEC, FAN_NONBLOCK = 0x0, 0x1, 0x2
FAN_MARK_ADD, FAN_MARK_MOUNT = 0x1, 0x10
FAN_MODIFY, FAN_ATTRIB, FAN_CLOSE_WRITE = 0x2, 0x4, 0x8
FAN_CREATE, FAN_ONDIR = 0x100, 0x40000000
O_RDONLY, O_LARGEFILE = 0x0, 0x8000
AT_FDCWD = -100

TIERS = [
    ("full (modify, close-write, create, attrib, ondir)",
     FAN_MODIFY | FAN_CLOSE_WRITE | FAN_CREATE | FAN_ATTRIB | FAN_ONDIR),
    ("no directory events (modify, close-write, attrib)",
     FAN_MODIFY | FAN_CLOSE_WRITE | FAN_ATTRIB),
    ("file content only (modify, close-write)",
     FAN_MODIFY | FAN_CLOSE_WRITE),
    ("minimal (modify)", FAN_MODIFY),
]


def main() -> int:
    path = (sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    if os.geteuid() != 0:
        print("fanotify mount marks require root/CAP_SYS_ADMIN. Re-run with sudo.")
        return 1

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.fanotify_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
    libc.fanotify_mark.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint64,
                                   ctypes.c_int, ctypes.c_char_p]

    fd = libc.fanotify_init(FAN_CLASS_NOTIF | FAN_CLOEXEC | FAN_NONBLOCK,
                            O_RDONLY | O_LARGEFILE)
    if fd < 0:
        print(f"fanotify_init failed (errno={ctypes.get_errno()})")
        return 1

    print(f"Probing mount marks on: {path}\n")
    any_ok = False
    for label, mask in TIERS:
        rc = libc.fanotify_mark(fd, FAN_MARK_ADD | FAN_MARK_MOUNT, mask,
                                AT_FDCWD, path.encode())
        if rc >= 0:
            print(f"  OK    {label}")
            any_ok = True
        else:
            print(f"  FAIL  {label}  (errno={ctypes.get_errno()})")
    os.close(fd)

    print()
    if any_ok:
        print("At least one tier works: mount-wide FIM is viable on this kernel.")
        print("Add CAP_SYS_ADMIN back to kaia-policy-gate.service to enable it.")
    else:
        print("No tier works: mount-wide FIM cannot run here, so CAP_SYS_ADMIN")
        print("would grant privilege for nothing. Leave it dropped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
