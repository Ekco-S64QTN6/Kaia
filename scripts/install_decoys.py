#!/usr/bin/env python3
"""Create missing filesystem decoys with their exact contents. Never overwrites.

Run as root by install_services.sh before the gate starts. A path that already
exists, whatever it holds, is left alone: it may be real data, and the
coordinator will only watch it if its bytes equal the decoy contents.
"""

import os
import sys

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, DEPLOY_ROOT)

from security.decoys import DECOYS  # noqa: E402


def provision(decoys=DECOYS, root: str = "") -> dict:
    """{path: 'created' | 'exists' | 'error: ...'} for each decoy."""
    results = {}
    for path, content in decoys.items():
        target = root + path
        parent = os.path.dirname(target)
        if not os.path.isdir(parent) or os.path.islink(parent):
            results[path] = "error: parent directory missing or a symlink"
            continue
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except FileExistsError:
            results[path] = "exists"
            continue
        except OSError as e:
            results[path] = f"error: {e}"
            continue
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        results[path] = "created"
    return results


if __name__ == "__main__":
    failed = False
    for path, outcome in provision().items():
        print(f"  {outcome:<8} {path}")
        failed = failed or outcome.startswith("error")
    sys.exit(1 if failed else 0)
