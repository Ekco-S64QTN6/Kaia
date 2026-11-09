#!/usr/bin/env python3
"""Verify setuid binaries, PAM, the loader and Kaia's executor binaries against pacman.

The same pass the Policy Gate runs every 6 hours (security/package_attestation.py),
on demand. Read-only. Run as root to read every file (some setuid helpers are
not world-readable).

Exit status: 0 nothing altered; 1 files altered or privileged files unowned;
2 the pass could not run.

    kaia_package_attest.py [--json]
"""

import json
import os
import sys

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, DEPLOY_ROOT)

from security import package_attestation as pa  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        report = pa.attest()
    except Exception as e:
        print(f"package attestation could not run: {e}", file=sys.stderr)
        return 2
    if "--json" in argv:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['checked']} critical files from {report['packages']} packages in {report['duration']} s")
        for f in report["findings"]:
            print(f"  {f['kind']}: {f['path']} ({f['package']}): {f['detail']}")
        for path in report["unowned_privileged"]:
            print(f"  unowned privileged file: {path}")
        for err in report["errors"]:
            print(f"  unreadable package record: {err}")
    altered = [f for f in report["findings"] if f["kind"] in pa.ALTERED_KINDS]
    return 1 if altered or report["unowned_privileged"] else 0


if __name__ == "__main__":
    sys.exit(main())
