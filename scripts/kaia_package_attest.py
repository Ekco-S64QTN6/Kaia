#!/usr/bin/env python3
"""Verify setuid binaries, PAM, the loader and Kaia's executor binaries against pacman.

The same pass the Policy Gate runs every 6 hours (security/package_attestation.py),
on demand. Read-only. Run as root to read every file (some setuid helpers are
not world-readable).

Exit status: 0 nothing altered; 1 files altered or privileged files unowned;
2 the pass could not run.

    kaia_package_attest.py [--json] [--signed]

--signed compares the files with the signed package archives in pacman's cache
instead of the local database (run daily by kaia-package-verify.timer).
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
    if "--signed" in argv:
        return signed_main(argv)
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


def signed_main(argv) -> int:
    try:
        report = pa.signed_attest()
    except Exception as e:
        print(f"signed package verification could not run: {e}", file=sys.stderr)
        return 2
    if "--json" in argv:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['checked']} critical files from {report['packages']} packages checked against "
              f"signed archives in {report['duration']} s")
        for f in report["findings"]:
            print(f"  {f['kind']}: {f['path']} ({f['package']}): {f['detail']}")
        if report["no_signed_package"]:
            print(f"  not verifiable (archive not cached): {', '.join(report['no_signed_package'][:20])}")
    bad = [f for f in report["findings"] if f["kind"] in pa.SIGNED_KINDS + ("signature_invalid",)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
