#!/usr/bin/env python3
"""Which installed packages are affected by Arch Linux security advisories?

Fetches the public advisory list (nothing about this host is sent), matches it
against pacman's installed versions (security/vuln_advisories.py) and prints the
affected packages, most severe first. Run daily by kaia-vuln-check.timer.

Exit: 0 nothing actionable; 1 a High/Critical advisory has a fixed version this
host has not installed (upgrade: pacman -Syu); 2 the check could not run.

    kaia_vuln_check.py [--json] [--file advisories.json]
"""

import json
import os
import ssl
import sys
import urllib.request

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, DEPLOY_ROOT)

from security import vuln_advisories as va  # noqa: E402

URL = "https://security.archlinux.org/issues/all.json"
MAX_BYTES = 32 * 1024 * 1024


def fetch(url: str = URL) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "kaia-vuln-check"})
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as resp:
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("advisory list exceeds size limit")
    return json.loads(data)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if "--file" in argv:
            with open(argv[argv.index("--file") + 1], encoding="utf-8") as f:
                groups = json.load(f)
        else:
            groups = fetch()
        hits = va.affected(groups, va.installed_versions())
    except Exception as e:
        print(f"vulnerability check could not run: {e}", file=sys.stderr)
        return 2
    actionable = [h for h in hits if h["fixed"] and va.SEVERITY_RANK[h["severity"]] >= 3]
    if "--json" in argv:
        print(json.dumps({"affected": hits, "actionable": actionable}, indent=2))
    else:
        print(f"{len(hits)} advisories affect installed packages; {len(actionable)} High/Critical with a fix available")
        for h in hits[:40]:
            fix = f"fixed in {h['fixed']} (upgrade)" if h["fixed"] else "no fix released yet"
            print(f"  {h['severity']:<8} {h['package']} {h['installed']}: {h['group']} {h['type']}, "
                  f"{len(h['cves'])} CVEs, {fix}")
    return 1 if actionable else 0


if __name__ == "__main__":
    sys.exit(main())
