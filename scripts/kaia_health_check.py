#!/usr/bin/env python3
"""Independent check that Kaia's security controls are running.

Reads the control health the Policy Gate publishes (security/control_health.py)
and the state of Kaia's systemd units. It runs outside the gate, from
kaia-health-check.timer, so it notices what the gate cannot report about
itself: a hung or dead gate (the file stops being refreshed), a control whose
thread died (its heartbeat stops), and a unit that fails on every boot, which is
what kaia-restore-blocks did until 2026-10-30.

It never changes anything; it reports.

Exit status: 0 every control active or deliberately off; 1 something degraded;
2 something failed, stale or unknown (including the gate itself).

    kaia_health_check.py [--path FILE] [--no-units] [--json] [--journal]
"""

import argparse
import json
import os
import subprocess
import sys
import time

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, DEPLOY_ROOT)

from security import control_health  # noqa: E402  (stays importable without core.config)

DEFAULT_PATH = os.path.join(os.environ.get("KAIA_STATE_DIR", "/var/lib/kaia"), control_health.FILE_NAME)

# unit -> what a healthy host shows. kaia-restore-blocks is a boot oneshot: the
# installer stops it, so between an install and the next boot it is inactive
# with a successful result, which is fine. What matters is that it is enabled
# (it will run at boot) and that its last run did not fail (blocks were lost).
UNITS = {
    "kaia-policy-gate.service": {"active_states": ("active",), "must_exist": True},
    "kaia-restore-blocks.service": {"active_states": ("active", "inactive"), "must_exist": False,
                                    "must_be_enabled": True},
}
SYSTEMCTL = "/usr/bin/systemctl"

# sd-daemon(3) priority prefixes, parsed by journald on a unit's stdout.
JOURNAL_PRIORITY = {"failed": "<3>", "stale": "<3>", "degraded": "<4>", "active": "<6>", "off": "<6>"}


def unit_states(units=UNITS, runner=subprocess.run) -> list:
    """[{name, state, detail}] for each unit, from `systemctl show` (no root needed)."""
    results = []
    for unit, expect in units.items():
        try:
            out = runner(
                [SYSTEMCTL, "show", unit, "--property=LoadState,ActiveState,SubState,Result,UnitFileState"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            results.append({"name": unit, "state": "failed", "detail": f"cannot query systemd: {e}"})
            continue
        props = dict(
            line.split("=", 1) for line in (out.stdout or "").splitlines() if "=" in line
        )
        load, active = props.get("LoadState", ""), props.get("ActiveState", "")
        sub, result = props.get("SubState", ""), props.get("Result", "")
        detail = f"{active}/{sub}, result {result or 'unknown'}"
        if out.returncode != 0 or not props:
            state, detail = "failed", f"systemctl show failed: {(out.stderr or '').strip()[:200]}"
        elif load == "not-found":
            state = "failed" if expect["must_exist"] else "off"
            detail = "unit not installed"
        elif active == "failed" or result not in ("", "success"):
            state = "failed"
        elif expect.get("must_be_enabled") and props.get("UnitFileState", "enabled") != "enabled":
            state = "degraded"
            detail += f"; not enabled ({props.get('UnitFileState')}): it will not run at boot"
        elif active in expect["active_states"]:
            state = "active"
        else:
            state = "failed" if expect["must_exist"] else "degraded"
        results.append({"name": unit, "state": state, "detail": detail})
    return results


def run(path: str, check_units: bool = True, now=None, runner=subprocess.run) -> dict:
    report = control_health.load_and_evaluate(path, now=now)
    units = unit_states(runner=runner) if check_units else []
    worst = max(
        [control_health.SEVERITY[report["overall"]]]
        + [control_health.SEVERITY[u["state"]] for u in units]
    )
    report["units"] = units
    report["overall"] = {0: "active", 1: "degraded", 2: "failed"}[worst]
    report["exit_status"] = worst
    return report


def format_lines(report: dict, journal: bool = False) -> list:
    def line(state: str, text: str) -> str:
        return (JOURNAL_PRIORITY.get(state, "<4>") if journal else "") + text

    items = [("control", c) for c in report["controls"]] + [("unit", u) for u in report["units"]]
    counts = {}
    for _, item in items:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "nothing reported"
    out = [line(report["overall"], f"Kaia controls: {report['overall'].upper()} ({summary})")]
    if report.get("problem"):
        out.append(line("failed", f"  gate: {report['problem']}"))
    for kind, item in items:
        if item["state"] in ("active", "off"):
            continue
        out.append(line(item["state"], f"  {kind} {item['name']}: {item['state']}: {item['detail']}"))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", default=DEFAULT_PATH, help="published control health file")
    parser.add_argument("--no-units", action="store_true", help="skip the systemd unit checks")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--journal", action="store_true", help="prefix lines with journal priorities")
    args = parser.parse_args(argv)

    report = run(args.path, check_units=not args.no_units, now=time.time())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for text in format_lines(report, journal=args.journal):
            print(text)
    return report["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
