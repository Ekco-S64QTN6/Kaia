"""The network radar conveys information: ring = last seen, sector = how discovered."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

HARNESS = r"""
global.window = {};
global.requestAnimationFrame = () => {};
global.performance = { now: () => 0 };
eval(require('fs').readFileSync(process.argv[2], 'utf8'));
const R = window.NetworkRadarCanvas;
console.log(JSON.stringify({
  r0: R.ageRadius(0), r5m: R.ageRadius(300), r1h: R.ageRadius(3600), r24: R.ageRadius(86400), rOld: R.ageRadius(9e9),
  arp: R.sectorOf('ARP'), mdns: R.sectorOf('mDNS'), nb: R.sectorOf('NBSTAT'), other: R.sectorOf('dhcp'), none: R.sectorOf(null),
}));
"""


def test_rings_and_sectors():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(HARNESS)
    r = subprocess.run([node, f.name, str(ROOT / "core/web/static/js/radar.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1])
    assert o["r0"] < o["r5m"] < o["r1h"] < o["r24"] <= 0.97 and o["rOld"] == o["r24"]
    assert (o["arp"], o["mdns"], o["nb"], o["other"], o["none"]) == ("ARP", "mDNS", "NBNS", "OTHER", "OTHER")


def test_radar_gets_every_recent_device_and_room():
    assert "get_recent_assets(40)" in (ROOT / "kaia_dashboard.py").read_text()
    css = (ROOT / "core/web/static/css/sentinel.css").read_text()
    assert "height: 162px;" in css and ".network-radar-wrap:not(.has-risk) .radar-risk-wrap { display: none; }" in css
    js = (ROOT / "core/web/static/js/radar.js").read_text()
    assert "ring = last seen" in js and "font = '10px" in js
