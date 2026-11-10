"""Opt-in audio cues: silent by default, never for events present at load, rate-limited.

Runs core/web/static/js/audio.js in node with a stubbed AudioContext.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

HARNESS = r"""
const tones = [];
let now = 0;
global.Date = { now: () => now };
global.window = {
  localStorage: { store: {}, getItem(k) { return this.store[k] || null; }, setItem(k, v) { this.store[k] = v; } },
  AudioContext: function () {
    this.currentTime = 0; this.destination = {};
    this.createOscillator = () => ({ frequency: {}, connect: (g) => g, start() {}, stop() {}, set type(v) {} });
    this.createGain = () => ({ gain: { setValueAtTime() {}, exponentialRampToValueAtTime() {} }, connect: () => ({}) });
  },
};
const realTone = null;
eval(require('fs').readFileSync(process.argv[2], 'utf8'));
const A = window.AlertAudio;
const orig = A.prototype.tone;
A.prototype.tone = function (f, d, g) { const before = this.lastTone; orig.call(this, f, d, g); if (this.lastTone !== before) tones.push(f.length); };
const ev = (iso, level) => ({ iso, event: 'x', detail: iso, alert_level: level });
const snap = (logs, lock) => ({ logs, system_security: { lockdown_state: lock || 'inactive' } });
const out = {};
const a = new A(null);
a.observe(snap([ev('1', 'critical')]));                   // present at load: primed, silent
a.observe(snap([ev('1', 'critical'), ev('2', 'critical')]));
out.disabled = tones.length;                               // disabled: silent
a.enabled = true;
now = 10000; a.observe(snap([ev('3', 'high')]));
out.high = tones.slice();
now = 10500; a.observe(snap([ev('4', 'critical')]));       // within 1.5 s: suppressed
out.rateLimited = tones.length;
now = 20000; a.observe(snap([ev('5', 'medium')]));         // medium: silent
out.medium = tones.length;
now = 30000; a.observe(snap([], 'active'));                // lockdown starts
out.lockdown = tones[tones.length - 1];
now = 40000; a.observe(snap([], 'active'));                // still locked: no repeat
out.final = tones.length;
console.log(JSON.stringify(out));
"""


def test_audio_cue_behaviour(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    h = tmp_path / "h.js"
    h.write_text(HARNESS)
    r = subprocess.run([node, str(h), str(ROOT / "core/web/static/js/audio.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["disabled"] == 0
    assert out["high"] == [1]
    assert out["rateLimited"] == 1 and out["medium"] == 1
    assert out["lockdown"] == 4 and out["final"] == 2


def test_hud_wiring():
    html = (ROOT / "core/web/static/index.html").read_text()
    assert 'id="audio-toggle"' in html and html.index("js/audio.js") < html.index("js/app.js")
    assert "alertAudio.observe(snap)" in (ROOT / "core/web/static/js/app.js").read_text()
    assert '"iso": getattr(log, "iso", "")' in (ROOT / "core/web_bridge.py").read_text()
