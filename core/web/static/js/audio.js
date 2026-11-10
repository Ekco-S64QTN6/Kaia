/**
 * KAIA // AUDIO CUES (opt-in)
 * Short Web Audio tones for a new high or critical security event and for a
 * lockdown. Off by default; the header toggle turns them on (browsers allow
 * audio only after a user gesture). Tones are synthesised: no audio files, and
 * nothing outside the self-hosted CSP. Events already present when the HUD
 * loads are never announced.
 */
class AlertAudio {
  constructor(button) {
    this.button = button;
    this.ctx = null;
    this.seen = new Set();
    this.primed = false;
    this.lastLock = null;
    this.lastTone = 0;
    this.enabled = false;
    try { this.enabled = window.localStorage.getItem('kaia-audio') === 'on'; } catch (e) { /* storage blocked */ }
    if (this.button) {
      this.button.addEventListener('click', () => this.toggle());
      this.paint();
    }
  }

  toggle() {
    this.enabled = !this.enabled;
    try { window.localStorage.setItem('kaia-audio', this.enabled ? 'on' : 'off'); } catch (e) { /* ignore */ }
    if (this.enabled) this.tone([660], 0.12, 0.05);   // confirm, and unlock audio within the gesture
    this.paint();
  }

  paint() {
    if (!this.button) return;
    this.button.innerHTML = `AUDIO <b class="${this.enabled ? 'ok' : ''}">${this.enabled ? 'on' : 'off'}</b>`;
    this.button.setAttribute('aria-pressed', String(this.enabled));
  }

  tone(freqs, duration, gap) {
    const now = Date.now();
    if (now - this.lastTone < 1500) return;          // never a barrage
    this.lastTone = now;
    try {
      if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      let t = this.ctx.currentTime;
      for (const f of freqs) {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = f;
        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.15, t + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);
        osc.connect(gain).connect(this.ctx.destination);
        osc.start(t);
        osc.stop(t + duration + 0.02);
        t += duration + gap;
      }
    } catch (e) { /* no audio device: stay silent */ }
  }

  observe(snap) {
    const logs = Array.isArray(snap.logs) ? snap.logs : [];
    let fresh = null;
    for (const log of logs) {
      const level = log.alert_level;
      if (level !== 'critical' && level !== 'high') continue;
      const key = `${log.iso || log.timestamp}|${log.event}|${log.detail}`;
      if (this.seen.has(key)) continue;
      this.seen.add(key);
      if (this.primed && (!fresh || level === 'critical')) fresh = level;
    }
    if (this.seen.size > 2000) this.seen = new Set(Array.from(this.seen).slice(-1000));
    const lock = (snap.system_security || {}).lockdown_state;
    const lockStarted = this.primed && lock === 'active' && this.lastLock !== 'active';
    this.lastLock = lock;
    this.primed = true;
    if (!this.enabled) return;
    if (lockStarted) this.tone([880, 440, 880, 440], 0.18, 0.04);
    else if (fresh === 'critical') this.tone([988, 740], 0.16, 0.05);
    else if (fresh === 'high') this.tone([740], 0.14, 0.05);
  }
}
window.AlertAudio = AlertAudio;
