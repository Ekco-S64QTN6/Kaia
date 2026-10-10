/**
 * KAIA // SENTINEL STATES
 *
 * Maps Kaia's 7 security postures (KaiamonSnapshot.sentinel_state) onto the
 * shared avatar engine (kaia-animator.js), following the reaction table in
 * Kaia_UI_Upgrade/2026-10-07-kaia-hardened-security-ui-upgrade-design-spec.md §5.2.
 * Heartbeat rates match the BPM readouts the dashboard shows (operative.js).
 *
 * This is the only project-specific avatar file; everything else in js/kaia/
 * is the shared engine, kept identical to Kaiagotchi's copy.
 */

(() => {
  const bpm = (n) => n / 60;
  const tint = (r, g, b) => ({ tintR: r, tintG: g, tintB: b });

  const STATES = {
    // Normal host operation: calm, steady forward gaze, soft emerald.
    SentinelWatch: { pulseRate: bpm(60) },

    // Policy Gate evaluating an intent: determined, eyes locked on the operator,
    // cyan conduits racing upward, iris reticles spinning up.
    PolicyEvaluating: {
      wide: 0.25, browInner: -0.25, browRaise: 0.1, lid: 0.04, pitch: -0.12,
      gazeX: 0, gazeY: -0.05, saccade: 0.3,
      glow: 1.35, flowSpeed: 2.6, flowDir: 1, conduit: 1.0,
      reticle: 0.9, reticleScale: 1.2, pulseRate: bpm(90), energy: 0.8,
      ...tint(0.3, 0.92, 1.0),
    },

    // eBPF anomaly / unknown process: skeptical, head tilt, furrow, scanning
    // saccades, amber conduits with micro-sparks.
    AnomalyDetected: {
      browAsym: 0.35, browInner: -0.4, lid: 0.16, squint: 0.2, roll: 0.07, yaw: -0.2,
      smile: -0.12, smileAsym: -0.22, saccade: 2.2,
      glow: 1.1, flowSpeed: 1.6, microGlitch: 1, grain: 0.09,
      reticle: 0.7, reticleScale: 0.85, pulseRate: bpm(95),
      ...tint(1.0, 0.86, 0.3),
    },

    // Policy violation / gate rejection: stern protective scowl, sharp
    // defensive recoil on entry, crimson conduits with an aberration pulse.
    ContainmentAlert: {
      browInner: -1.0, browRaise: -0.3, lid: 0.2, squint: 0.35, smile: -0.55, pitch: 0.12,
      glow: 1.5, flowSpeed: 2.4, conduit: 1.1, bloom: 0.8, aberration: 0.3, grain: 0.09,
      reticle: 0.85, reticleDouble: 1, saccade: 0.5, pulseRate: bpm(115), energy: 1.2,
      kick: { pitch: -0.9, zoom: -0.12 },
      ...tint(1.0, 0.36, 0.32),
    },

    // Tamper tripwire / lockdown: frightened but combat-ready, red emergency
    // strobe, glitch artifacts, pulled back from the glass.
    LockdownMode: {
      wide: 0.65, browRaise: 0.4, browInner: 0.45, mouthOpen: 0.1, zoom: 0.96, pitch: -0.08,
      glow: 1.6, flowSpeed: 3.2, conduit: 1.4, flicker: 1, strobe: 1, glitch: 0.22, aberration: 0.45,
      reticle: 1.0, reticleDouble: 1, reticleScale: 0.8, saccade: 1.8, pulseRate: bpm(135), energy: 1.5,
      kick: { pitch: -1.1, zoom: -0.2 },
      ...tint(1.0, 0.28, 0.26),
    },

    // Audit ledger surge: data overload, binary cascade washing down, fast
    // reading saccades.
    AuditSurge: {
      wide: 0.2, glyph: 1.0, flowSpeed: 4.2, flowDir: -1, glow: 1.8, conduit: 1.2, dither: 0.5,
      reticle: 0.8, reticleScale: 1.1, saccade: 3.0, pulseRate: bpm(105), energy: 1.2,
      ...tint(0.5, 1.0, 0.62),
    },

    // Idle / low resource: power-save standby, eyelids half-closed, head
    // resting slightly down, conduits dimmed, pale teal.
    SleepyIdle: {
      lid: 0.6, gazeY: 0.25, pitch: 0.35, roll: 0.05, mouthOpen: 0.04,
      glow: 0.15, flowSpeed: 0.3, brightness: 0.72, conduit: 0.15, glyph: 0.4, dither: 0.5, reticle: 0.1,
      breathRate: 0.12, breathAmp: 1.5, pulseRate: bpm(35),
      blinkMin: 3.5, blinkMax: 7, blinkSpeed: 0.4, saccade: 0.25, energy: 0.35,
      ...tint(0.42, 0.92, 0.84),
    },
  };

  for (const [name, preset] of Object.entries(STATES)) window.KaiaAnimator.registerMood(name, preset);
  window.KAIA_SENTINEL_STATES = Object.keys(STATES);
})();
