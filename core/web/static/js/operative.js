/**
 * Kaia Cyber-Sentinel: Tactical Sentinel Operative (Kaia Avatar) Controller
 * Real-time security posture to avatar biometric reaction state machine.
 *
 * The character itself is the shared Kaia codec avatar (js/kaia/): a rigged,
 * WebGL2-rendered portrait whose expression, gaze, breathing and FX are driven by
 * the security state (js/kaia/kaia-sentinel-states.js). It mounts automatically
 * into #kaia-character-container. mountAnimationViewport() still lets another
 * renderer take over the stage.
 */

class SentinelAvatarController {
  constructor(viewportId, glowId, pulseId, conduitId, stanceId, badgeId) {
    this.viewport = document.getElementById(viewportId);
    this.glowBorder = document.getElementById(glowId);
    this.pulseEl = document.getElementById(pulseId);
    this.conduitEl = document.getElementById(conduitId);
    this.stanceEl = document.getElementById(stanceId);
    this.badgeEl = document.getElementById(badgeId);

    this.currentState = 'SentinelWatch';
    this.pulseBpm = 60;
    this.pulseTimer = null;
    this.characterRenderer = null;
    this.stateChangeListeners = [];
    this._mountCodec();
  }

  /** Mount the shared Kaia codec avatar as the default character renderer. */
  _mountCodec() {
    const container = document.getElementById('kaia-character-container');
    if (!container || !window.KaiaCodec) return;
    const face = new URLSearchParams(location.search).get('face') || undefined;   // ?face=k59 etc.
    let codec = null;
    this.mountAnimationViewport({
      init: (el, state) => {
        window.KaiaCodec.mount(el, { portrait: face })
          .then((c) => { codec = c; c.setMood(this.currentState || state); })
          .catch((err) => console.error('[KaiaAvatar] codec failed to mount:', err));
      },
      onStateTransition: (next) => {
        if (!codec) return;
        codec.setMood(next);
        if (next === 'ContainmentAlert' || next === 'LockdownMode') codec.react('alert');
      },
      setLink: (ok) => codec && codec.setLink(ok),
    });
  }

  /**
   * Registers a callback triggered when the Sentinel state changes.
   */
  onStateChange(callback) {
    if (typeof callback === 'function') {
      this.stateChangeListeners.push(callback);
    }
  }

  /**
   * Mount hook for Claude Opus 5.5 to plug in the 2D/3D character engine.
   */
  mountAnimationViewport(renderer) {
    this.characterRenderer = renderer;
    const container = document.getElementById('kaia-character-container');
    if (container && typeof renderer.init === 'function') {
      renderer.init(container, this.currentState);
    }
    console.info('[KaiaAvatar] character renderer mounted.');
  }

  /**
   * Updates operative state based on live KaiamonSnapshot telemetry.
   */
  updatePosture(snapshot) {
    const nextState = snapshot.sentinel_state || 'SentinelWatch';
    const oldState = this.currentState;

    if (nextState !== oldState) {
      this.currentState = nextState;
      this.applyStateVisuals(nextState, snapshot);
      
      // Notify external listeners (including Opus 5.5 renderer)
      for (const listener of this.stateChangeListeners) {
        try {
          listener(nextState, oldState, snapshot);
        } catch (e) {
          console.error('[KaiaAvatar] Listener error:', e);
        }
      }
      
      if (this.characterRenderer && typeof this.characterRenderer.onStateTransition === 'function') {
        this.characterRenderer.onStateTransition(nextState, oldState);
      }
    }
  }

  applyStateVisuals(state, snapshot) {
    let conduitText = 'EMERALD / 60 BPM (STABLE)';
    let stanceText = 'SENTINEL WATCH';
    let badgeText = 'STATUS: SENTINEL WATCH';
    let badgeClass = 'state-badge';
    let glowClass = 'conduit-glow-border';
    this.pulseBpm = 60;

    switch (state) {
      case 'PolicyEvaluating':
        conduitText = 'CYAN / 90 BPM (INTENT EVALUATION)';
        stanceText = 'ANALYZING INTENT';
        badgeText = 'POLICY: EVALUATING';
        badgeClass = 'state-badge';
        glowClass = 'conduit-glow-border evaluating';
        this.pulseBpm = 90;
        break;

      case 'AnomalyDetected':
        conduitText = 'AMBER / 95 BPM (ANOMALY DETECTED)';
        stanceText = 'THREAT PROBING';
        badgeText = 'ANOMALY: SKEPTICAL';
        badgeClass = 'state-badge alert';
        glowClass = 'conduit-glow-border anomaly';
        this.pulseBpm = 95;
        break;

      case 'ContainmentAlert':
        conduitText = 'CRIMSON / 115 BPM (VIOLATION DEFENSE)';
        stanceText = 'DEFENSIVE RECOIL';
        badgeText = 'VIOLATION: REJECTED';
        badgeClass = 'state-badge alert';
        glowClass = 'conduit-glow-border alert';
        this.pulseBpm = 115;
        break;

      case 'LockdownMode':
        conduitText = 'STROBE RED / 135 BPM (LOCKDOWN ACTIVE)';
        stanceText = 'COMBAT LOCKDOWN SHIELD';
        badgeText = 'LOCKDOWN: CONTAINED';
        badgeClass = 'state-badge alert';
        glowClass = 'conduit-glow-border alert';
        this.pulseBpm = 135;
        break;

      case 'AuditSurge':
        conduitText = 'EMERALD MATRIX / 105 BPM (DATA SURGE)';
        stanceText = 'HIGH VELOCITY AUDIT';
        badgeText = 'AUDIT SURGE (> 30 LPS)';
        badgeClass = 'state-badge';
        glowClass = 'conduit-glow-border';
        this.pulseBpm = 105;
        break;

      case 'SleepyIdle':
        conduitText = 'PALE TEAL / 35 BPM (STANDBY)';
        stanceText = 'POWER-SAVE STANDBY';
        badgeText = 'STANDBY: POWER-SAVE';
        badgeClass = 'state-badge';
        glowClass = 'conduit-glow-border';
        this.pulseBpm = 35;
        break;

      case 'SentinelWatch':
      default:
        conduitText = 'EMERALD / 60 BPM (STABLE WATCH)';
        stanceText = 'VIGILANT DEFENSE';
        badgeText = 'STATUS: SENTINEL WATCH';
        badgeClass = 'state-badge';
        glowClass = 'conduit-glow-border';
        this.pulseBpm = 60;
        break;
    }

    if (this.glowBorder) this.glowBorder.className = glowClass;
    if (this.conduitEl) this.conduitEl.textContent = conduitText;
    if (this.stanceEl) this.stanceEl.textContent = stanceText;
    if (this.badgeEl) {
      this.badgeEl.textContent = badgeText;
      this.badgeEl.className = badgeClass;
    }
    if (this.pulseEl) {
      this.pulseEl.textContent = `${this.pulseBpm} BPM`;
    }
  }
}

window.SentinelAvatarController = SentinelAvatarController;
