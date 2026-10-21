/**
 * KAIA // PASSIVE-DISCOVERY RADAR
 * Nodes are deterministic visual placements for observed LAN assets, not
 * geographic coordinates or inferred network connections. Block nodes and
 * reputation meters use only records supplied by ThreatIntelCollector.
 */
class NetworkRadarCanvas {
  constructor(canvas, metaEl, riskEl) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.metaEl = metaEl;
    this.riskEl = riskEl;
    this.assets = [];
    this.blocks = [];
    this.activeBlocks = 0;
    this.gateOffline = false;
    this.lastNow = performance.now();
    this.angle = 0;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    if (canvas) {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
    }
    requestAnimationFrame((now) => this.frame(now));
  }

  resize() {
    if (!this.canvas || !this.ctx) return;
    const r = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = r.width;
    this.height = r.height;
    this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  update(threatIntel) {
    const data = threatIntel || {};
    this.assets = Array.isArray(data.recent_assets) ? data.recent_assets.filter((x) => x && x.ip) : [];
    this.blocks = Array.isArray(data.recent_blocks) ? data.recent_blocks.filter((x) => x && x.ip) : [];
    this.activeBlocks = Number.isFinite(Number(data.active_blocks)) ? Number(data.active_blocks) : 0;
    this.gateOffline = Boolean(data.gate_offline);
    if (this.metaEl) {
      const state = this.gateOffline ? ' · GATE OFFLINE' : '';
      this.metaEl.textContent = `${this.assets.length} LAN NODES · ${this.blocks.length} RECENT BLOCKS${state}`;
    }
    this.renderRisk();
  }

  renderRisk() {
    if (!this.riskEl) return;
    const rows = this.blocks.slice(0, 4).map((block) => {
      const raw = Number(block.reputation_score);
      const score = Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : null;
      if (score === null) {
        return `<div class="risk-row"><span class="risk-ip" title="${window.esc(block.ip)}">${window.esc(block.ip)}</span><span class="risk-unknown">NO SCORE</span><b class="risk-score">--</b></div>`;
      }
      const cls = score >= 70 ? 'bad' : score >= 35 ? 'warn' : '';
      return `<div class="risk-row" title="${window.esc(block.threat_level || 'IP reputation')}">`
        + `<span class="risk-ip">${window.esc(block.ip)}</span>`
        + `<svg class="risk-meter" viewBox="0 0 100 8" preserveAspectRatio="none" role="img" aria-label="IP reputation ${score} out of 100">`
        + `<rect class="risk-track" x="0" y="0" width="100" height="8" rx="2"></rect>`
        + `<rect class="risk-fill ${cls}" x="0" y="0" width="${score}" height="8" rx="2"></rect></svg>`
        + `<b class="risk-score">${score}</b></div>`;
    });
    this.riskEl.innerHTML = rows.length ? rows.join('') : '<div class="radar-risk-empty">No scored block records.</div>';
  }

  hash(value) {
    let h = 2166136261;
    for (const ch of String(value)) {
      h ^= ch.charCodeAt(0);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  frame(now) {
    if (!document.hidden) this.draw(now);
    requestAnimationFrame((next) => this.frame(next));
  }

  draw(nowMs) {
    if (!this.ctx || !this.width || !this.height) return;
    const { ctx, width: w, height: h } = this;
    const dt = Math.max(0, Math.min(0.1, (nowMs - this.lastNow) / 1000));
    this.lastNow = nowMs;
    this.angle = (this.angle + dt * 0.75) % (Math.PI * 2);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#03070c';
    ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const r = Math.max(8, Math.min(w * 0.42, h * 0.43));
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    for (const f of [0.35, 0.67, 1]) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * f, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
    ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.18)';
    ctx.stroke();

    // Rotating sweep with a short phosphor trail.
    for (let i = 8; i >= 0; i--) {
      const a = this.angle - i * 0.035;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, a - 0.025, a + 0.025);
      ctx.closePath();
      ctx.fillStyle = `rgba(0, 240, 255, ${0.012 + (8 - i) * 0.008})`;
      ctx.fill();
    }

    const plotNode = (ip, kind, i) => {
      const hash = this.hash(ip);
      const angle = ((hash % 360) / 180) * Math.PI + (kind === 'block' ? 0.12 : 0);
      const dist = r * (0.25 + ((hash >>> 9) % 60) / 100);
      const x = cx + Math.cos(angle) * dist;
      const y = cy + Math.sin(angle) * dist;
      const color = kind === 'block' ? '#ff3860' : '#3dff8a';
      ctx.save();
      ctx.shadowBlur = 7;
      ctx.shadowColor = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, kind === 'block' ? 2.5 : 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      if (i < 2) {
        ctx.font = '7px ui-monospace, monospace';
        ctx.fillStyle = color;
        ctx.fillText(String(ip), Math.max(1, Math.min(w - 60, x + 4)), Math.max(7, Math.min(h - 3, y - 4)));
      }
    };
    this.assets.slice(0, 20).forEach((asset, i) => plotNode(asset.ip, 'asset', i));
    this.blocks.slice(0, 5).forEach((block, i) => plotNode(block.ip, 'block', i));
    ctx.fillStyle = '#00f0ff';
    ctx.shadowBlur = 6;
    ctx.shadowColor = '#00f0ff';
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = '7px ui-monospace, monospace';
    ctx.fillStyle = this.gateOffline ? '#ff3860' : '#7283a4';
    ctx.textAlign = 'center';
    ctx.fillText(this.gateOffline ? 'GATE OFFLINE' : `ACTIVE BLOCKS ${this.activeBlocks}`, cx, h - 4);
    ctx.textAlign = 'left';
  }
}
window.NetworkRadarCanvas = NetworkRadarCanvas;
