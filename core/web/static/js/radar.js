/**
 * KAIA // PASSIVE-DISCOVERY RADAR
 * Each LAN asset seen in the last 24 h is plotted by meaning: distance from the
 * centre is how long ago it was last seen (log scale: 5 min, 1 h, 24 h rings),
 * the sector is how it was discovered (ARP, mDNS, SSDP, LLMNR, NBNS). Position
 * within a sector is a stable hash of the address. Not geography or topology. Block nodes and
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
    this.inputDropRules = 0;
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
    this.inputDropRules = Number.isFinite(Number(data.active_blocks)) ? Number(data.active_blocks) : 0;
    this.gateOffline = Boolean(data.gate_offline);
    if (this.metaEl) {
      const state = this.gateOffline ? ' · GATE OFFLINE' : '';
      const nodes = Number.isInteger(data.lan_nodes_24h) ? data.lan_nodes_24h : this.assets.length;
      this.metaEl.textContent = `${nodes} LAN NODES (24 H) · ${this.blocks.length} RECENT BLOCKS${state}`;
    }
    this.renderRisk();
    // The reputation column only takes space when there is something to score.
    const wrap = this.canvas && this.canvas.parentElement;
    if (wrap) wrap.classList.toggle('has-risk', this.blocks.length > 0);
  }

  renderRisk() {
    if (!this.riskEl) return;
    const rows = this.blocks.slice(0, 4).map((block) => {
      const raw = block.reputation_score;
      const score = typeof raw === 'number' && Number.isFinite(raw)
        ? Math.max(0, Math.min(100, raw))
        : null;
      const detailParts = [];
      if (Array.isArray(block.reputation_tags) && block.reputation_tags.length) {
        detailParts.push(`Local reputation: ${block.reputation_tags.join(', ')}`);
      }
      if (Array.isArray(block.tags) && block.tags.length) {
        detailParts.push(`InternetDB tags: ${block.tags.join(', ')}`);
      }
      if (Array.isArray(block.ports) && block.ports.length) {
        detailParts.push(`InternetDB ports: ${block.ports.join(', ')}`);
      }
      const tagText = detailParts.join(' | ') || 'No additional reputation details';
      if (score === null) {
        return `<div class="risk-row" title="${window.esc(tagText)}"><span class="risk-ip" title="${window.esc(block.ip)}">${window.esc(block.ip)}</span><span class="risk-unknown">UNRATED</span><b class="risk-score">--</b></div>`;
      }
      // Reputation scale is 0=malicious and 100=clean (see threat_intel.py).
      const cls = score <= 30 ? 'bad' : score <= 65 ? 'warn' : '';
      return `<div class="risk-row" title="${window.esc(tagText)}">`
        + `<span class="risk-ip">${window.esc(block.ip)}</span>`
        + `<svg class="risk-meter" viewBox="0 0 100 8" preserveAspectRatio="none" role="img" aria-label="IP reputation score ${score} out of 100; higher is cleaner">`
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

  /** Ring radius (0..1) for a last-seen age: log scale, now at the centre, 24 h at the edge. */
  static ageRadius(ageSec) {
    const a = Math.max(0, Math.min(86400, ageSec));
    return 0.14 + 0.82 * (Math.log10(a + 1) / Math.log10(86401));
  }

  draw(nowMs) {
    if (!this.ctx || !this.width || !this.height) return;
    const { ctx, width: w, height: h } = this;
    const dt = Math.max(0, Math.min(0.1, (nowMs - this.lastNow) / 1000));
    this.lastNow = nowMs;
    this.angle = (this.angle + dt * 0.6) % (Math.PI * 2);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#03070c';
    ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2 + 4;
    const r = Math.max(10, Math.min(w * 0.46, h * 0.44));

    // Rings are last-seen age: 5 min, 1 h, 24 h (log scale).
    ctx.lineWidth = 1;
    ctx.font = '9px ui-monospace, monospace';
    for (const [sec, label] of [[300, '5m'], [3600, '1h'], [86400, '24h']]) {
      const rr = r * NetworkRadarCanvas.ageRadius(sec);
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.22)';
      ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = 'rgba(114, 131, 164, 0.9)';
      ctx.fillText(label, cx + 3, cy - rr + 10);
    }

    // Sectors are how a device was discovered.
    const sectors = NetworkRadarCanvas.SECTORS;
    const span = (Math.PI * 2) / sectors.length;
    sectors.forEach((name, i) => {
      const a0 = -Math.PI / 2 + i * span;
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.10)';
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a0) * r, cy + Math.sin(a0) * r); ctx.stroke();
      const am = a0 + span / 2;
      ctx.fillStyle = 'rgba(0, 240, 255, 0.55)';
      ctx.textAlign = 'center';
      ctx.fillText(name, cx + Math.cos(am) * (r + 2), cy + Math.sin(am) * (r + 2) + 3);
    });
    ctx.textAlign = 'left';

    // Sweep.
    for (let i = 8; i >= 0; i--) {
      const a = this.angle - i * 0.035;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, a - 0.025, a + 0.025); ctx.closePath();
      ctx.fillStyle = `rgba(0, 240, 255, ${0.01 + (8 - i) * 0.006})`;
      ctx.fill();
    }

    const now = Date.now();
    const placed = [];
    const plot = (ip, label, sector, ageSec, kind) => {
      const idx = Math.max(0, sectors.indexOf(sector));
      const a0 = -Math.PI / 2 + idx * span;
      const hash = this.hash(ip);
      const angle = a0 + span * (0.08 + 0.84 * ((hash % 1000) / 1000));
      // Small stable radial jitter separates devices last seen at the same time.
      const dist = r * Math.min(0.98, NetworkRadarCanvas.ageRadius(ageSec) + (((hash >>> 10) % 7) - 3) * 0.012);
      const x = cx + Math.cos(angle) * dist, y = cy + Math.sin(angle) * dist;
      const color = kind === 'block' ? '#ff3860'
        : ageSec < 600 ? '#3dff8a' : ageSec < 3600 ? '#2bbf6a' : '#1f7a48';
      ctx.save();
      ctx.shadowBlur = ageSec < 600 || kind === 'block' ? 8 : 0;
      ctx.shadowColor = color;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, kind === 'block' ? 4 : 3.2, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      // Labels: skip one that would overlap an earlier label.
      const tx = Math.max(2, Math.min(w - 92, x + 6)), ty = Math.max(10, Math.min(h - 4, y + 3));
      if (placed.some(([px, py]) => Math.abs(px - tx) < 70 && Math.abs(py - ty) < 10)) return;
      placed.push([tx, ty]);
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = kind === 'block' ? '#ff8aa0' : '#c6f7d8';
      ctx.fillText(label, tx, ty);
    };
    const ageOf = (when) => {
      const t = Date.parse(when || '');
      return Number.isFinite(t) ? Math.max(0, (now - t) / 1000) : 86400;
    };
    const shortName = (a) => {
      const host = String(a.hostname || '').split('.')[0];
      const name = host && host.toLowerCase() !== 'unknown' ? host.slice(0, 12) : '';
      const octet = String(a.ip).includes('.') ? '.' + String(a.ip).split('.').pop() : String(a.ip).slice(-6);
      return name ? `${octet} ${name}` : octet;
    };
    this.assets
      .filter((a) => ageOf(a.last_seen) <= 86400)
      .sort((a, b) => ageOf(a.last_seen) - ageOf(b.last_seen))      // newest labels win
      .slice(0, 40)
      .forEach((a) => plot(a.ip, shortName(a), NetworkRadarCanvas.sectorOf(a.detection_vector), ageOf(a.last_seen), 'asset'));
    this.blocks.slice(0, 8).forEach((b) => plot(b.ip, `${b.ip} BLOCKED`, 'OTHER', ageOf(b.timestamp || b.time), 'block'));

    ctx.fillStyle = '#00f0ff';
    ctx.shadowBlur = 8; ctx.shadowColor = '#00f0ff';
    ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = '#7fd8ff';
    ctx.textAlign = 'center';
    ctx.fillText('THIS HOST', cx, cy + 14);
    ctx.fillStyle = this.gateOffline ? '#ff3860' : '#7283a4';
    ctx.fillText(this.gateOffline ? 'GATE OFFLINE' : 'ring = last seen · sector = how discovered', cx, h - 3);
    ctx.textAlign = 'left';
  }
}
NetworkRadarCanvas.SECTORS = ['ARP', 'mDNS', 'SSDP', 'LLMNR', 'NBNS', 'OTHER'];
NetworkRadarCanvas.sectorOf = (vector) => {
  const v = String(vector || '').toUpperCase();
  const hit = NetworkRadarCanvas.SECTORS.find((s) => s.toUpperCase() === v || (s === 'NBNS' && v.startsWith('NB')));
  return hit || 'OTHER';
};
window.NetworkRadarCanvas = NetworkRadarCanvas;
