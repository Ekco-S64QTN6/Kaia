/**
 * KAIAGOTCHI CYBER-DECK // TIME-SERIES CHARTS
 * Network-analyzer styled canvas line charts: thin line, recessive grid,
 * no glow, no fill. Only real readings are plotted (no seeded data).
 */

const INK = { grid: 'rgba(60, 80, 112, 0.28)', axis: 'rgba(120, 140, 170, 0.75)', muted: 'rgba(120, 140, 170, 0.5)' };
const MONO = '"JetBrains Mono", "Share Tech Mono", monospace';

function niceMax(v, floor) {
  if (v <= floor) return floor;
  const steps = [1, 2, 2.5, 5, 10];
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  for (const s of steps) if (s * mag >= v) return s * mag;
  return 10 * mag;
}

function fmtCompact(v) {
  if (Math.abs(v) >= 10000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}K`;
  return `${Math.round(v)}`;
}

function fmtClock(sec) {
  const d = new Date(sec * 1000);
  return `${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

class LineChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} o  color, windowSec, yMin, yMax, yTicks (array) | autoMax floor,
   *                    axes (bool), label (string|fn), unit, minimal (sparkline)
   */
  constructor(canvas, o) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = { windowSec: 60, axes: true, unit: '', ...o };
    this.points = [];
    this.hoverX = null;
    this.dirty = true;

    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
    if (this.o.axes) {
      canvas.addEventListener('mousemove', (e) => {
        const r = canvas.getBoundingClientRect();
        this.hoverX = e.clientX - r.left;
      });
      canvas.addEventListener('mouseleave', () => (this.hoverX = null));
    }
    this.frameIntervalMs = Number.isFinite(this.o.frameIntervalMs) ? Math.max(16, this.o.frameIntervalMs) : 66;
    this._lastDrawAt = 0;
    const tick = (now) => {
      if (now - this._lastDrawAt >= this.frameIntervalMs) {
        this.draw();
        this._lastDrawAt = now;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
  }

  push(value, t = Date.now() / 1000) {
    if (typeof value !== 'number' || !isFinite(value)) return;
    this.points.push({ t, v: value });
    const cutoff = t - this.o.windowSec - 2;
    while (this.points.length && this.points[0].t < cutoff) this.points.shift();
  }

  average() {
    const now = Date.now() / 1000;
    const pts = this.points.filter((p) => p.t >= now - this.o.windowSec);
    return pts.length ? pts.reduce((a, p) => a + p.v, 0) / pts.length : null;
  }

  _yRange() {
    const o = this.o;
    if (o.autoMax) {
      const max = Math.max(0, ...this.points.map((p) => p.v));
      const top = niceMax(max * 1.08, o.autoMax);
      const steps = this.h < 110 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];   // small charts: fewer ticks
      return { min: 0, max: top, ticks: steps.map((f) => f * top) };
    }
    return { min: o.yMin, max: o.yMax, ticks: o.yTicks || [] };
  }

  draw() {
    const { ctx, w, h, o } = this;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);

    const pad = o.axes ? { l: 42, r: 8, t: o.label ? 18 : 6, b: 18 } : { l: 0, r: 0, t: 3, b: 3 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const now = Date.now() / 1000;
    const t0 = now - o.windowSec;
    const yr = this._yRange();
    const X = (t) => pad.l + ((t - t0) / o.windowSec) * pw;
    const Y = (v) => pad.t + (1 - (Math.min(Math.max(v, yr.min), yr.max) - yr.min) / (yr.max - yr.min)) * ph;

    ctx.font = `10px ${MONO}`;
    ctx.lineWidth = 1;

    if (o.axes) {
      ctx.strokeStyle = INK.grid;
      ctx.fillStyle = INK.axis;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const v of yr.ticks) {
        const y = Math.round(Y(v)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
        ctx.fillText(o.tickFormat ? o.tickFormat(v) : fmtCompact(v), pad.l - 6, y);
      }
      // x ticks every 15 s, aligned to wall clock
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const step = o.windowSec / 4;
      for (let t = Math.ceil(t0 / step) * step; t <= now; t += step) {
        const x = Math.round(X(t)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + ph);
        ctx.stroke();
        if (x > pad.l + 16 && x < w - pad.r - 14) ctx.fillText(fmtClock(t), x, pad.t + ph + 4);
      }
      if (o.label) {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillStyle = INK.axis;
        ctx.fillText(typeof o.label === 'function' ? o.label() : o.label, pad.l + 2, 2);
      }
    }

    if (o.threshold !== undefined && o.threshold >= yr.min && o.threshold <= yr.max) {
      const ty = Y(o.threshold);
      ctx.save();
      ctx.strokeStyle = o.thresholdColor || '#ff3860';
      ctx.fillStyle = o.thresholdColor || '#ff3860';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(pad.l, Math.round(ty) + 0.5);
      ctx.lineTo(w - pad.r, Math.round(ty) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.font = `9px ${MONO}`;
      ctx.fillText(o.thresholdLabel || `THRESHOLD ${o.threshold}`, w - pad.r - 2, ty - 2);
      ctx.restore();
    }

    const pts = this.points.filter((p) => p.t >= t0 - 1);
    if (pts.length < 2) {
      if (o.axes) {
        ctx.fillStyle = INK.muted;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('AWAITING DATA', pad.l + pw / 2, pad.t + ph / 2);
      }
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, 0, pw, h);
    ctx.clip();
    // Optional translucent area fill for the LPS oscillogram.
    if (o.areaFill) {
      const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
      grad.addColorStop(0, o.areaTop || 'rgba(61, 255, 138, 0.24)');
      grad.addColorStop(1, o.areaBottom || 'rgba(61, 255, 138, 0.015)');
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.v)) : ctx.moveTo(X(p.t), Y(p.v))));
      ctx.lineTo(X(pts[pts.length - 1].t), pad.t + ph);
      ctx.lineTo(X(pts[0].t), pad.t + ph);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    ctx.strokeStyle = o.color;
    ctx.lineWidth = o.axes ? 1.5 : 1.25;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.v)) : ctx.moveTo(X(p.t), Y(p.v))));
    ctx.stroke();
    ctx.restore();

    // hover: crosshair + value at nearest reading
    if (this.hoverX !== null && this.hoverX > pad.l && this.hoverX < w - pad.r) {
      const t = t0 + ((this.hoverX - pad.l) / pw) * o.windowSec;
      let best = pts[0];
      for (const p of pts) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
      const x = X(best.t), y = Y(best.v);
      ctx.strokeStyle = INK.axis;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, pad.t);
      ctx.lineTo(Math.round(x) + 0.5, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = o.color;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
      const text = `${fmtClock(best.t)}  ${o.valueFormat ? o.valueFormat(best.v) : fmtCompact(best.v)}${o.unit ? ' ' + o.unit : ''}`;
      ctx.font = `11px ${MONO}`;
      const tw = ctx.measureText(text).width + 12;
      const bx = Math.min(Math.max(x - tw / 2, pad.l), w - pad.r - tw);
      const by = y - 26 < pad.t ? y + 8 : y - 26;
      ctx.fillStyle = 'rgba(6, 10, 18, 0.94)';
      ctx.strokeStyle = INK.grid;
      ctx.fillRect(bx, by, tw, 18);
      ctx.strokeRect(bx + 0.5, by + 0.5, tw - 1, 17);
      ctx.fillStyle = '#e8f0fe';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, bx + 6, by + 9);
    }
  }
}

window.LineChart = LineChart;
window.fmtCompact = fmtCompact;
