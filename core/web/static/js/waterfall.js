/**
 * KAIA // PROCESS EVENT WATERFALL
 * Renders only events received from the snapshot serializer. "Observed" is
 * intentionally neutral; this widget never infers that a process is safe.
 */
class SyscallWaterfallCanvas {
  constructor(canvas, metaEl, inspectorEl) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.metaEl = metaEl;
    this.inspectorEl = inspectorEl;
    this.events = [];
    this.source = '';
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    if (canvas) {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      canvas.addEventListener('mousemove', (event) => this.inspect(event));
      canvas.addEventListener('mouseleave', () => {
        if (this.inspectorEl) this.inspectorEl.textContent = 'No process event selected.';
      });
      this.resize();
    }
    requestAnimationFrame((now) => this.frame(now));
  }

  resize() {
    if (!this.canvas || !this.ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.max(1, Math.round(rect.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * this.dpr));
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  update(events, source) {
    this.events = (Array.isArray(events) ? events : [])
      .filter((event) => event && Number.isFinite(Number(event.timestamp)))
      .map((event) => ({ ...event, timestamp: Number(event.timestamp) }))
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(-60);
    this.source = String(source || 'EVENT SOURCE UNKNOWN');
    if (this.metaEl) {
      const live = this.events.filter((event) => Date.now() / 1000 - event.timestamp <= 60).length;
      this.metaEl.textContent = `${this.source.toUpperCase()} · ${live} EVENTS`;
    }
  }

  inspect(event) {
    if (!this.canvas || !this.inspectorEl || !this.width) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(this.width, event.clientX - rect.left));
    const now = Date.now() / 1000;
    const targetTime = now - 60 + (x / this.width) * 60;
    let nearest = null;
    for (const item of this.events) {
      if (!nearest || Math.abs(item.timestamp - targetTime) < Math.abs(nearest.timestamp - targetTime)) nearest = item;
    }
    if (!nearest || Math.abs(nearest.timestamp - targetTime) > 6) {
      this.inspectorEl.textContent = 'No event near cursor.';
      return;
    }
    const stamp = new Date(nearest.timestamp * 1000).toLocaleTimeString();
    this.inspectorEl.textContent = `${stamp} · ${nearest.syscall || 'process event'} · PID ${nearest.pid ?? '?'} · ${nearest.comm || 'unknown'} · ${nearest.filename || ''}`;
  }

  frame(nowMs) {
    if (!document.hidden) this.draw(nowMs / 1000);
    requestAnimationFrame((next) => this.frame(next));
  }

  draw(now) {
    if (!this.ctx || !this.width || !this.height) return;
    const { ctx, width: w, height: h } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#03070c';
    ctx.fillRect(0, 0, w, h);
    const left = Math.min(43, Math.max(36, w * 0.18));
    const right = Math.max(left + 1, w - 3);
    const plotW = right - left;
    const top = 3;
    const bottom = h - 12;
    const plotH = Math.max(1, bottom - top);

    ctx.font = '8px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#657596';
    for (let i = 0; i <= 4; i++) {
      const x = left + (plotW * i) / 4;
      ctx.strokeStyle = 'rgba(85, 110, 150, 0.23)';
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      const labels = ['-60s', '-45s', '-30s', '-15s', 'NOW'];
      ctx.textAlign = i === 4 ? 'right' : i === 0 ? 'left' : 'center';
      ctx.fillText(labels[i], i === 4 ? x - 1 : x, h - 5);
    }
    ctx.textAlign = 'left';

    const lanes = [
      { label: 'EXECVE', match: /execve/i },
      { label: 'OPENAT2', match: /openat2/i },
      { label: 'OTHER', match: /.*/ },
    ];
    lanes.forEach((lane, idx) => {
      const y = top + (plotH * (idx + 0.5)) / lanes.length;
      ctx.fillStyle = '#7283a4';
      ctx.fillText(lane.label, 2, y);
      ctx.strokeStyle = 'rgba(85, 110, 150, 0.17)';
      ctx.beginPath();
      ctx.moveTo(left, top + (plotH * idx) / lanes.length);
      ctx.lineTo(right, top + (plotH * idx) / lanes.length);
      ctx.stroke();
    });

    const minTime = now - 60;
    for (const event of this.events) {
      const age = now - event.timestamp;
      if (age < 0 || age > 60) continue;
      const syscall = String(event.syscall || '');
      const laneIndex = /execve/i.test(syscall) ? 0 : /openat2/i.test(syscall) ? 1 : 2;
      const y = top + (plotH * (laneIndex + 0.5)) / lanes.length;
      const x = left + ((event.timestamp - minTime) / 60) * plotW;
      const disposition = String(event.disposition || 'observed').toLowerCase();
      const color = disposition === 'blocked' ? '#ff3860'
        : disposition === 'sandboxed' ? '#ffe45c'
        : disposition === 'safe' ? '#3dff8a' : '#00f0ff';
      ctx.save();
      ctx.shadowBlur = 6;
      ctx.shadowColor = color;
      ctx.fillStyle = color;
      ctx.fillRect(x - 1, y - 4, 2, 8);
      ctx.restore();
    }

    if (!this.events.some((event) => event.timestamp >= now - 60)) {
      ctx.fillStyle = '#5d6b8f';
      ctx.textAlign = 'center';
      ctx.fillText('AWAITING PROCESS EVENTS', left + plotW / 2, top + plotH / 2);
      ctx.textAlign = 'left';
    }
  }
}
window.SyscallWaterfallCanvas = SyscallWaterfallCanvas;
