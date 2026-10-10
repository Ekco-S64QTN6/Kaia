/**
 * Kaia Cyber-Sentinel: panel helpers
 * HTML escaping, fit-to-panel tables (no scrollbars) and audit-line parsing.
 */

/** Escape untrusted text (process names, paths, hostnames, log lines) for innerHTML. */
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

/**
 * A table whose body shows as many rows as fit its panel, newest first.
 * render() returns {shown, total} so callers can print "showing N of M".
 */
class FitTable {
  constructor(tbodyId, columns) {
    this.tbody = document.getElementById(tbodyId);
    this.columns = columns;
    this.box = this.tbody ? this.tbody.closest('.table-fit') : null;
    this.capacity = 8;
    this.lastRows = null;
    if (this.box) {
      new ResizeObserver(() => this._measure()).observe(this.box);
      this._measure();
    }
  }

  _measure() {
    const head = this.box.querySelector('thead');
    const rowH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--row-h')) || 21;
    const avail = this.box.clientHeight - (head ? head.offsetHeight : 0);
    this.capacity = Math.max(1, Math.floor(avail / rowH));
    if (this.lastRows) this.render(this.lastRows.rows, this.lastRows.empty);
  }

  /** rows: array of row HTML strings (already escaped), newest first. */
  render(rows, emptyText = 'No data') {
    this.lastRows = { rows, empty: emptyText };
    if (!this.tbody) return { shown: 0, total: 0 };
    const shown = rows.slice(0, this.capacity);
    this.tbody.innerHTML = shown.length
      ? shown.join('')
      : `<tr><td class="empty" colspan="${this.columns}">${esc(emptyText)}</td></tr>`;
    return { shown: shown.length, total: rows.length };
  }
}

/**
 * Parse Policy Gate audit lines such as
 *   "✗ FILE_WRITE_VIOLATION │ denied   │ hash:4524e694 (policy_gate)"
 * into columns. Unrecognised lines fall back to the raw message.
 */
function parseAuditLine(log) {
  const msg = String(log.message || '');
  const parts = msg.split('│').map((p) => p.trim());
  let event = msg, result = '', source = '', hash = '';
  if (parts.length >= 2) {
    event = parts[0].replace(/^[✓✗⚠•\s]+/, '');
    result = parts[1].toLowerCase();
    const tail = parts.slice(2).join(' ');
    const m = tail.match(/^(?:hash:(\S+))?\s*\(([^)]+)\)\s*$/);
    if (m) { hash = m[1] || ''; source = m[2]; } else { source = tail; }
  }
  const level = String(log.level || '').toUpperCase();
  if (!result) {
    if (level === 'ERROR' || level === 'CRITICAL') result = 'error';
    else if (level === 'SUCCESS') result = 'ok';
    else result = level.toLowerCase() || 'info';
  }
  return { time: log.timestamp || '', event, result, source, hash, level };
}

function resultTag(result) {
  const r = String(result).toLowerCase();
  const cls = /denied|error|blocked|violation|critical/.test(r) ? 'bad'
    : /approved|ok|success|allowed/.test(r) ? 'ok'
    : /warn|sandbox/.test(r) ? 'warn' : 'neutral';
  return `<span class="tag ${cls}">${esc(r.toUpperCase())}</span>`;
}

/** "3m ago" style relative time from an ISO string or unix seconds. */
function ago(when) {
  const t = typeof when === 'number' ? when * 1000 : Date.parse(when);
  if (!isFinite(t)) return '--';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function clockTime(unixSec) {
  return new Date(unixSec * 1000).toTimeString().slice(0, 8);
}

window.esc = esc;
window.FitTable = FitTable;
window.parseAuditLine = parseAuditLine;
window.resultTag = resultTag;
window.ago = ago;
window.clockTime = clockTime;
