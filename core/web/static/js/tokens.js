/**
 * KAIA // CAPABILITY TOKEN LEDGER
 * Compact cards expose the captured token fingerprint, capability, target and
 * signed expiry time. TTL rings consume the snapshot's expiry timestamps.
 */
class CapabilityTokenLedger {
  constructor(container) { this.container = container; }

  render(tokens) {
    if (!this.container) return;
    const live = (Array.isArray(tokens) ? tokens : [])
      .filter((token) => token && !token.is_expired && Number(token.ttl_seconds) > 0)
      .slice(0, 8);
    if (!live.length) {
      this.container.innerHTML = '<div class="token-empty">No active capability tokens</div>';
      return;
    }
    this.container.innerHTML = live.map((token) => {
      const ttl = Math.max(0, Math.floor(Number(token.ttl_seconds) || 0));
      const minutes = Math.floor(ttl / 60);
      const seconds = ttl % 60;
      const label = `${minutes}:${String(seconds).padStart(2, '0')}`;
      const issuedAt = Date.parse(token.issued_at || '');
      const expiresAt = Date.parse(token.expires || '');
      const duration = expiresAt > issuedAt ? (expiresAt - issuedAt) / 1000 : Math.max(ttl, 1);
      const fraction = Math.max(0, Math.min(1, ttl / duration));
      const circumference = 81.68;
      const offset = circumference * (1 - fraction);
      const stateClass = fraction <= 0.15 ? 'bad' : fraction <= 0.4 ? 'warn' : '';
      const id = window.esc(token.id || '--------');
      const capability = window.esc(token.capability || 'UNKNOWN');
      const target = window.esc(token.target || '*');
      const fingerprint = window.esc(token.signature || token.id || '');
      return `<article class="token-card" title="Signature fingerprint: ${fingerprint}">`
        + `<svg class="token-ring" viewBox="0 0 40 40" role="img" aria-label="Time to expiry ${label}">`
        + `<circle class="token-ring-track" cx="20" cy="20" r="13"></circle>`
        + `<circle class="token-ring-progress ${stateClass}" cx="20" cy="20" r="13" transform="rotate(-90 20 20)" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"></circle>`
        + `<text x="20" y="20">${label}</text></svg>`
        + `<div class="token-card-copy"><div class="token-id">${id}</div><div class="token-scope" title="${capability}">${capability}</div><div class="token-target" title="${target}">${target}</div></div></article>`;
    }).join('');
  }
}
window.CapabilityTokenLedger = CapabilityTokenLedger;
