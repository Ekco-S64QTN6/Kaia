/**
 * Kaia Cyber-Sentinel: Interactive Operator Command Console
 * Dispatches deterministic commands to /api/command with live ANSI/cyber formatting.
 */

// Console commands are authorised by a per-start session key. The launcher
// hands it over in the URL fragment (never sent to a server, never in a log);
// it is kept for this tab only and removed from the address at once.
const SESSION_KEY = (() => {
  let key = '';
  try {
    const m = /(?:^#|&)session=([A-Za-z0-9_-]{32,128})(?:&|$)/.exec(location.hash);
    if (m) {
      key = m[1];
      sessionStorage.setItem('kaia-session', key);
      history.replaceState(null, '', location.pathname + location.search);
    } else {
      key = sessionStorage.getItem('kaia-session') || '';
    }
  } catch (e) { /* storage unavailable: commands will be refused, the HUD still reads */ }
  return key;
})();

class SentinelTerminalConsole {
  constructor(inputSelector, outputSelector) {
    this.input = document.querySelector(inputSelector);
    this.output = document.querySelector(outputSelector);
    this.history = [];
    this.historyIdx = -1;

    this.initListeners();
    this.printBanner();
  }

  printBanner() {
    this.appendLines([
      '<span class="term-cyan term-bold">KAIA ZERO-TRUST CYBER-SENTINEL OPERATIONS CENTER</span>',
      '<span class="term-dim">Deterministic Policy Gate active on 127.0.0.1. Type <b>help</b> for command list.</span>',
      '<span class="term-line-bright">─────────────────────────────────────────────────────────────────────────────</span>',
    ]);
  }

  initListeners() {
    if (!this.input) return;

    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const val = this.input.value.trim();
        if (val) {
          this.executeCommand(val);
          this.history.push(val);
          this.historyIdx = this.history.length;
          this.input.value = '';
        }
      } else if (e.key === 'ArrowUp') {
        if (this.history.length > 0 && this.historyIdx > 0) {
          this.historyIdx--;
          this.input.value = this.history[this.historyIdx];
        }
      } else if (e.key === 'ArrowDown') {
        if (this.historyIdx < this.history.length - 1) {
          this.historyIdx++;
          this.input.value = this.history[this.historyIdx];
        } else {
          this.historyIdx = this.history.length;
          this.input.value = '';
        }
      }
    });

    // Quick action buttons
    document.querySelectorAll('.cmd-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const cmd = btn.getAttribute('data-cmd');
        if (cmd === 'clear') {
          this.clear();
        } else if (cmd) {
          this.executeCommand(cmd);
        }
      });
    });
  }

  async executeCommand(cmdLine) {
    this.appendLines([`<span class="term-green">KAIA_SENTINEL:~$</span> <span class="term-text-bright">${esc(cmdLine)}</span>`]);

    if (cmdLine.toLowerCase() === 'clear') {
      this.clear();
      return;
    }

    try {
      const resp = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Kaia-Session': SESSION_KEY },
        body: JSON.stringify({ command: cmdLine }),
      });
      const data = await resp.json();

      if (data.lines && Array.isArray(data.lines)) {
        // Response lines can carry LAN hostnames, Shodan data, paths: escape before colouring.
        const colorized = data.lines.map((raw) => {
          const l = esc(raw);
          if (l.includes('[APPROVED]') || l.includes('✓')) {
            return `<span class="term-green">${l}</span>`;
          } else if (l.includes('[DENIED]') || l.includes('[ERROR]') || l.includes('✗')) {
            return `<span class="term-red">${l}</span>`;
          } else if (l.includes('[CRITICAL]') || l.includes('⚠')) {
            return `<span class="term-yellow term-bold">${l}</span>`;
          } else if (l.startsWith('===') || l.startsWith('---')) {
            return `<span class="term-dim">${l}</span>`;
          }
          return `<span class="term-text">${l}</span>`;
        });
        this.appendLines(colorized);
      }
    } catch (err) {
      this.appendLines([`<span class="term-red">Command transport error: ${esc(err.message)}</span>`]);
    }
  }

  appendLines(lines) {
    if (!this.output) return;
    for (const l of lines) {
      const div = document.createElement('div');
      div.innerHTML = l;
      this.output.appendChild(div);
    }
    this.output.scrollTop = this.output.scrollHeight;
  }

  clear() {
    if (!this.output) return;
    this.output.innerHTML = '';
    this.printBanner();
  }
}

window.SentinelTerminalConsole = SentinelTerminalConsole;
