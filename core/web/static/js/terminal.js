/**
 * Kaia Cyber-Sentinel: Interactive Operator Command Console
 * Dispatches deterministic commands to /api/command with live ANSI/cyber formatting.
 */

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
      '<span style="color:var(--cyan);font-weight:bold">KAIA ZERO-TRUST CYBER-SENTINEL OPERATIONS CENTER</span>',
      '<span style="color:var(--dim)">Deterministic Policy Gate active on 127.0.0.1. Type <b>help</b> for command list.</span>',
      '<span style="color:var(--line-bright)">─────────────────────────────────────────────────────────────────────────────</span>',
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
    this.appendLines([`<span style="color:var(--green)">KAIA_SENTINEL:~$</span> <span style="color:var(--text-bright)">${esc(cmdLine)}</span>`]);

    if (cmdLine.toLowerCase() === 'clear') {
      this.clear();
      return;
    }

    try {
      const resp = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmdLine }),
      });
      const data = await resp.json();

      if (data.lines && Array.isArray(data.lines)) {
        // Response lines can carry LAN hostnames, Shodan data, paths: escape before colouring.
        const colorized = data.lines.map((raw) => {
          const l = esc(raw);
          if (l.includes('[APPROVED]') || l.includes('✓')) {
            return `<span style="color:var(--green)">${l}</span>`;
          } else if (l.includes('[DENIED]') || l.includes('[ERROR]') || l.includes('✗')) {
            return `<span style="color:var(--red)">${l}</span>`;
          } else if (l.includes('[CRITICAL]') || l.includes('⚠')) {
            return `<span style="color:var(--yellow);font-weight:bold">${l}</span>`;
          } else if (l.startsWith('===') || l.startsWith('---')) {
            return `<span style="color:var(--dim)">${l}</span>`;
          }
          return `<span style="color:var(--text)">${l}</span>`;
        });
        this.appendLines(colorized);
      }
    } catch (err) {
      this.appendLines([`<span style="color:var(--red)">Command transport error: ${esc(err.message)}</span>`]);
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
