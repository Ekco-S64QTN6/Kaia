/**
 * Kaia Cyber-Sentinel: Master Application Controller
 * WebSocket telemetry stream -> operations panels + Kaia avatar posture.
 * All values shown are real snapshot fields; anything the host does not report shows "--".
 */

document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const setText = (id, t) => { const el = $(id); if (el) el.textContent = t; };
  const num = (v, d = 0) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '--');
  const setClass = (id, base, extra) => { const el = $(id); if (el) el.className = extra ? `${base} ${extra}` : base; };

  // ── Avatar & console ──
  const avatarController = new SentinelAvatarController(
    'panel-operative', 'conduit-glow', 'biometric-pulse', 'biometric-conduit', 'biometric-stance', 'operative-badge',
  );
  window.KaiaAvatar = avatarController;
  new SentinelTerminalConsole('#terminal-input', '#terminal-output');

  // ── Charts (single series each: the label names the series) ──
  const spark = (id, color) => new LineChart($(id), { color, autoMax: 1, axes: false });
  const sparkCpu = spark('spark-cpu', '#00f0ff');
  const sparkRam = spark('spark-ram', '#00f0ff');
  const sparkCtemp = spark('spark-ctemp', '#00f0ff');
  const sparkGpu = spark('spark-gpu', '#00f0ff');
  const lpsChart = new LineChart($('lps-chart'), {
    color: '#3dff8a', autoMax: 5, unit: 'logs/s', label: 'log events per second (60 s)',
    frameIntervalMs: 16, areaFill: true, threshold: 35, thresholdLabel: 'AUDIT SURGE 35 LPS',
    valueFormat: (v) => v.toFixed(1),
  });

  // ── Tables ──
  const tServices = new FitTable('services-body', 5);
  const tTop = new FitTable('top-body', 5);
  const tListen = new FitTable('listen-body', 4);
  const tokenLedger = new CapabilityTokenLedger($('token-ledger'));
  const waterfall = new SyscallWaterfallCanvas($('waterfall-canvas'), $('waterfall-meta'), $('waterfall-inspector'));
  const networkRadar = new NetworkRadarCanvas($('network-radar'), $('network-meta'), $('radar-risk'));
  const tProcs = new FitTable('procs-body', 5);
  const tAudit = new FitTable('audit-body', 4);
  const tBlocks = new FitTable('blocks-body', 3);
  const tAssets = new FitTable('assets-body', 5);

  let logFilter = 'all';
  let lastLogs = [];
  document.querySelectorAll('.filter-pill').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      logFilter = btn.dataset.filter;
      renderAudit(lastLogs);
    });
  });

  // Local clock
  setInterval(() => setText('clock', new Date().toTimeString().slice(0, 8)), 1000);

  // ── Header ──
  function renderHeader(snap) {
    setText('host-name', snap.host || '--');
    const running = snap.system_security && snap.system_security.gate_running;
    setText('policy-gate-badge', running ? 'ENFORCING' : 'OFFLINE');
    $('policy-gate-badge').classList.toggle('state-good', Boolean(running));
    $('policy-gate-badge').classList.toggle('state-bad', !running);
    const tl = snap.threat_level || '--';
    setText('threat-badge-val', tl);
    $('threat-pill').className = tl === 'CRITICAL' ? 'pill alert' : tl === 'ELEVATED' ? 'pill warn' : 'pill';
    setText('lps-badge-val', `${num(snap.lps, 1)}/s`);
  }

  // ── Host & services ──
  function renderHost(snap) {
    const hw = snap.host_telemetry || {};
    setText('kpi-cpu', `${num(hw.cpu_pct)}%`);
    setText('kpi-cpu-sub', hw.cpu_freq_mhz ? `${(hw.cpu_freq_mhz / 1000).toFixed(2)} GHz` : 'frequency --');
    setText('kpi-ram', `${num(hw.ram_pct)}%`);
    setText('kpi-ram-sub', `${num(hw.ram_used_gb, 1)} / ${num(hw.ram_total_gb, 1)} GB`);

    const ct = hw.cpu_temp;
    setText('kpi-ctemp', ct ? `${num(ct)}°C` : '--');
    setClass('kpi-ctemp', 'kpi-value', ct >= 95 ? 'bad' : ct >= 85 ? 'warn' : '');
    setText('kpi-ctemp-sub', `${hw.cpu_power ? `${num(hw.cpu_power, 1)} W` : '-- W'} · ${hw.cpu_throttled ? 'THROTTLED' : 'not throttled'}`);

    const gt = hw.gpu_temp;
    setText('kpi-gpu', gt ? `${num(gt)}°C` : '--');
    setClass('kpi-gpu', 'kpi-value', gt >= 90 ? 'bad' : gt >= 80 ? 'warn' : '');
    setText('kpi-gpu-sub', gt
      ? `${num(hw.gpu_power, 0)} W · fan ${num(hw.gpu_fan)}% · ${num((hw.gpu_mem_used_mb || 0) / 1024, 1)} GB VRAM`
      : 'no GPU telemetry');

    sparkCpu.push(hw.cpu_pct);
    sparkRam.push(hw.ram_pct);
    if (ct) sparkCtemp.push(ct);
    if (gt) sparkGpu.push(gt);

    const services = snap.services || [];
    tServices.render(services.map((s) => {
      const state = s.active ? `<span class="tag ok">${esc((s.substate || 'active').toUpperCase())}</span>`
        : `<span class="tag bad">${esc((s.substate || 'inactive').toUpperCase())}</span>`;
      return `<tr><td>${esc(s.name)}</td><td>${state}</td><td class="num mid">${s.pid ? esc(s.pid) : '--'}</td>`
        + `<td class="num">${typeof s.cpu === 'number' ? `${num(s.cpu, 1)}%` : '--'}</td>`
        + `<td class="num">${typeof s.memory_mb === 'number' ? `${num(s.memory_mb)} MB` : '--'}</td></tr>`;
    }), 'Service collector unavailable');
    tTop.render((snap.top_processes || []).map((p) => (
      `<tr><td>${esc(p.name)}</td><td class="mid">${esc(p.user)}</td><td class="num mid">${esc(p.pid)}</td>`
      + `<td class="num">${num(p.cpu, 1)}%</td><td class="num">${typeof p.memory_mb === 'number' ? `${num(p.memory_mb)} MB` : '--'}</td></tr>`
    )), 'Process list unavailable');

    const down = services.filter((s) => !s.active).length;
    setText('host-meta', down ? `${down} SERVICE${down > 1 ? 'S' : ''} DOWN` : `${services.length} SERVICES UP`);
    $('host-meta').classList.toggle('state-bad', down > 0);
    setText('host-foot', `split-lock events ${hw.split_lock_events ?? '--'}`
      + (hw.split_lock_last_app && hw.split_lock_last_app !== 'None' ? ` (last: ${hw.split_lock_last_app})` : '')
      + ' · trends: last 60 s · process CPU as % of all cores');
  }

  // ── Policy gate & containment ──
  function renderPolicy(snap) {
    const sec = snap.system_security || {};
    const con = snap.containment || {};
    const ti = snap.threat_intel || {};
    const recent = snap.recent_activity || {};

    const gate = sec.gate_running
      ? `<span class="item"><span class="dot green"></span>Gate <b class="ok">ENFORCING</b> pid ${esc(sec.gate_pid)} · up ${esc(sec.gate_uptime || '--')}</span>`
      : '<span class="item"><span class="dot red"></span>Gate <b class="bad">OFFLINE</b></span>';
    const lock = sec.lockdown_active
      ? '<span class="item">Lockdown <b class="bad">ACTIVE</b></span>'
      : '<span class="item">Lockdown <b>off</b></span>';
    const posture = `<span class="item">Posture <b>${esc(snap.sentinel_state || '--')}</b></span>`;
    $('gate-strip').innerHTML = gate + lock + posture;
    setText('gate-meta', snap.timestamp_unix ? `UPDATED ${clockTime(snap.timestamp_unix)}` : '');

    setText('st-tok-active', sec.tokens_active ?? '--');
    setText('st-tok-expired', `${sec.tokens_expired ?? '--'} expired`);
    setText('st-rejected', sec.tokens_rejected ?? '--');
    const rj = recent.rejections_10m || 0;
    setText('st-rejected-sub', rj ? `+${rj} in last 10 min` : 'none in last 10 min');
    setClass('st-rejected-sub', 'stat-sub', rj ? 'bad' : '');
    setClass('st-rejected', 'stat-value', rj ? 'bad' : '');

    setText('st-alerts', con.sentinel_alerts ?? '--');
    const al = recent.alerts_10m || 0;
    setText('st-alerts-sub', al ? `+${al} in last 10 min` : 'none in last 10 min');
    setClass('st-alerts-sub', 'stat-sub', al ? 'bad' : '');
    setClass('st-alerts', 'stat-value', al ? 'bad' : '');

    const fim = (con.fim_alerts || []).length, priv = (con.privilege_alerts || []).length;
    setText('st-fim', `${fim} / ${priv}`);
    setClass('st-fim', 'stat-value', fim + priv ? 'bad' : '');
    setText('st-fim-sub', 'FIM / privilege');

    const rc = sec.rules_counts || ti.rules_counts || {};
    setText('st-blocks', ti.active_blocks ?? '--');
    setText('st-rules', `${rc.input ?? '-'} in · ${rc.forward ?? '-'} fwd · ${rc.output ?? '-'} out`);
    setText('st-yara', sec.yara_rules_count ?? '--');
    setClass('st-yara', 'stat-value', sec.yara_rules_count === 0 ? 'warn' : '');

    const hp = ti.honeypot_status || {};
    setText('st-honeypot', hp.total_triggers ?? '--');
    setClass('st-honeypot', 'stat-value', hp.total_triggers ? 'bad' : '');
    setText('st-honeypot-sub', hp.last_trigger_ip && hp.last_trigger_ip !== 'None' ? `last ${hp.last_trigger_ip}` : 'no trips');

    setText('st-sandbox', con.sandbox_tier || '--');
    setText('st-cgroup', `${con.cgroup_cpu ?? '-'} cpu · ${con.cgroup_mem ?? '-'}`);
    $('st-cgroup').title = `cgroup limits: cpu ${con.cgroup_cpu} · mem ${con.cgroup_mem} · pids ${con.cgroup_pids}`;
    setText('lattice-display', con.lattice_str || '--');
    const lastAlert = con.last_alert_path && con.last_alert_path !== 'None' ? con.last_alert_path : 'none recorded';
    setText('last-alert-path', lastAlert);

    tokenLedger.render(sec.token_cards || []);
  }

  // ── Process executions ──
  const seenPids = new Set();
  function renderProcs(snap) {
    const events = snap.ebpf_waterfall || [];
    waterfall.update(events, snap.process_source || '');
    const byPid = new Map();
    for (const ev of events) if (!byPid.has(ev.pid)) byPid.set(ev.pid, ev);   // newest first already
    const rows = [...byPid.values()];
    const verdictTag = (d) => {
      const v = String(d || 'observed').toLowerCase();
      const cls = v === 'blocked' ? 'bad' : v === 'sandboxed' ? 'warn' : v === 'safe' ? 'ok' : 'neutral';
      return `<span class="tag ${cls}">${esc(v.toUpperCase())}</span>`;
    };
    const html = rows.map((ev) => {
      const fresh = !seenPids.has(ev.pid);
      return `<tr class="${fresh ? 'new' : ''}"><td class="dim">${clockTime(ev.timestamp)}</td><td class="num mid">${esc(ev.pid)}</td>`
        + `<td>${esc(ev.comm)}</td><td class="mid" title="${esc(ev.filename)}">${esc(ev.filename)}</td><td>${verdictTag(ev.disposition)}</td></tr>`;
    });
    const { shown, total } = tProcs.render(html, 'No process executions observed yet');
    rows.forEach((ev) => seenPids.add(ev.pid));
    const lastMin = rows.filter((ev) => Date.now() / 1000 - ev.timestamp <= 60).length;
    setText('procs-meta', (snap.process_source || '').toUpperCase());
    setText('procs-foot', `showing ${shown} of ${total} · ${lastMin} in the last 60 s · execve`);
  }

  // ── Audit log ──
  function renderAudit(logs) {
    lastLogs = logs;
    const parsed = logs.map(parseAuditLine).reverse();
    const isDenied = (p) => /denied|error|violation|blocked/.test(p.result);
    const isApproved = (p) => /approved|ok|success/.test(p.result);
    const filtered = parsed.filter((p) => logFilter === 'all'
      || (logFilter === 'denied' && isDenied(p))
      || (logFilter === 'approved' && isApproved(p))
      || (logFilter === 'system' && !isDenied(p) && !isApproved(p)));
    const { shown, total } = tAudit.render(filtered.map((p) => (
      `<tr><td class="dim">${esc(p.time)}</td><td>${resultTag(p.result)}</td>`
      + `<td title="${esc(p.hash ? `hash ${p.hash}` : '')}">${esc(p.event)}</td><td class="mid">${esc(p.source)}</td></tr>`
    )), 'No matching audit events');
    const denied = parsed.filter(isDenied).length, approved = parsed.filter(isApproved).length;
    setText('audit-foot', `showing ${shown} of ${total} · buffer: ${denied} denied, ${approved} approved · full history: "show audit --since 2h"`);
  }

  // ── Network ──
  function renderNetwork(snap) {
    const ti = snap.threat_intel || {};
    networkRadar.update(ti);
    const rc = ti.rules_counts || {};
    const hp = ti.honeypot_status || {};
    $('net-strip').innerHTML =
      `<span class="item">Input drop/reject rules <b>${esc(ti.active_blocks ?? '--')}</b></span>`
      + `<span class="item">nft rules <b>${esc(rc.input ?? '-')}</b> in · <b>${esc(rc.forward ?? '-')}</b> fwd · <b>${esc(rc.output ?? '-')}</b> out</span>`
      + `<span class="item">Honeypot <b class="${hp.total_triggers ? 'bad' : ''}">${esc(hp.total_triggers ?? 0)}</b> trips</span>`
      + (ti.gate_offline ? '<span class="item"><b class="bad">GATE OFFLINE: firewall state unverified</b></span>' : '');

    const blocks = ti.recent_blocks || [];
    tBlocks.render(blocks.map((b) => {
      const ip = b.ip || b.target || b.address || '?';
      const reason = b.reason || b.rule || b.source || '';
      const when = b.timestamp || b.time || b.blocked_at || '';
      return `<tr><td>${esc(ip)}</td><td class="mid">${esc(reason)}</td><td class="dim">${esc(when ? ago(when) : '--')}</td></tr>`;
    }), 'No recent blocks');

    const listen = snap.listening || [];
    tListen.render(listen.map((l) => (
      `<tr><td>${esc(l.ip.includes(':') ? `[${l.ip}]` : l.ip)}:${esc(l.port)}</td><td class="mid">${esc(l.proto)}</td>`
      + `<td class="mid">${l.process ? `${esc(l.process)} (${esc(l.pid)})` : 'other user'}</td>`
      + `<td>${l.exposed ? '<span class="tag warn">EXPOSED</span>' : '<span class="tag neutral">LOCAL</span>'}</td></tr>`
    )), 'No listening sockets visible');

    const assets = [...(ti.recent_assets || [])].sort((a, b) => Date.parse(b.last_seen) - Date.parse(a.last_seen));
    tAssets.render(assets.map((a) => {
      const who = [a.vendor, a.hostname].filter((x) => x && x.toLowerCase() !== 'unknown').join(' / ') || 'unknown';
      return `<tr><td>${esc(a.ip)}</td><td class="mid">${esc(a.mac)}</td><td class="mid">${esc(who)}</td>`
        + `<td class="dim">${esc(a.detection_vector)}</td><td class="dim">${esc(ago(a.last_seen))}</td></tr>`;
    }), 'No LAN assets discovered');
  }

  function handleSnapshot(snap) {
    if (!snap || typeof snap !== 'object') return;
    renderHeader(snap);
    renderHost(snap);
    renderPolicy(snap);
    renderProcs(snap);
    if (Array.isArray(snap.logs)) renderAudit(snap.logs);
    renderNetwork(snap);
    if (typeof snap.lps === 'number') lpsChart.push(snap.lps);
    avatarController.updatePosture(snap);
  }

  // ── WebSocket (read-only telemetry) ──
  let reconnectTimer = null;
  function connectTelemetry() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onopen = () => {
      $('conn-dot').className = 'dot green';
      setText('conn-pill-text', 'ONLINE');
      const r = avatarController.characterRenderer;
      if (r && r.setLink) r.setLink(true);
    };
    ws.onmessage = (event) => {
      try {
        handleSnapshot(JSON.parse(event.data));
      } catch (err) {
        console.error('[KaiaHUD] Error handling telemetry snapshot:', err);
      }
    };
    ws.onclose = () => {
      $('conn-dot').className = 'dot red';
      setText('conn-pill-text', 'OFFLINE (RECONNECTING)');
      const r = avatarController.characterRenderer;
      if (r && r.setLink) r.setLink(false);
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectTelemetry, 1500);
    };
    ws.onerror = () => ws.close();
  }
  connectTelemetry();
});
