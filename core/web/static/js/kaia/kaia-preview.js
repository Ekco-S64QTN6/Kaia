    (async () => {
      const qs = new URLSearchParams(location.search);
      const still = qs.has('still');
      const codec = await KaiaCodec.mount(document.getElementById('stage'), { still, portrait: qs.get('face') || undefined });
      if (qs.get('open')) codec.animator.forceOpen = parseFloat(qs.get('open'));
      if (qs.get('lid')) codec.animator.forceLid = parseFloat(qs.get('lid'));
      if (qs.get('pose')) codec.animator.forcePose = qs.get('pose').split(',').map(Number);
      window.codec = codec;
      if (qs.has('stats')) {
        const el = document.createElement('div');
        el.id = 'stats';
        el.className = 'stats-overlay';
        document.body.appendChild(el);
        codec.profile((st) => {
          const gpu = st.gpu === null ? (st.gpuAvailable ? '…' : 'n/a') : st.gpu.toFixed(2);
          el.textContent = `${st.fps.toFixed(0)} fps · cpu ${st.cpu.toFixed(2)} ms · gpu ${gpu} ms`;
          const dbg = codec.renderer.gl.getExtension('WEBGL_debug_renderer_info');
          const gpuName = dbg ? codec.renderer.gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
          const c = codec.renderer.canvas;
          document.title = `${el.textContent} · ${c.width}x${c.height} · exposure ${codec.renderer.exposure.toFixed(2)} · ${gpuName}`;
        });
      }

      const box = document.getElementById('moods');
      const buttons = {};
      const pick = (m) => {
        codec.setMood(m);
        Object.values(buttons).forEach((b) => b.classList.toggle('on', b.dataset.mood === m));
      };
      // Kaia's 7 security states first, then the shared base moods.
      const ordered = [...(window.KAIA_SENTINEL_STATES || []).map((m) => m.toLowerCase()), ...KaiaAnimator.moods];
      [...new Set(ordered)].forEach((m) => {
        const b = document.createElement('button');
        b.textContent = m;
        b.dataset.mood = m;
        b.onclick = () => pick(m);
        buttons[m] = b;
        box.appendChild(b);
      });
      pick(qs.get('mood') || 'neutral');
      // dev: ?react=capture / ?then=<mood> fire 1.5 s after load (for captures of transitions)
      if (qs.get('react')) setTimeout(() => codec.react(qs.get('react')), 1500);
      if (qs.get('then')) setTimeout(() => pick(qs.get('then')), 1500);

      document.querySelectorAll('[data-react]').forEach((b) => (b.onclick = () => codec.react(b.dataset.react)));
      const say = () => codec.speak(document.getElementById('line').value);
      document.getElementById('say').onclick = say;
      document.getElementById('line').onkeydown = (e) => e.key === 'Enter' && say();

      let timer = null;
      document.getElementById('cycle').onclick = (e) => {
        if (timer) { clearInterval(timer); timer = null; e.target.classList.remove('on'); return; }
        e.target.classList.add('on');
        let i = 0;
        timer = setInterval(() => pick(KaiaAnimator.moods[++i % KaiaAnimator.moods.length]), 3500);
      };
      let linkOk = true;
      document.getElementById('link').onclick = (e) => {
        linkOk = !linkOk;
        codec.setLink(linkOk);
        e.target.classList.toggle('on', !linkOk);
      };
      if (!still) setTimeout(() => codec.speak('Codec link established. Monitoring signals.'), 900);
    })();
