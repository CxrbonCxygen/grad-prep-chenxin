/* 前向加噪：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
 * 完全闭式，不需要任何模型 —— 这正是 DDPM 训练时能"一步跳到任意 t"的原因。
 */
(function (global) {
  'use strict';
  const DL = (global.DL = global.DL || {});
  const SIZE = 128;             // 内部计算分辨率
  const T_DEFAULT = 1000;

  let sch = null, T = T_DEFAULT;
  let x0 = null, eps = null;    // Float32Array(SIZE*SIZE*3)，取值 [-1,1]
  let curT = 0, playing = false, rafId = 0, seed = 7;

  const $ = (id) => document.getElementById(id);
  let el = {};

  /* ---------- 内置示例图（程序化生成，无外部资源） ---------- */
  function builtinImage() {
    const c = document.createElement('canvas');
    c.width = c.height = SIZE;
    const g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, SIZE, SIZE);
    grd.addColorStop(0, '#12324a');
    grd.addColorStop(0.5, '#1f6f8b');
    grd.addColorStop(1, '#0b1a24');
    g.fillStyle = grd; g.fillRect(0, 0, SIZE, SIZE);

    g.fillStyle = '#f6ad55';
    g.beginPath(); g.arc(SIZE * 0.32, SIZE * 0.36, SIZE * 0.16, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#4fd1c5';
    g.beginPath();
    g.moveTo(SIZE * 0.62, SIZE * 0.2); g.lineTo(SIZE * 0.86, SIZE * 0.62); g.lineTo(SIZE * 0.38, SIZE * 0.62);
    g.closePath(); g.fill();
    g.strokeStyle = '#a78bfa'; g.lineWidth = 6;
    for (let i = 0; i < 6; i++) {
      g.beginPath();
      g.moveTo(0, SIZE * (0.72 + i * 0.045));
      g.lineTo(SIZE, SIZE * (0.68 + i * 0.045));
      g.stroke();
    }
    g.fillStyle = '#e6edf3';
    g.font = 'bold 22px system-ui';
    g.fillText('DL', SIZE * 0.44, SIZE * 0.94);
    return c;
  }

  function canvasToArray(canvas) {
    const g = canvas.getContext('2d');
    const d = g.getImageData(0, 0, SIZE, SIZE).data;
    const a = new Float32Array(SIZE * SIZE * 3);
    for (let p = 0; p < SIZE * SIZE; p++) {
      for (let k = 0; k < 3; k++) a[p * 3 + k] = d[p * 4 + k] / 127.5 - 1;
    }
    return a;
  }

  function regenerateNoise() {
    const rnd = DL.rand.mulberry32(seed);
    eps = new Float32Array(SIZE * SIZE * 3);
    for (let i = 0; i < eps.length; i++) eps[i] = DL.rand.gaussian(rnd);
  }

  /* ---------- 渲染 ---------- */
  function abarAt(t) {
    if (t <= 0) return 1;
    return sch.abar[Math.min(t, T) - 1];
  }

  function drawTo(ctx, arr, canvas) {
    const img = ctx.createImageData(canvas.width, canvas.height);
    const scale = SIZE / canvas.width;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const sx = Math.min(SIZE - 1, Math.floor(x * scale));
        const sy = Math.min(SIZE - 1, Math.floor(y * scale));
        const si = (sy * SIZE + sx) * 3;
        const di = (y * canvas.width + x) * 4;
        for (let k = 0; k < 3; k++) {
          img.data[di + k] = Math.max(0, Math.min(255, ((arr[si + k] + 1) / 2) * 255));
        }
        img.data[di + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function render() {
    const a = abarAt(curT);
    const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
    const xt = new Float32Array(x0.length);
    for (let i = 0; i < x0.length; i++) xt[i] = sa * x0[i] + sn * eps[i];
    drawTo(el.canvas.getContext('2d'), xt, el.canvas);

    el.tOut.textContent = curT;
    const snr = a / Math.max(1 - a, 1e-12);
    el.stats.innerHTML =
      `<span>t = <b>${curT}</b> / ${T}</span>` +
      `<span>ᾱ<sub>t</sub> = <b>${a.toFixed(4)}</b></span>` +
      `<span>√ᾱ<sub>t</sub> = <b>${sa.toFixed(3)}</b></span>` +
      `<span>√(1−ᾱ<sub>t</sub>) = <b>${sn.toFixed(3)}</b></span>` +
      `<span>信噪比 ᾱ/(1−ᾱ) = <b>${snr < 1e-4 ? snr.toExponential(2) : snr.toFixed(4)}</b></span>` +
      `<span>调度 = <b>${el.sched.value}</b></span>`;
  }

  function renderStrip() {
    el.strip.innerHTML = '';
    const fracs = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    fracs.forEach((f) => {
      const t = Math.round(f * T);
      const a = abarAt(t);
      const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
      const xt = new Float32Array(x0.length);
      for (let i = 0; i < x0.length; i++) xt[i] = sa * x0[i] + sn * eps[i];
      const cv = document.createElement('canvas');
      cv.width = cv.height = SIZE;
      drawTo(cv.getContext('2d'), xt, cv);
      const fig = document.createElement('figure');
      fig.appendChild(cv);
      const cap = document.createElement('figcaption');
      cap.textContent = `t=${t}`;
      fig.appendChild(cap);
      el.strip.appendChild(fig);
    });
  }

  function drawOriginal() {
    drawTo(el.orig.getContext('2d'), x0, el.orig);
  }

  /* ---------- 交互 ---------- */
  function rebuild() {
    T = T_DEFAULT;
    sch = DL.Schedule.build(el.sched.value, T);
    el.t.max = T;
    curT = Math.min(curT, T);
    renderStrip();
    render();
  }

  function play() {
    if (playing) { stop(); return; }
    playing = true;
    el.play.textContent = '⏸ 暂停';
    const tick = () => {
      curT = Math.min(T, curT + 8);
      render();
      if (curT >= T) { stop(); return; }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function stop() {
    playing = false;
    el.play.textContent = '▶ 播放';
    cancelAnimationFrame(rafId);
  }

  function bind() {
    el = {
      canvas: $('fw-canvas'), orig: $('fw-orig'), t: $('fw-t'), tOut: $('fw-t-out'),
      sched: $('fw-sched'), play: $('fw-play'), noise: $('fw-noise'), reset: $('fw-reset'),
      file: $('fw-file'), stats: $('fw-stats'), strip: $('fw-strip'),
    };
    if (!el.canvas) return;

    el.t.addEventListener('input', () => { stop(); curT = +el.t.value; render(); });
    el.sched.addEventListener('change', () => { rebuild(); });
    el.play.addEventListener('click', play);
    el.noise.addEventListener('click', () => { seed = (seed * 31 + 17) % 100000; regenerateNoise(); renderStrip(); render(); });
    el.reset.addEventListener('click', () => { stop(); curT = 0; el.t.value = 0; render(); });
    el.file.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = c.height = SIZE;
        const g = c.getContext('2d');
        g.fillStyle = '#000'; g.fillRect(0, 0, SIZE, SIZE);
        const s = Math.max(SIZE / img.width, SIZE / img.height);
        const w = img.width * s, h = img.height * s;
        g.drawImage(img, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
        x0 = canvasToArray(c);
        drawOriginal(); renderStrip(); render();
      };
      img.src = URL.createObjectURL(f);
    });

    x0 = canvasToArray(builtinImage());
    regenerateNoise();
    drawOriginal();
    rebuild();
  }

  DL.Forward = { bind };
})(window);
