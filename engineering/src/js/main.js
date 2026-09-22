/* 页面路由 + 各视图的 UI 绑定 */
(function (global) {
  'use strict';
  const DL = global.DL;
  const $ = (id) => document.getElementById(id);

  /* ============ 路由 ============ */
  function switchView(name) {
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === name));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + name));
    if (name === 'schedule') drawScheduleCharts();
    if (name === 'reverse') ensureReverseReady();
  }

  function initTabs() {
    document.getElementById('tabs').addEventListener('click', (e) => {
      const b = e.target.closest('.tab');
      if (b) switchView(b.dataset.view);
    });
    document.querySelectorAll('.card[data-goto]').forEach((c) => {
      c.addEventListener('click', () => switchView(c.dataset.goto));
    });
  }

  /* ============ 噪声调度视图 ============ */
  const SC_SIZE = 72;
  function testPattern(size) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, size, size);
    grd.addColorStop(0, '#12324a'); grd.addColorStop(1, '#0b1a24');
    g.fillStyle = grd; g.fillRect(0, 0, size, size);
    g.fillStyle = '#f6ad55';
    g.beginPath(); g.arc(size * 0.35, size * 0.4, size * 0.18, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#4fd1c5';
    g.fillRect(size * 0.55, size * 0.55, size * 0.3, size * 0.25);
    return c;
  }

  let baseImg = null, baseArr = null, baseNoise = null;
  function ensureBase() {
    if (baseImg) return;
    const c = testPattern(SC_SIZE);
    const d = c.getContext('2d').getImageData(0, 0, SC_SIZE, SC_SIZE).data;
    baseArr = new Float32Array(SC_SIZE * SC_SIZE * 3);
    for (let p = 0; p < SC_SIZE * SC_SIZE; p++) {
      for (let k = 0; k < 3; k++) baseArr[p * 3 + k] = d[p * 4 + k] / 127.5 - 1;
    }
    const rnd = DL.rand.mulberry32(2024);
    baseNoise = new Float32Array(baseArr.length);
    for (let i = 0; i < baseNoise.length; i++) baseNoise[i] = DL.rand.gaussian(rnd);
    baseImg = c;
  }

  function noisyCanvas(a) {
    const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
    const cv = document.createElement('canvas');
    cv.width = cv.height = SC_SIZE;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(SC_SIZE, SC_SIZE);
    for (let p = 0; p < SC_SIZE * SC_SIZE; p++) {
      for (let k = 0; k < 3; k++) {
        const v = sa * baseArr[p * 3 + k] + sn * baseNoise[p * 3 + k];
        img.data[p * 4 + k] = Math.max(0, Math.min(255, ((v + 1) / 2) * 255));
      }
      img.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  function selectedSchedules() {
    return ['linear', 'cosine', 'quadratic', 'sigmoid'].filter((n) => $(`sc-${n === 'quadratic' ? 'quad' : n === 'sigmoid' ? 'sigmoid' : n}`).checked);
  }

  function drawScheduleCharts() {
    const T = +$('sc-T').value;
    $('sc-T-out').textContent = T;
    const names = selectedSchedules();
    if (!names.length) names.push('linear');
    const schs = names.map((n) => ({ name: n, sch: DL.Schedule.build(n, T) }));

    const xs = Array.from({ length: T }, (_, i) => i + 1);
    DL.Charts.plot($('sc-beta'), schs.map((s) => ({
      label: DL.Schedule.LABELS[s.name], color: DL.Schedule.COLORS[s.name],
      xs, ys: Array.from(s.sch.betas),
    })), { xLabel: 't', yLabel: 'βt' });

    DL.Charts.plot($('sc-abar'), schs.map((s) => ({
      label: DL.Schedule.LABELS[s.name], color: DL.Schedule.COLORS[s.name],
      xs, ys: Array.from(s.sch.abar),
    })), { xLabel: 't', yLabel: 'ᾱt' });

    DL.Charts.plot($('sc-snr'), schs.map((s) => ({
      label: DL.Schedule.LABELS[s.name], color: DL.Schedule.COLORS[s.name],
      xs, ys: Array.from({ length: T }, (_, i) => DL.Schedule.logSNR(s.sch, i)),
    })), { xLabel: 't', yLabel: 'log-SNR' });

    // 实际加噪效果
    ensureBase();
    const grid = $('sc-grid');
    grid.innerHTML = '';
    const fracs = [0.15, 0.35, 0.55, 0.75, 0.95];
    fracs.forEach((f) => {
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = `t = ${Math.round(f * T)}`;
      fig.appendChild(cap);
      schs.forEach((s) => {
        const a = s.sch.abar[Math.min(T - 1, Math.max(0, Math.round(f * T) - 1))];
        const cv = noisyCanvas(a);
        cv.style.borderColor = DL.Schedule.COLORS[s.name];
        cv.title = DL.Schedule.LABELS[s.name];
        fig.appendChild(cv);
      });
      grid.appendChild(fig);
    });
    const lg = document.createElement('div');
    lg.className = 'legend';
    lg.innerHTML = schs.map((s) =>
      `<span><i style="background:${DL.Schedule.COLORS[s.name]}"></i>${DL.Schedule.LABELS[s.name]}</span>`).join('');
    grid.parentElement.insertBefore(lg, grid.nextSibling);
    if (grid._lg && grid._lg !== lg) grid._lg.remove();
    grid._lg = lg;
  }

  function initScheduleView() {
    ['sc-T', 'sc-linear', 'sc-cosine', 'sc-quad', 'sc-sigmoid'].forEach((id) => {
      $(id).addEventListener('input', drawScheduleCharts);
    });
  }

  /* ============ 反向采样视图 ============ */
  const RV = { canvas: null, ctx: null, bg: null, ready: false, running: false };
  const RV_X = 3.8, RV_Y = 3.1;

  function rvMap(cv, x, y) {
    return [((x + RV_X) / (2 * RV_X)) * cv.width, (1 - (y + RV_Y) / (2 * RV_Y)) * cv.height];
  }

  function ensureReverseReady() {
    if (RV.ready) return;
    RV.canvas = $('rv-canvas');
    RV.ctx = RV.canvas.getContext('2d');
    buildBackground();
    RV.ready = true;
  }

  function buildBackground() {
    const cv = RV.canvas;
    const img = RV.ctx.createImageData(cv.width, cv.height);
    let mn = Infinity, mx = -Infinity;
    const vals = new Float32Array(cv.width * cv.height);
    for (let py = 0; py < cv.height; py++) {
      const y = RV_Y - (2 * RV_Y * py) / cv.height;
      for (let px = 0; px < cv.width; px++) {
        const x = -RV_X + (2 * RV_X * px) / cv.width;
        const v = DL.Reverse.logp0(x, y);
        vals[py * cv.width + px] = v;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
    }
    for (let i = 0; i < vals.length; i++) {
      const g = Math.round(12 + 150 * Math.pow(Math.max(0, (vals[i] - mn) / (mx - mn)), 0.55));
      img.data[i * 4] = g; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = cv.width; off.height = cv.height;
    off.getContext('2d').putImageData(img, 0, 0);
    RV.bg = off;
  }

  function drawParticles(P, color, clear) {
    const cv = RV.canvas, ctx = RV.ctx;
    if (clear) { ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(RV.bg, 0, 0); }
    ctx.fillStyle = color || '#4fd1c5';
    for (let i = 0; i < P.length; i++) {
      const [px, py] = rvMap(cv, P[i][0], P[i][1]);
      ctx.fillRect(px - 1, py - 1, 2.2, 2.2);
    }
  }

  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

  async function runReverse() {
    if (RV.running) return;
    RV.running = true;
    ensureReverseReady();
    const S = +$('rv-steps').value;
    const N = +$('rv-particles').value;
    const spacing = $('rv-spacing').value;
    const isDDPM = $('rv-sampler').value === 'ddpm';
    const eta = isDDPM ? 1 : 0;
    const color = isDDPM ? '#f6ad55' : '#4fd1c5';

    if (!DL.Reverse.getSchedule()) DL.Reverse.setSchedule(DL.Schedule.build('cosine', 1000));
    const rnd = DL.rand.mulberry32(11);
    const P = new Array(N);
    for (let i = 0; i < N; i++) P[i] = [DL.rand.gaussian(rnd) * 1.6, DL.rand.gaussian(rnd) * 1.2];
    const ts = DL.Reverse.makeSchedule(S, spacing);
    drawParticles(P, color, true);

    const t0 = performance.now();
    const drawEvery = Math.max(1, Math.ceil(ts.length / 60));
    for (let i = 0; i < ts.length - 1; i++) {
      DL.Reverse.step(P, ts[i], ts[i + 1], eta, rnd);
      if (i % drawEvery === 0 || i === ts.length - 2) {
        drawParticles(P, color, true);
        $('rv-hint').textContent = `采样中… ${i + 1}/${ts.length - 1} 步（当前 t=${ts[i + 1]}）`;
        await nextFrame();
      }
    }
    const ms = performance.now() - t0;
    const m = DL.Reverse.evaluate(P);
    $('rv-hint').textContent = '采样完成。彩色点为最终样本，灰底为真实数据密度。';
    $('rv-metrics').innerHTML =
      `<span>采样器 = <b>${isDDPM ? 'DDPM (η=1)' : 'DDIM (η=0)'}</b></span>` +
      `<span>步数 S = <b>${S}</b>（共 ${ts.length - 1} 次网络/score 计算）</span>` +
      `<span>粒子数 = <b>${N}</b></span>` +
      `<span>平均 log p₀ = <b>${m.loglik.toFixed(3)}</b></span>` +
      `<span>模式命中率 = <b>${(m.hit * 100).toFixed(1)}%</b></span>` +
      `<span>耗时 = <b>${ms.toFixed(0)} ms</b></span>`;
    $('rv-legend').innerHTML =
      `<span><i style="background:${color}"></i>${isDDPM ? 'DDPM 样本' : 'DDIM 样本'}</span>` +
      `<span><i style="background:#ffffff"></i>真实数据模式中心</span>` +
      `<span>时间步选取：${spacing === 'snr' ? 'log-SNR 等分' : '均匀'}</span>`;
    RV.running = false;
  }

  function runAB() {
    const N = 400;
    const steps = [5, 10, 20, 50, 100, 200];
    const out = { ddim: [], ddpm: [] };
    if (!DL.Reverse.getSchedule()) DL.Reverse.setSchedule(DL.Schedule.build('cosine', 1000));
    steps.forEach((S) => {
      ['ddim', 'ddpm'].forEach((name) => {
        const r = DL.Reverse.sample({ n: N, steps: S, eta: name === 'ddpm' ? 1 : 0, seed: 5, spacing: 'uniform' });
        out[name].push({ S, loglik: r.metrics.loglik, hit: r.metrics.hit, ms: r.ms });
      });
    });
    const xs = steps;
    DL.Charts.plot($('rv-ab-chart'), [
      { label: 'DDIM', color: '#4fd1c5', xs, ys: out.ddim.map((d) => d.loglik), marker: true },
      { label: 'DDPM', color: '#f6ad55', xs, ys: out.ddpm.map((d) => d.loglik), marker: true },
    ], { xLabel: '采样步数 S', yLabel: '平均 log p₀' });
    DL.Charts.plot($('rv-ab-time'), [
      { label: 'DDIM', color: '#4fd1c5', xs, ys: out.ddim.map((d) => d.ms), marker: true },
      { label: 'DDPM', color: '#f6ad55', xs, ys: out.ddpm.map((d) => d.ms), marker: true },
    ], { xLabel: '采样步数 S', yLabel: '耗时 (ms)' });

    let html = '<table><tr><th>采样步数 S</th>';
    ['ddim', 'ddpm'].forEach(() => {});
    html += '<th>DDIM log p₀</th><th>DDIM 命中率</th><th>DDIM 耗时</th>' +
            '<th>DDPM log p₀</th><th>DDPM 命中率</th><th>DDPM 耗时</th></tr>';
    steps.forEach((S, i) => {
      const a = out.ddim[i], b = out.ddpm[i];
      html += `<tr><td>${S}</td><td>${a.loglik.toFixed(3)}</td><td>${(a.hit * 100).toFixed(1)}%</td>` +
              `<td>${a.ms.toFixed(0)} ms</td><td>${b.loglik.toFixed(3)}</td>` +
              `<td>${(b.hit * 100).toFixed(1)}%</td><td>${b.ms.toFixed(0)} ms</td></tr>`;
    });
    html += '</table><p class="muted" style="font-size:12px;margin-top:8px">' +
            '粒子数固定 400，cosine 调度，均匀时间步；结果由浏览器实时计算得出。</p>';
    $('rv-ab-table').innerHTML = html;
  }

  function initReverseView() {
    $('rv-steps').addEventListener('input', () => { $('rv-steps-out').textContent = $('rv-steps').value; });
    $('rv-particles').addEventListener('input', () => { $('rv-particles-out').textContent = $('rv-particles').value; });
    $('rv-run').addEventListener('click', runReverse);
    $('rv-ab').addEventListener('click', runAB);
  }

  /* ============ 启动 ============ */
  function boot() {
    initTabs();
    DL.Forward.bind();
    initScheduleView();
    initReverseView();
    DL.Reverse.setSchedule(DL.Schedule.build('cosine', 1000));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})(window);
