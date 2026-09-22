/* 极简 Canvas 折线图 —— 无第三方依赖，够用即可 */
(function (global) {
  'use strict';
  const DL = (global.DL = global.DL || {});

  function plot(canvas, series, opts) {
    opts = opts || {};
    const ctx = canvas.getContext('2d');
    const dpr = global.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const pad = { l: 52, r: 14, t: 14, b: 32 };
    const W = cssW - pad.l - pad.r;
    const H = cssH - pad.t - pad.b;

    // 取值范围
    let xmin = opts.xmin != null ? opts.xmin : Infinity;
    let xmax = opts.xmax != null ? opts.xmax : -Infinity;
    let ymin = opts.ymin != null ? opts.ymin : Infinity;
    let ymax = opts.ymax != null ? opts.ymax : -Infinity;
    series.forEach((s) => {
      const xs = s.xs || s.ys.map((_, i) => i);
      const n = s.ys.length;
      for (let i = 0; i < n; i++) {
        const x = xs[i], y = s.ys[i];
        if (Number.isFinite(x)) { if (xmin === Infinity || x < xmin) xmin = x; if (x > xmax) xmax = x; }
        if (Number.isFinite(y)) { if (ymin === Infinity || y < ymin) ymin = y; if (y > ymax) ymax = y; }
      }
    });
    if (!Number.isFinite(xmin)) { xmin = 0; xmax = 1; }
    if (!Number.isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (xmax === xmin) xmax = xmin + 1;
    if (ymax === ymin) { ymax += 0.5; ymin -= 0.5; }
    const padY = (ymax - ymin) * 0.08;
    ymin -= padY; ymax += padY;

    const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * W;
    const Y = (v) => pad.t + (1 - (v - ymin) / (ymax - ymin)) * H;

    // 网格 + 刻度
    ctx.strokeStyle = '#242c38';
    ctx.fillStyle = '#9aa7b4';
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui, sans-serif';
    for (let k = 0; k <= 4; k++) {
      const yv = ymin + ((ymax - ymin) * k) / 4;
      const y = Y(yv);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + W, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(fmt(yv), pad.l - 6, y);
      const xv = xmin + ((xmax - xmin) * k) / 4;
      const x = X(xv);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + H); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(fmt(xv), x, pad.t + H + 5);
    }
    ctx.strokeStyle = '#3a4553';
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + H); ctx.lineTo(pad.l + W, pad.t + H); ctx.stroke();

    // 轴标签
    ctx.fillStyle = '#9aa7b4';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    if (opts.xLabel) ctx.fillText(opts.xLabel, pad.l + W / 2, cssH - 2);
    if (opts.yLabel) {
      ctx.save();
      ctx.translate(12, pad.t + H / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'top'; ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }

    // 曲线
    series.forEach((s) => {
      const xs = s.xs || s.ys.map((_, i) => i);
      ctx.strokeStyle = s.color || '#4fd1c5';
      ctx.lineWidth = s.width || 2;
      ctx.lineJoin = 'round';
      if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.ys.length; i++) {
        const x = X(xs[i]), y = Y(s.ys[i]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.marker) {
        ctx.fillStyle = s.color || '#4fd1c5';
        for (let i = 0; i < s.ys.length; i++) {
          ctx.beginPath(); ctx.arc(X(xs[i]), Y(s.ys[i]), 2.6, 0, Math.PI * 2); ctx.fill();
        }
      }
    });

    // 图例
    if (series.length && opts.legend !== false) {
      let lx = pad.l + 6, ly = pad.t + 6;
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      series.forEach((s) => {
        const label = s.label || '';
        const w = ctx.measureText(label).width;
        if (lx + w + 26 > pad.l + W) { lx = pad.l + 6; ly += 15; }
        ctx.fillStyle = s.color || '#4fd1c5';
        ctx.fillRect(lx, ly - 1, 10, 2.5);
        ctx.fillStyle = '#e6edf3';
        ctx.fillText(label, lx + 14, ly);
        lx += w + 30;
      });
    }
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
    if (a >= 10) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    return v.toFixed(3);
  }

  DL.Charts = { plot };
})(window);
