/* 噪声调度 + 通用随机工具 —— Diffusion Lab
 * 对应论文：
 *   linear   : Ho et al., DDPM, NeurIPS 2020 (β: 1e-4 -> 0.02)
 *   cosine   : Nichol & Dhariwal, Improved DDPM, ICML 2021
 *   quadratic/sigmoid : 社区常见变体，用于对比
 */
(function (global) {
  'use strict';
  const DL = (global.DL = global.DL || {});

  /* ---------------- 可复现随机数 ---------------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function linspace(a, b, n) {
    const out = new Float64Array(n);
    const step = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + step * i;
    return out;
  }

  /* ---------------- 调度 ---------------- */
  const SIGMOID = (x) => 1 / (1 + Math.exp(-x));

  function betasLinear(T, b0 = 1e-4, b1 = 0.02) {
    const b = new Float64Array(T);
    for (let i = 0; i < T; i++) b[i] = b0 + (b1 - b0) * (i / (T - 1));
    return b;
  }

  function betasCosine(T, s = 8e-3) {
    // 直接对 alpha_bar 做余弦衰减，再由 alpha_bar 反推 beta
    const f = (t) => Math.cos(((t / T + s) / (1 + s)) * Math.PI * 0.5) ** 2;
    const abar = new Float64Array(T);
    for (let i = 0; i < T; i++) abar[i] = f(i + 1) / f(0);
    const b = new Float64Array(T);
    let prev = 1.0;
    for (let i = 0; i < T; i++) {
      b[i] = Math.min(0.999, 1 - abar[i] / prev);
      prev = abar[i];
    }
    return b;
  }

  function betasQuadratic(T, b0 = 1e-4, b1 = 0.02) {
    // 对 sqrt(beta) 线性插值，等价于 beta 随 t 二次增长
    const b = new Float64Array(T);
    const s0 = Math.sqrt(b0), s1 = Math.sqrt(b1);
    for (let i = 0; i < T; i++) {
      const s = s0 + (s1 - s0) * (i / (T - 1));
      b[i] = s * s;
    }
    return b;
  }

  function betasSigmoid(T, b0 = 1e-4, b1 = 0.02) {
    const b = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      const x = -6 + 12 * (i / (T - 1));
      b[i] = SIGMOID(x) * (b1 - b0) + b0;
    }
    return b;
  }

  const BUILDERS = {
    linear: betasLinear,
    cosine: betasCosine,
    quadratic: betasQuadratic,
    sigmoid: betasSigmoid,
  };

  const COLORS = {
    linear: '#4fd1c5',
    cosine: '#f6ad55',
    quadratic: '#a78bfa',
    sigmoid: '#f87171',
  };

  const LABELS = {
    linear: 'linear (DDPM)',
    cosine: 'cosine (Improved DDPM)',
    quadratic: 'quadratic',
    sigmoid: 'sigmoid',
  };

  /** 由 betas 派生全部系数 */
  function derive(betas) {
    const T = betas.length;
    const alphas = new Float64Array(T);
    const abar = new Float64Array(T);
    const abarPrev = new Float64Array(T);
    let cum = 1.0;
    for (let i = 0; i < T; i++) {
      alphas[i] = 1 - betas[i];
      abarPrev[i] = cum;
      cum *= alphas[i];
      abar[i] = cum;
    }
    return { T, betas, alphas, abar, abarPrev };
  }

  /** 统一入口：build('cosine', 1000) -> {T, betas, alphas, abar, abarPrev} */
  function build(name, T) {
    const f = BUILDERS[name] || betasCosine;
    return derive(f(T));
  }

  /** log-SNR(t) = log(abar / (1 - abar)) */
  function logSNR(sch, i) {
    const a = sch.abar[i];
    return Math.log(Math.max(a, 1e-12) / Math.max(1 - a, 1e-12));
  }

  DL.Schedule = {
    build, derive, logSNR,
    names: Object.keys(BUILDERS),
    COLORS, LABELS,
  };
  DL.rand = { mulberry32, gaussian, linspace };
})(window);
