/* 反向采样可视化：2D 高斯混合 + 解析 score
 *
 * 为什么这里能"真采样"而不需要训练？
 *   数据 p_0 是 K 个各向同性高斯分量的混合；前向加噪是线性高斯的，
 *   所以 p_t 仍然是高斯混合：
 *       p_t(x) = Σ_k π_k N(x; √ᾱ_t μ_k, (ᾱ_t σ² + 1 − ᾱ_t) I)
 *   于是 score 可以闭式写出，再由
 *       ε*(x, t) = −√(1−ᾱ_t) · ∇log p_t(x)
 *   得到贝叶斯最优的噪声预测器 —— 它等价于"训练到最优"的 DDPM 模型。
 *   因此下面的采样与论文中的 DDPM / DDIM 公式完全一致，只是把 ε_θ 换成了解析最优解。
 */
(function (global) {
  'use strict';
  const DL = (global.DL = global.DL || {});
  const T = 1000;

  /* ---------------- 玩具数据：8 个高斯分量 ---------------- */
  const K = 8, R = 2.2, SIG = 0.26;
  const MU = [];
  for (let k = 0; k < K; k++) {
    const a = (2 * Math.PI * k) / K;
    MU.push([R * Math.cos(a), R * Math.sin(a)]);
  }
  const PI = new Array(K).fill(1 / K);
  const LOGNORM = -Math.log(2 * Math.PI * SIG * SIG);

  function logp0(x, y) {
    // log Σ_k π_k N(x; μ_k, σ²I)
    const m = new Array(K);
    for (let k = 0; k < K; k++) m[k] = -(((x - MU[k][0]) ** 2 + (y - MU[k][1]) ** 2) / (2 * SIG * SIG)) + LOGNORM;
    let mx = -Infinity;
    for (let k = 0; k < K; k++) if (m[k] > mx) mx = m[k];
    let s = 0;
    for (let k = 0; k < K; k++) s += PI[k] * Math.exp(m[k] - mx);
    return Math.log(s) + mx;
  }

  /* ---------------- 调度 ---------------- */
  let sch = null;
  function abar(t) { return t <= 0 ? 1 : sch.abar[Math.min(t, T) - 1]; }

  /* ---------------- 解析 score ---------------- */
  function scoreAndEps(px, py, t) {
    const a = abar(t);
    const v = a * SIG * SIG + (1 - a);          // p_t 各分量的方差
    const sa = Math.sqrt(a), sn = Math.sqrt(1 - a);
    const m0 = new Array(K), m1 = new Array(K), w = new Array(K);
    let mx = -Infinity;
    for (let k = 0; k < K; k++) {
      const c0 = sa * MU[k][0], c1 = sa * MU[k][1];
      m0[k] = c0; m1[k] = c1;
      w[k] = -(((px - c0) ** 2 + (py - c1) ** 2) / (2 * v)) - Math.log(2 * Math.PI * v);
      if (w[k] > mx) mx = w[k];
    }
    let sw = 0;
    for (let k = 0; k < K; k++) { w[k] = Math.exp(w[k] - mx); sw += w[k]; }
    let s0 = 0, s1 = 0;
    for (let k = 0; k < K; k++) {
      const p = w[k] / sw;
      s0 += p * (-(px - m0[k]) / v);
      s1 += p * (-(py - m1[k]) / v);
    }
    // ε* = −√(1−ᾱ_t) · ∇log p_t
    return [-sn * s0, -sn * s1];
  }

  /* ---------------- 一步反向（DDIM 论文 Eq.12 的通用形式） ----------------
   * η = 0  → DDIM（确定性 ODE）
   * η = 1  → DDPM 的随机 ancestral 采样（σ² 等于后验方差）
   */
  function step(P, t, s, eta, rnd) {
    const a_t = abar(t), a_s = abar(s);
    const sa_t = Math.sqrt(a_t), sn_t = Math.sqrt(1 - a_t);
    const sa_s = Math.sqrt(a_s);
    let sigma = 0;
    if (eta > 0) {
      const inner = 1 - a_t / a_s;
      sigma = eta * Math.sqrt((1 - a_s) / (1 - a_t)) * Math.sqrt(Math.max(0, inner));
    }
    const dirCoef = Math.sqrt(Math.max(0, 1 - a_s - sigma * sigma));
    for (let i = 0; i < P.length; i++) {
      const p = P[i];
      const [e0, e1] = scoreAndEps(p[0], p[1], t);
      const x0_0 = (p[0] - sn_t * e0) / sa_t;
      const x0_1 = (p[1] - sn_t * e1) / sa_t;
      p[0] = sa_s * x0_0 + dirCoef * e0 + (sigma ? sigma * DL.rand.gaussian(rnd) : 0);
      p[1] = sa_s * x0_1 + dirCoef * e1 + (sigma ? sigma * DL.rand.gaussian(rnd) : 0);
    }
  }

  /* ---------------- 时间步子序列 ---------------- */
  function makeSchedule(S, spacing) {
    if (spacing === 'snr') {
      const lo = DL.Schedule.logSNR(sch, 0);      // t=1
      const hi = DL.Schedule.logSNR(sch, T - 1);  // t=T
      const out = [];
      for (let k = 0; k < S; k++) {
        const target = hi + (lo - hi) * (k / (S - 1));
        let t = 1, best = Infinity;
        for (let i = 0; i < T; i++) {
          const d = Math.abs(DL.Schedule.logSNR(sch, i) - target);
          if (d < best) { best = d; t = i + 1; }
        }
        out.push(t);
      }
      out.push(0);
      return dedup(out);
    }
    const out = [];
    for (let k = 0; k < S; k++) out.push(Math.round(T - ((T - 1) * k) / (S - 1)));
    out.push(0);
    return dedup(out);
  }

  function dedup(arr) {
    const res = [];
    for (const v of arr) if (res.length === 0 || res[res.length - 1] !== v) res.push(v);
    return res;
  }

  /* ---------------- 采样主函数 ---------------- */
  function sample(opts) {
    const N = opts.n, S = opts.steps, eta = opts.eta, spacing = opts.spacing || 'uniform';
    const rnd = DL.rand.mulberry32(opts.seed || 1);
    const P = new Array(N);
    for (let i = 0; i < N; i++) P[i] = [DL.rand.gaussian(rnd) * 1.6, DL.rand.gaussian(rnd) * 1.2];
    const ts = makeSchedule(S, spacing);
    const t0 = performance.now();
    for (let i = 0; i < ts.length - 1; i++) {
      step(P, ts[i], ts[i + 1], eta, rnd);
      if (opts.onStep) opts.onStep(P, ts[i], ts[i + 1], i);
    }
    const ms = performance.now() - t0;
    return { P, ms, ts, metrics: evaluate(P) };
  }

  /* ---------------- 指标 ---------------- */
  function evaluate(P) {
    let ll = 0, hit = 0;
    for (let i = 0; i < P.length; i++) {
      ll += logp0(P[i][0], P[i][1]);
      let d = Infinity;
      for (let k = 0; k < K; k++) {
        const dd = Math.hypot(P[i][0] - MU[k][0], P[i][1] - MU[k][1]);
        if (dd < d) d = dd;
      }
      if (d < 3 * SIG) hit++;
    }
    return { loglik: ll / P.length, hit: hit / P.length };
  }

  DL.Reverse = { sample, evaluate, logp0, step, makeSchedule, abar, MU, SIG, K, T,
    setSchedule: (s) => { sch = s; },
    getSchedule: () => sch };
})(window);
