# 研究想法 01 · ATS-Diff：采样时间步的自适应稀疏化

> 状态：初步构想（未实验） · 产生于精读 DDIM 之后
> 拟投：**ICML / NeurIPS**（机器学习顶会）或 **IEEE TMM**（如果最终偏应用）

---

## 1. 要解决的问题（Motivation）

DDIM 证明了：一个用 1000 步训练好的扩散模型，采样时**只需要走其中 10~50 步**就能保住画质。
但 DDIM 的原论文和几乎所有开源实现，挑选这几十步的方式都是**手工的、均匀的**：

$$\tau=\big\{\lfloor i\cdot T/S\rfloor\big\}_{i=0}^{S-1}\quad\text{（或按 }\bar\alpha\text{ 均匀分位）}$$

我认为这里有一个被忽略的不合理性：**不同时间步对最终结果的贡献是不均匀的**。

直觉上有两类证据支持这一点：

1. **信噪比视角**：反向过程在不同 $t$ 上的局部离散化误差不同。
   在 $\bar\alpha$ 变化剧烈的区间（通常是噪声水平中等的那一段，即"语义成型期"），
   跳步带来的误差最大；而在 $t\to T$（纯噪声区）和 $t\to 0$（细节打磨区），
   跳步的代价相对小。均匀跳步等于给所有区间分配了相同的预算，显然不是最优。
2. **经验视角**：社区里广为流传的实测结论（如 DPM-Solver、UniPC 的消融）显示，
   手工调整时间步分配能在同等步数下再压低一点 FID，说明**存在可优化空间**，
   但目前缺少一个「有原则的、可学习的」分配方法。

**问题陈述**：给定采样步数预算 $S$，如何自动决定取哪 $S$ 个时间步（以及它们的分布），
使得生成质量（FID / 感知指标）最优？

## 2. 核心思路 / 创新点

**核心思路：把"选哪些时间步"建模成一个可以离线求解的一维资源分配问题，
用一个轻量打分器（scorer）预测每个候选步的"跳过代价"，再用动态规划在预算内选步。**

具体三块：

### (1) 定义"单步跳过代价"

对相邻两步 $t\to s$（跳过了中间若干步），定义局部截断误差的代理量：
用同一个模型在同一批 $x_t$ 上做「一步跳到 $s$」和「逐步走到 $s$」两次采样，
比较得到的 $x_s$ 的差异：

$$c(t,s)=\mathbb E_{x_t}\Big\|x_s^{\text{jump}}-x_s^{\text{stepwise}}\Big\|_2^2$$

这个量可以在**训练完成后离线统计一次**（对一批数据、几百个时间步对），成本很低。

### (2) 轻量打分器泛化到任意数据集/调度

离线统计只对"见过的"配置有效。因此训练一个极小的 MLP scorer
（输入：$t$、$\bar\alpha_t$、$\bar\alpha_s$、局部 SNR 变化量；输出：预测代价），
用它把代价函数泛化到新的调度与数据集，避免每次重新统计。

### (3) 预算内的动态规划选步

在时间步网格上做一次 DP：

$$\min_{\tau:\,|\tau|=S}\ \sum_{i}c(\tau_{i+1},\tau_i)$$

因为代价近似可加，DP 是多项式的且只需在**采样前算一次**（毫秒级），
之后所有样本共享同一套时间步 → **推理时零额外开销**。

### 与已有工作的区别

| 工作 | 做法 | 与本想法的区别 |
|------|------|------|
| DDIM | 均匀跳步 | 固定策略，不因模型/数据而变 |
| DPM-Solver / UniPC | 换更高阶的求解器 | 改的是"怎么算"，不是"在哪几步算"；两者正交可叠加 |
| 步数蒸馏（Consistency / Progressive Distillation） | 重新训练模型 | 需要额外训练成本；本想法**完全不动模型权重** |
| 早期工作（学习采样调度, 如 Watson et al.） | 用 ELBO 相关量做贪心选步 | 只对 DDPM 的随机采样成立；本想法面向 DDIM/ODE 的确定性采样，且用可学习的代价模型 |

## 3. 初步实验方案

**阶段 A：验证"均匀不是最优"（最关键，成本最低）**

- 复用本仓库已训练好的 MNIST DDPM 权重（[`../reproduction/01-ddpm-mnist/`](../reproduction/01-ddpm-mnist/)）。
- 基线：均匀跳步的 DDIM，S ∈ {50, 20, 10}。
- 对照组：随机搜索 / 网格搜索 200 组非均匀时间步分配，统计最优与均匀策略的指标差。
- **判据**：若最优非均匀策略比均匀策略在 FID 近似上有稳定（多次 seed）优势，则假设成立。
  若没有优势，说明该方向价值有限，及时止损。

**阶段 B：代价模型能否预测"最优"**

- 统计 $c(t,s)$ 真值，与 DP 选出的时间步做相关性分析（Spearman）。
- 看 DP 选出的 $\tau$ 是否落在阶段 A 搜索出的优秀区域内。

**阶段 C：上规模验证**

- 数据集：CIFAR-10（无条件/类条件），模型用开源的 DiT-S/2 或 ADM 权重（不自己训练）。
- 指标：FID、sFID、Inception Score、采样耗时。
- 对比：均匀 DDIM、DPM-Solver-2/3、本方法、本方法 + DPM-Solver（验证正交可叠加）。
- 额外验证：同一套 DP 时间步迁移到 CelebA-HQ / LSUN，检验泛化性。

**风险与应对**

| 风险 | 应对 |
|------|------|
| 阶段 A 无差异 | 说明均匀策略已接近最优 → 转去研究"步数预算与调度类型（cosine/poly/EDM）的交互" |
| DP 代价不可加 | 退化为贪心选步 + 局部微调，仍可接受 |
| scorer 泛化差 | 去掉 scorer，改为"每个数据集离线统计一次"，仍是一个可用工具 |

## 4. 拟投会议 / 期刊

- 首选：**ICML** 或 **NeurIPS**（高效生成/采样方向，属于主流的"efficiency"赛道）
- 备选：**IEEE TMM / ACM MM**（若最终做成偏应用的"即插即用采样加速工具"）
- 时间上若成果扎实，可先挂 arXiv  preprint，再投最近的会议周期。

## 5. 参考文献

1. Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021. arXiv:2010.02502
2. Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. arXiv:2006.11239
3. Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021. arXiv:2102.09672
4. Lu et al. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*. NeurIPS 2022. arXiv:2206.00927
5. Zhao et al. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models*. NeurIPS 2023. arXiv:2302.04867
6. Watson et al. *Learning to Efficiently Sample from Diffusion Probabilistic Models*. arXiv:2106.03802
7. Song et al. *Consistency Models*. ICML 2023. arXiv:2303.01469
8. Peebles & Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. arXiv:2212.09748
