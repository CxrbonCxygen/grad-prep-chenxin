# 科研部分 · 扩散生成模型（Diffusion Models）

研究方向：**扩散生成模型的高效采样、网络架构与条件控制**。
以 DDPM → DDIM → Latent Diffusion → DiT 这条主线精读，用「机制性复现」验证自己的理解，
再基于此提出两个自己的研究想法。

---

## 一、精读论文列表

| # | 论文 | 会议 / 年份 | 为什么选它 | 笔记 | 复现 |
|---|------|-----------|-----------|------|------|
| 01 | Denoising Diffusion Probabilistic Models (Ho et al.) | **NeurIPS 2020** | 整个领域的奠基工作，定义了训练目标与参数化方式 | [笔记](./paper-notes/01-ddpm.md) | [✅ 已复现](./reproduction/01-ddpm-mnist/) |
| 02 | Denoising Diffusion Implicit Models (Song et al.) | **ICLR 2021** | 把采样从 1000 步打到 10~50 步，也是后续所有加速工作的起点 | [笔记](./paper-notes/02-ddim.md) | [✅ 已复现](./reproduction/02-ddim-sampling/) |
| 03 | High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al.) | **CVPR 2022** | Stable Diffusion 的技术底座，把扩散搬进隐空间 | [笔记](./paper-notes/03-latent-diffusion.md) | ❌ 未复现（需 GPU + 大规模数据） |
| 04 | Scalable Diffusion Models with Transformers / DiT (Peebles & Xie) | **ICCV 2023** | 近 3 年代表性工作：用 Transformer 替换 U-Net，证明可扩展性 | [笔记](./paper-notes/04-dit.md) | ❌ 未复现（需大规模算力） |

> 说明：前两篇虽不在「近 3 年」范围内，但它们是后两篇的直接前提，
> 不读透就无法判断后两篇到底改进了什么，因此一并精读。
> 论文出处与链接见 [papers/README.md](./papers/README.md)（不上传 PDF，遵守版权）。

## 二、复现进度与结果

本机环境：**无 NVIDIA GPU，纯 CPU（8 核）**。因此复现策略是
「**保留完整机制，压缩数据规模与模型容量**」——目标是把论文的算法流程真实跑通并拿到可验证的结果，
而不是追求论文原报告的指标。

### [01 · DDPM 训练与采样](./reproduction/01-ddpm-mnist/)

| 项目 | 内容 |
|------|------|
| 实现内容 | 噪声调度（linear / cosine）、前向闭式加噪 $q(x_t\mid x_0)$、$L_{\text{simple}}$ 训练、epsilon 预测 UNet、ancestral 反向采样 |
| 数据 / 模型 | MNIST 32×32，20k 子集，UNet 0.64M 参数，T=1000，cosine 调度，20 epoch |
| 产出 | loss 曲线、训练过程采样网格、最终 64 张样本、去噪轨迹图、指标 JSON |
| 验证的现象 | ① loss 单调下降，训练无不稳定；② 训练中期即可生成可辨识数字；③ 采样步数从 1000 降到 200 时画质几乎无损 → 印证 DDIM 论文"大量步数被浪费"的观察 |

### [02 · DDIM 加速采样对比](./reproduction/02-ddim-sampling/)

| 项目 | 内容 |
|------|------|
| 实现内容 | DDPM/DDIM 统一采样式（η=0/1）、子序列跳步、静态钳制（clip_denoised） |
| 关键设计 | **复用 01 训练好的权重，完全不重新训练** —— 这正是 DDIM 的核心卖点 |
| 对比维度 | 采样器 {DDPM ancestral, DDIM} × 步数 {1000, 250, 100, 50, 20, 10} |
| 指标 | 采样耗时（加速比）、自训练 MNIST 分类器（acc 0.954）特征空间的 FID 近似、分类置信度/类别熵 |
| 实际结论 | ① 加速成立：耗时与步数严格成正比，S=50 提速 21 倍且质量可用；② 与论文预期不同，本设定下 DDPM(η=1) 质量更稳——欠训练模型下确定性 ODE 会放大 ε_θ 误差（实测 |x̂0| 发散到 4414，需静态钳制），这个调试过程本身比指标更有收获 |

## 三、研究想法总览

| # | 想法 | 一句话 | 拟投 | 文档 |
|---|------|--------|------|------|
| 01 | 自适应时间步稀疏化（ATS-Diff） | DDIM 的跳步子序列是手工均匀取的，改成由一个轻量打分器按「每步的离散化误差」自适应分配步数预算 | **ICML / NeurIPS** 或 IEEE TMM | [idea-01](./ideas/idea-01-adaptive-step-sparsity.md) |
| 02 | 条件适配器的低秩门控融合（LoRA-Adapter） | 多条件同时注入时 cross-attention 会互相干扰，用低秩 + 零初始化门控做条件解耦 | **ACM MM / ICCV** | [idea-02](./ideas/idea-02-condition-adapter.md) |

## 四、原创性说明

- 4 篇笔记的「我的理解与思考」「遗留问题」部分为本人独立撰写，公式与实验结论引自原论文并已注明出处。
- 复现代码为本人按论文算法从零编写（未直接搬运官方实现），可在本机环境完整跑通。
- 所有 `results/` 下的曲线与图片均来自本机实际运行，未做任何美化修饰。
