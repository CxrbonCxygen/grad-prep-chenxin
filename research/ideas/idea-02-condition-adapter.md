# 研究想法 02 · LoRA-Gate：多条件注入时的低秩门控解耦

> 状态：初步构想（未实验） · 产生于精读 Latent Diffusion 与 DiT 之后
> 拟投：**ACM MM / ICCV**（多模态生成方向）

---

## 1. 要解决的问题（Motivation）

Latent Diffusion 用 **cross-attention** 作为通用条件接口，DiT 用 **adaLN-Zero** 做条件调制。
这两套设计让同一个基座模型能接文本、布局、深度图、参考图等多种条件，
也直接催生了 ControlNet、IP-Adapter、T2I-Adapter 这一整条"插件生态"。

但**多条件同时注入**时会出问题：

1. **条件竞争 / 相互稀释**：文本 prompt 要求"一只猫"，结构条件要求"这个姿态"，
   两者在 attention 里争夺同一组特征通道，实测中常表现为
   "结构对齐了但语义丢了"或"语义对了但姿态跑偏"。
2. **插件叠加的灾难**：同时插 ControlNet(深度) + IP-Adapter(风格) 时，
   社区经验是要手工调各自的权重系数，且换一个 prompt 就要重调。
3. **全参数微调代价高**：每加一个新条件就微调一次基座，会产生 N 份大模型副本。

**问题陈述**：能否让多个条件在注入时**自动协商各自的生效强度与生效位置**，
使得 (a) 不互相稀释，(b) 新增条件无需改动基座，(c) 推理开销可忽略？

## 2. 核心思路 / 创新点

**核心思路：把每个条件封装成一个「低秩适配器 + 零初始化门控」，
门控值由"当前条件与当前空间位置/语义通道的相关性"动态决定，实现条件间的自适应解耦。**

### (1) 结构：LoRA-Adapter

对基座（冻结）的每个注意力/调制层，为每个条件 $k$ 挂一组低秩分支：

$$h' = h + \underbrace{g_k(x,t)\cdot B_k A_k h}_{\text{条件 }k\text{ 的贡献}},\quad A_k\in\mathbb R^{r\times d},\ B_k\in\mathbb R^{d\times r},\ r\ll d$$

- 基座冻结 → **新增条件只增加约 0.1%~1% 参数**，可插拔、可分发。
- 多个条件的低秩分支**并行**挂载，互不覆盖。

### (2) 关键创新：动态门控 $g_k(x,t)$

不是给每个条件一个固定权重，而是让网络自己算：

- 输入：当前时间步嵌入 $t$、该层特征的通道统计量（pooling 后）、
  以及该条件的全局嵌入 $e_k$；
- 输出：$g_k\in[0,1]^{d}$（**逐通道**的门控向量，而非标量）；
- 归一化：跨条件做 softmax 或 sparsemax，使 $\sum_k g_k \le 1$，
  从机制上保证"总注入能量有上限"，从而**结构上避免条件稀释**。

**为什么是逐通道门控**：我的假说是——不同条件天然对应不同的特征通道子集
（结构/边缘类条件偏好低频通道，风格/纹理类偏好高频通道）。
逐通道门控让它们在通道维度上"各占一段"，而标量门控只能在强度上此消彼长。
这条假说是可以被验证的（见实验方案）。

### (3) 训练策略

- 基座冻结，**只训练 adapter + gate**（参数量极小，单卡可训）。
- **条件 dropout**：训练时随机丢弃某个条件（类似 classifier-free guidance 的做法），
  强迫 gate 学会"缺条件时把门关上"，同时获得无条件分支以便做 CFG。
- 加一个**正交正则** $\sum_{k\neq l}\langle \bar g_k,\bar g_l\rangle$，显式推动不同条件占用不同通道。

### 与已有工作的区别

| 工作 | 做法 | 与本想法的区别 |
|------|------|------|
| ControlNet | 复制一份编码器，输出加到残差上，权重固定 | 全参数、条件间无协商机制 |
| IP-Adapter | 解耦 cross-attn 的 K/V | 只针对图像条件，且强度是标量 |
| LoRA / DreamBooth | 低秩微调 | 用于"学概念"，不做多条件动态门控 |
| adaLN-Zero (DiT) | 零初始化门控 | 只处理单一条件向量；本想法扩展到**多条件 + 逐通道 + 归一化预算** |

## 3. 初步实验方案

**阶段 A：在小规模上验证"逐通道门控 > 标量门控"**

- 数据：MNIST / CIFAR-10，两个简单条件：类别标签 + 颜色/位置属性。
- 模型：小 DiT 或本仓库的 UNet（[`../reproduction/01-ddpm-mnist/`](../reproduction/01-ddpm-mnist/) 可直接扩展）。
- 对比：① 固定标量权重；② 可学习标量；③ 逐通道门控（本方法）；④ 逐通道 + 正交正则。
- 指标：两个条件各自的**条件准确率**（用预训练分类器判定生成图是否满足条件），
  以及「双条件同时满足率」——这是核心指标，能直接反映"是否互相稀释"。
- **判据**：若 ③/④ 的双条件同时满足率显著高于 ②，则核心假设成立。

**阶段 B：验证通道占用假说（可解释性实验）**

- 可视化 $\bar g_k$（对 batch 平均后的通道分布），看不同条件是否真的占据了不同通道段。
- 若假说成立，这本身就构成一个有说服力的分析性贡献。

**阶段 C：上规模**

- 基座：Stable Diffusion 1.5 / SDXL（冻结）。条件：文本 + canny/depth + 风格参考图。
- 对比：ControlNet、IP-Adapter、两者简单叠加、本方法。
- 指标：CLIP-Score（语义）、结构相似度（边缘/深度对齐误差）、FID、人工偏好评测。
- 效率：报告 adapter 参数量、显存增量、推理延迟增量。

**风险与应对**

| 风险 | 应对 |
|------|------|
| 门控塌缩（一个条件吃掉所有预算） | 加熵正则 / 给每个 gate 设下界；或改用 softmax temperature 退火 |
| 低秩容量不足 | 提高 rank；或对"强结构类"条件保留 ControlNet 式分支，混合架构 |
| 训练数据需要多条件标注 | 阶段 A/B 用可合成的条件（CIFAR 的颜色/位置），无需额外标注 |

## 4. 拟投会议 / 期刊

- 首选：**ACM MM**（多模态 + 生成，非常契合）或 **ICCV / ECCV**（若视觉生成实验做得扎实）
- 备选：**IEEE TIP / TMM**（做成通用的多条件控制框架）

## 5. 参考文献

1. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. arXiv:2112.10752
2. Peebles & Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. arXiv:2212.09748
3. Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*. ICCV 2023. arXiv:2302.05543
4. Ye et al. *IP-Adapter: Text Compatible Image Prompt Adapter*. arXiv:2308.06721
5. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. arXiv:2106.09685
6. Ho & Salimans. *Classifier-Free Diffusion Guidance*. arXiv:2207.12598
7. Mou et al. *T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models*. AAAI 2024. arXiv:2302.08453
8. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018. arXiv:1709.07871
