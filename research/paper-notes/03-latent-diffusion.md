# 论文精读 03 · Latent Diffusion Models（Stable Diffusion）

> **High-Resolution Image Synthesis with Latent Diffusion Models**
> Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
> CVPR 2022 (Oral, Best Paper 候选) · arXiv:2112.10752
> 官方代码：https://github.com/CompVis/latent-diffusion （后续为 Stability-AI/stablediffusion）

---

## 1. 一句话概括

把扩散过程从「像素空间」搬到**预训练自编码器的低维隐空间**上去做，
配合 cross-attention 做通用条件控制，在**算力下降一个数量级**的同时首次把扩散模型推到
百万级数据、1024×1024 分辨率的实用规模——这就是 Stable Diffusion 的技术底座。

## 2. 动机：像素空间扩散的两笔浪费

作者非常直白地把扩散模型的训练开销拆成两部分：

1. **感知压缩率不足**：一张 512×512×3 的图里，绝大部分像素信息是高频细节，
   对这些细节做扩散，等于让模型把容量花在"人眼不在乎的东西"上。
2. **语义信息冗余**：图像有很强的局部相关性和空间规律性。

结论：先在**语义保留但维度大幅压缩**的空间里做生成，最后再解码回像素。

## 3. 方法

### 3.1 两阶段训练

| 阶段 | 内容 | 关键 |
|------|------|------|
| 阶段一 | 训练一个 VAE-like 自编码器（ perceptual loss + 少量 KL + patch 判别器） | 下采样因子 $f\in\{4,8,16\}$ |
| 阶段二 | 在隐空间 $z=\mathcal E(x)$ 上训练标准扩散模型（UNet + 时间步/条件） | 训练与采样都在低维空间 |

损失：$\mathbb E_{\mathcal E(x),y,\epsilon,t}\big\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\big\|_2^2$

### 3.2 为什么 KL 项要"很弱"

作者用了两种正则：KL（得到标准高斯隐空间）和 VQ（离散化）。
并且强调 **KL 权重必须很小**（甚至用 VQ），否则隐空间方差被压得太狠，重建会糊。
这是一个典型的「生成质量 vs 隐空间规整性」的权衡：
隐空间越像标准高斯，扩散越好学；但压缩得越狠，解码越糊。

### 3.3 条件注入：cross-attention

$$\text{Attention}(Q,K,V)=\text{softmax}\Big(\frac{QK^\top}{\sqrt d}\Big)V,\quad
Q=W_Q\phi_i(z_t),\ K=W_K\tau_\theta(y),\ V=W_V\tau_\theta(y)$$

其中 $\tau_\theta$ 是任意领域专用编码器（文本 → BERT/CLIP、布局 → 图网络、类别 → 嵌入）。
**这一个设计让同一套 UNet 骨架能服务文生图、超分、inpainting、语义合成等多种任务**，
是"通用生成基座"思路的雏形。

### 3.4 效率账

| 指标 | 像素扩散 | Latent Diffusion (f=8) |
|------|------|------|
| 训练 GPU 天（同画质） | ~1000 V100 days（论文估算） | ~ tens of V100 days |
| 单卡推理 | 慢，显存吃紧 | 消费级显卡可跑 |

## 4. 主要实验结论

- 在 ImageNet / CelebA-HQ / FFHQ / COCO 上，类条件与文生图任务取得当时 SOTA 或接近 SOTA 的 FID。
- 文生图在 COCO 上 FID 显著优于 DALL-E、GLIDE，且参数量和算力更低。
- 下采样因子消融：小的 $f$（4）偏慢但重建好；大的 $f$（16）快但细节损失明显 → **$f=8$ 是实操甜点**。
- 同一骨架可以直接迁移到超分、inpainting、语义合成、class-conditional 四类任务（只需换条件编码器）。

## 5. 我的理解与思考

1. **这篇文章的方法论价值大于技术新颖性**。它没有提出新的扩散公式（用的就是 DDPM 那一套），
   真正的贡献是**重新划定了「生成」和「压缩」的职责边界**：
   自编码器负责「感知层面的像素细节」，扩散模型负责「语义层面的分布建模」。
   这种"让每个模块只做它擅长的事"的分解思路，是我在读论文时最受用的部分。

2. **"两阶段"带来的问题也不可忽视**：
   - 第一阶段 VAE 的重建误差会成为生成质量的天花板（SD 早期被人吐槽的手部/文字问题，一部分根源在这里）
   - 两阶段误差会累积，且第一阶段训练完就冻结，无法被第二阶段反向修正
   → 我认为「端到端联合微调自编码器」或者「用更强的 tokenizer（如 VQ-GAN 改进版 / 1D tokenizer）」
   是仍有油水可挖的方向（见 idea-02）。

3. **cross-attention 作为通用条件接口，是后来 ControlNet / IP-Adapter / T2I-Adapter 的先决条件**。
   可以说 SD 生态之所以能长出来，就是因为条件注入点被设计成了一个"插件位"。
   这提示我：**架构设计时要为未来的扩展预留标准化接口**。

4. **一个容易被低估的细节**：隐空间扩散的噪声调度要**重新调**。
   因为隐变量的数值分布与像素完全不同（方差尺度不同），直接照搬像素空间的 $\beta$ 调度会出问题。
   作者的复现提醒我：**换空间 = 换分布 = 换超参，不能无脑迁移。**

5. **局限**：VAE 解码是确定性的、单次的，无法表达"像素层面的多模态"（比如精细纹理的多种可能），
   这也是后来 pixel-space 的 cascade / 上采样模块存在的理由。

## 6. 复现情况

- 本次**未**复现（完整 LDM 需要 GPU + 大规模数据，超出本机 CPU 条件）。
- 我在 [`../reproduction/`](../reproduction/) 中复现的是 DDPM / DDIM 这两篇更基础的机制性工作；
  LDM 的架构思想被用在了我的研究想法 [`../ideas/idea-02-condition-adapter.md`](../ideas/idea-02-condition-adapter.md) 中。
- 明确标注：本节内容来自论文阅读，非实验结果。

## 7. 遗留问题

- 自编码器与扩散模型能否在训练后期做低成本的联合微调？
- $f$ 能否随图像内容自适应（纹理密区域用小 $f$，平滑区域用大 $f$）？
