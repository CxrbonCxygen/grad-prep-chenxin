# 论文精读 01 · DDPM

> **Denoising Diffusion Probabilistic Models**
> Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
> NeurIPS 2020 (Oral) · arXiv:2006.11239
> 官方代码：https://github.com/hojonathanho/diffusion
> 扩展版：*Improved Denoising Diffusion Probabilistic Models*（Nichol & Dhariwal, ICML 2021, arXiv:2102.09672）

---

## 1. 一句话概括

把「逐步加噪」和「逐步去噪」都限定为高斯马尔可夫链，用**预测噪声**这个极简目标训练一个 UNet，
第一次让扩散模型在无条件图像生成上达到（甚至部分超过）GAN 的画质，**并且训练过程极其稳定**。

## 2. 动机：为什么要用扩散模型

生成模型长期被三类目标函数支配：

| 范式 | 目标 | 痛点 |
|------|------|------|
| GAN | 对抗损失 | 训练不稳定、模式坍塌、需要调一堆 trick |
| VAE / Flow | 似然（ELBO） | 表示能力受限（Flow 要求可逆、VAE 的后验假设太强） |
| 自回归 | 逐像素似然 | 采样慢、无法并行 |

扩散模型的想法是：**不去硬扛一个难优化的损失，而是把一个难问题拆成 T 个简单问题**——
每一步只需要「从一张带噪图里把噪声猜出来」，这是一个有唯一正确答案的回归任务。

## 3. 方法拆解

### 3.1 前向过程（固定、不可学习）

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\quad q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$

由于是线性高斯链，**可以一步跳到任意 t**（这是全文最关键的性质）：

$$q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad \alpha_t=1-\beta_t,\ \ \bar\alpha_t=\prod_{s=1}^{t}\alpha_s$$

于是训练时**不需要真的跑 T 步**：直接采样 $t$，采样 $\epsilon$，闭式得到 $x_t$。

### 3.2 反向过程（可学习）

$p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\ \sigma_t^2 I)$，其中 $\sigma_t^2$ 固定为 $\beta_t$ 或后验方差。

### 3.3 三种等价的参数化（论文 Section 3.2）

论文推导出优化 $\mu_\theta$ 等价于优化下面三者之一，作者最终选了**预测噪声 $\epsilon$**：

1. 直接预测 $\mu_\theta$（后验均值）
2. 预测 $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$
3. **预测 $\epsilon$** ← 采用

### 3.4 简化目标（论文 Eq.14）

$$L_{\text{simple}} = \mathbb E_{t,x_0,\epsilon}\Big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\Big\|^2$$

**注意这里丢掉了变分下界里的权重系数**。作者的理由是：加权后的目标会让模型把容量浪费在
「 $\bar\alpha_t$ 接近 1、几乎没什么噪声」的简单时间步上，而简化目标把容量均匀分配、
且在实际中生成质量更好。代价是：$L_{\text{simple}}$ 不再是一个严格的似然下界（论文也没报 bits/dim 作为主指标）。

### 3.5 关键点：共享参数 + 时间步条件

同一个网络处理所有噪声级别，时间步 $t$ 用 sinusoidal 位置编码注入到每个残差块（类似 Transformer）。
这让「一个模型 = T 个专用去噪器」，参数量与 T 无关。

### 3.6 架构

U-Net：卷积残差块 + 多尺度下/上采样 + 全局时间步嵌入；后续工作加了自注意力（在 16×16 尺度）。

## 4. 主要实验结论

| 结论 | 内容 |
|------|------|
| 画质 | CIFAR-10 无条件 FID **3.17**，超过当时最好的 GAN（SOTA）；LSUN 也很好 |
| Inception Score | CIFAR-10 IS 9.46（高质量 + 多样性兼顾） |
| 似然 | 不如自回归模型，但优于之前的多数似然模型 |
| 采样代价 | 需要 T=1000 次网络前向，比 GAN 慢 1~2 个数量级 ← **最大短板** |
| 训练稳定性 | 不需要对抗训练，loss 单调下降，几乎没有 trick |

## 5. 我的理解与思考

1. **"拆成很多简单回归问题" 是这篇文章真正的贡献**，而不是数学上的变分下界。
   下界推导的作用主要是给出一个「可以优化的东西」，真正让效果变好的是：
   (a) 预测噪声的参数化，(b) 把权重丢掉（$L_{\text{simple}}$），(c) 参数在所有时间步共享。
   后面一整条工作线（DDIM、improved DDPM、classifier-free guidance）都是在这三点上做手术。

2. **$L_{\text{simple}}$ 丢弃权重这一点很有意思**：它说明「优化一个理论上正确的目标」
   和「优化一个实际好用的目标」之间确实存在 gap。生成模型中这类「理论目标 vs 感知质量」
   的矛盾很常见（例如 VAE 的 KL 项要打折），DDPM 给了一个干净的处理范式。

3. **为什么必须是 U-Net？** 因为去噪任务的输入输出同分辨率、需要多尺度上下文，
   而且卷积的局部性归纳偏置在小样本下更省参数。后面 DiT 用 ViT 替换它，
   说明这个归纳偏置在数据量足够时是可以被换掉的——这正好是「架构红利 vs 数据规模」的经典权衡。

4. **一个容易被忽略的细节**：论文 Eq.7 中，当 $t=1$ 时后验方差取 $\beta_1$ 而不是 $\tilde\beta_1$，
   且最后一步不再加噪声（$\sigma_t=0$）。我在复现里也照做了，否则生成图会残留一层颗粒感。

5. **局限**：1000 步采样让 diffusion 在相当长时间里被诟病为「慢」。
   这直接催生了 DDIM（我的第 2 篇笔记）——它把反向过程从「马尔可夫采样」改成「确定性 ODE 积分」，
   从而允许大步跳跃。**这也是我下面复现实验想亲自验证的点。**

## 6. 复现情况

- 复现目录：[`../reproduction/01-ddpm-mnist/`](../reproduction/01-ddpm-mnist/)
- 内容：从零实现噪声调度、前向闭式加噪、UNet（epsilon 预测）、ancestral 采样。
- 环境：CPU（本机无 GPU），MNIST 32×32、小规模子集，T=1000、cosine 调度。
- 验证的现象：loss 单调下降；采样步数从 1000 降到 200 时，画质**没有明显下降**
  → 说明大量采样步确实被"浪费"了，这与 DDIM 论文的观察一致。

## 7. 遗留问题

- $L_{\text{simple}}$ 与真实 ELBO 之间的 gap 到底有多大？是否可以用「按 SNR 加权」的折中目标同时拿到两者好处？
- 反向过程的方差 $\sigma_t^2$ 固定在两个极端（$\beta_t$ 或 $\tilde\beta_t$）之间，
  中间是否有更好的插值（这一点 improved DDPM 用可学习方差回答了）。
