# 论文精读 02 · DDIM

> **Denoising Diffusion Implicit Models**
> Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford)
> ICLR 2021 (Oral) · arXiv:2010.02502
> 官方代码：https://github.com/ermongroup/ddim

---

## 1. 一句话概括

指出 DDPM 的训练目标其实**只依赖于边缘分布 $q(x_t\mid x_0)$，而不依赖于那条特定的马尔可夫前向链**，
于是可以为「同一个训练好的模型」重新配一条**非马尔可夫、且可以确定性（无随机噪声）**的反向过程，
从而用 10~50 步就采样出和 1000 步 DDPM 相当质量的样本，并且支持**语义插值**。

## 2. 动机

DDPM 的采样必须走满 T 步，因为它的反向过程是前向马尔可夫链的唯一逆。
作者提出的问题是：**前向链是不是必需的？**

关键洞察：DDPM 的 $L_{\text{simple}}$ 只用到 $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$。
而满足这个边缘分布的前向过程有**无穷多个**（马尔可夫链只是其中一个特例）。
→ 只要找到一个「边缘分布相同、但允许大步跳跃」的反向过程，就能复用原模型加速采样。

## 3. 方法

### 3.1 非马尔可夫前向过程

定义 $q_\sigma(x_{1:T}\mid x_0)$，其边缘仍为 $\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，
但 $x_{t-1}$ 可以**依赖于 $x_t$ 之外还依赖 $x_0$**：

$$q_\sigma(x_{t-1}\mid x_t,x_0)=\mathcal N\Big(\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2 I\Big)$$

### 3.2 反向采样（论文 Eq.12）

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\underbrace{\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}}_{\hat x_0\ (\text{预测的 }x_0)}
+\underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t,t)}_{\text{指向 }x_t\text{ 的方向}}
+\underbrace{\sigma_t z}_{\text{随机项}}$$

三个特例：

| $\sigma_t$ | 对应 | 性质 |
|------|------|------|
| $\sigma_t=\sqrt{\tilde\beta_t}$（后验方差） | **DDPM** | 随机、马尔可夫、需 T 步 |
| $\sigma_t=0$ | **DDIM** | **完全确定性**，对应一个 ODE，可大步跳 |
| $\sigma_t=\sqrt{(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)}\sqrt{1-\bar\alpha_t/\bar\alpha_{t-1}}$ | DDPM 的另一种写法 | — |

### 3.3 加速的本质：子序列采样

DDIM 不是「跳过一部分时间步然后马马虎虎去噪」，而是：
在原时间序列 $\{1..T\}$ 中取一个长度为 $S$ 的子序列 $\tau$（例如每隔 10 步取一个），
**只在 $\tau$ 上做反向积分**。因为前向边缘分布是闭式的，$\bar\alpha_{\tau_i}$ 直接可查，
不需要真的走完中间那些步。

**这一步为什么成立**：反向过程被写成了一个以 $\hat x_0$ 和 $\epsilon_\theta$ 为端点的插值，
步长的改变只影响离散化误差，不改变「要走到哪个分布」这个目标。

### 3.4 与 Neural ODE / 流模型的联系

$\sigma_t=0$ 时反向过程是确定性的，作者证明它在连续极限下对应一个**概率流 ODE**；
这使得 DDIM 样本拥有和 GAN / Flow 一样的「隐变量 → 样本」的确定性映射，
因而可以在隐空间做插值（DDPM 做不到，因为每步都在随机采样）。

## 4. 主要实验结论

| 设置 | 结论 |
|------|------|
| CIFAR-10, S=50 | FID ≈ 4.0 左右，优于 1000 步 DDPM 的大部分设置 |
| CIFAR-10, S=10 | 仍能生成可辨识样本，FID 明显优于同等步数的 DDPM |
| CelebA, S=20 | 画质几乎无损 |
| 一致性 | 同一个 $x_T$ 用不同步数采样得到的样本**语义一致**（高阶信息一致，细节不同） |
| 加速比 | 相对 1000 步 DDPM 提速 10~100 倍 |

**一个重要观察**：论文指出，当 DDPM 用很少的步数（比如 10 步）采样时画质崩坏，
而 DDIM 仍然可用——原因是 DDPM 的随机项在大步长下会累积误差，DDIM 没有这一项。

## 5. 我的理解与思考

1. **这篇论文是"重新审视训练目标的自由度"的典范**。
   它没有改模型、没有改损失、没有加数据，只是发现「训练目标没有用满前向链的全部结构」，
   于是把多出来的自由度用在了加速上。我很喜欢这种"从已有框架里挤出自由度"的做法。

2. **"确定性采样" 的价值不止是快**：
   - 可复现（同一 $x_T$ 必得同一结果）
   - 可做隐空间插值/编辑（DDIM inversion 这一整条工作线的起点，后面 SDEdit、Prompt-to-Prompt、Null-text Inversion 都建立在其上）
   - 与 ODE 求解器生态打通（后面 DPM-Solver、UniPC 都是在 DDIM 的 ODE 视角上做高阶求解）

3. **但确定性也有代价**：DDIM 在小步数下的样本多样性略低（随机项消失，样本集中在模式中心）。
   这也是后面「consistency model / 蒸馏」之外还要保留随机性探索的原因。

4. **一个实用的判断**：DDIM 的加速比与调度强相关。
   cosine 调度下 $\bar\alpha$ 变化更平缓，跳步的离散化误差更小——我的复现里就用了 cosine，
   并准备在 DDIM 对比实验中固定调度、只改步数，以隔离变量。

5. **启发我的想法**：既然 DDIM 说明「时间步可以被稀疏采样」，
   那么**哪些时间步值得保留**应该是一个可学习/可自适应决定的问题（见我的 idea-01）。

## 6. 复现情况

- 复现目录：[`../reproduction/02-ddim-sampling/`](../reproduction/02-ddim-sampling/)
- 复用 01 训练好的 DDPM 权重（**完全不重新训练**，这正是 DDIM 的卖点）
- 对比：DDPM(ancestral) vs DDIM，步数 {1000, 250, 100, 50, 20, 10}
- 指标：采样耗时 + 用自训练 MNIST 分类器特征计算的 FID 近似 + 分类器置信度

## 7. 遗留问题

- DDIM 的 $\eta$（随机项权重）在不同数据集/调度上的最优值如何自动确定？
- 子序列 $\tau$ 除了均匀采样，是否存在"按 SNR 等分"更优的选取策略？
