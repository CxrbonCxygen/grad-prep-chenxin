# 论文出处与链接

本目录**不存放论文 PDF**（遵守出版方版权），仅记录出处与官方链接。
所有链接均为作者/会议官方公开页面，最后核对时间：2026-09。

| # | 论文 | 作者 | 会议 | arXiv | 官方代码 |
|---|------|------|------|-------|---------|
| 01 | Denoising Diffusion Probabilistic Models | Ho, Jain, Abbeel | NeurIPS 2020 | https://arxiv.org/abs/2006.11239 | https://github.com/hojonathanho/diffusion |
| 02 | Denoising Diffusion Implicit Models | Song, Meng, Ermon | ICLR 2021 | https://arxiv.org/abs/2010.02502 | https://github.com/ermongroup/ddim |
| 03 | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach, Blattmann, Lorenz, Esser, Ommer | CVPR 2022 | https://arxiv.org/abs/2112.10752 | https://github.com/CompVis/latent-diffusion |
| 04 | Scalable Diffusion Models with Transformers (DiT) | Peebles, Xie | ICCV 2023 | https://arxiv.org/abs/2212.09748 | https://github.com/facebookresearch/DiT |

## 延伸阅读（笔记中引用，未精读）

| 主题 | 文献 | 说明 |
|------|------|------|
| 改进版 DDPM | Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021, arXiv:2102.09672 | cosine 调度、可学习方差 —— 我在复现中直接使用了它的 cosine 调度 |
| Score-based 视角 | Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021, arXiv:2011.13456 | 把 DDPM/DDIM 统一到 SDE/ODE 框架 |
| Classifier-Free Guidance | Ho & Salimans, 2021, arXiv:2207.12598 | 条件生成的标准做法 |
| 高阶求解器 | Lu et al., *DPM-Solver*, NeurIPS 2022, arXiv:2206.00927 | 在 DDIM 的 ODE 视角上做高阶积分 |
| 一致性模型 | Song et al., *Consistency Models*, ICML 2023, arXiv:2303.01469 | 一步/少步生成的另一条路线 |
