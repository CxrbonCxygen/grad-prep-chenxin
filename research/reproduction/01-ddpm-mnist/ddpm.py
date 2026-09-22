"""
DDPM (Denoising Diffusion Probabilistic Models) 核心组件 —— 从零实现。

参考论文：
    Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models". NeurIPS 2020.
    arXiv:2006.11239  官方实现 https://github.com/hojonathanho/diffusion

本文件只实现「机制」，不依赖任何第三方扩散模型库；
训练/采样脚本见 train.py / sample.py。

实现要点（与论文公式一一对应）：
    前向（加噪）  q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    训练目标      L_simple = E || eps - eps_theta(sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t) ||^2
    反向（采样）  x_{t-1} = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# 1. 噪声调度 (variance schedule)
# ----------------------------------------------------------------------------------
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """DDPM 原文使用的线性调度（论文 Section 4, T=1000, 1e-4 -> 0.02）。"""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 8e-3) -> torch.Tensor:
    """Improved DDPM (Nichol & Dhariwal, ICML 2021) 提出的 cosine 调度。

    直接对 alpha_bar 做余弦衰减，避免线性调度在高分辨率下后段噪声过强、
    导致大量采样步被浪费在「纯噪声」区域的问题。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def get_beta_schedule(name: str, timesteps: int, **kwargs) -> torch.Tensor:
    name = name.lower()
    if name == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    if name == "cosine":
        return cosine_beta_schedule(timesteps, **kwargs)
    raise ValueError(f"unknown schedule: {name}")


# ----------------------------------------------------------------------------------
# 2. 由 beta 派生的一堆系数
# ----------------------------------------------------------------------------------
class DiffusionSchedule:
    """预计算扩散过程需要的所有系数（与具体数据无关，可复用）。"""

    def __init__(self, betas: torch.Tensor, device: torch.device | str = "cpu"):
        self.timesteps = int(betas.shape[0])
        betas = betas.to(device).double()

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=betas.device, dtype=torch.float64), alpha_bar[:-1]]).float()
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).float()

        # 后验 q(x_{t-1} | x_t, x_0) 的方差 (论文 Eq.7)
        self.posterior_variance = (betas * (1.0 - self.alpha_bar_prev) / (1.0 - alpha_bar)).float()
        # 论文中 t=1 时置 0，t>1 时取 min(beta_t, posterior_variance)（Improved DDPM 的修正）
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )


# ----------------------------------------------------------------------------------
# 3. 扩散过程：前向加噪 + DDPM 反向采样
# ----------------------------------------------------------------------------------
class GaussianDiffusion:
    def __init__(self, schedule: DiffusionSchedule):
        self.s = schedule
        self.T = schedule.timesteps

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """按 batch 内每个样本的时间步 t 取系数，并 broadcast 成 x 的形状。"""
        out = arr.to(t.device)[t]
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # ---- 前向：q(x_t | x_0) 闭式采样 ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.s.sqrt_alpha_bar, t, x0.shape) * x0
            + self._extract(self.s.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )

    # ---- 训练损失：L_simple ----
    def p_losses(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    # ---- 由 eps 预测还原 x_0（论文 Eq.15 的系数写法）----
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(torch.sqrt(1.0 / self.s.alpha_bar), t, x_t.shape) * x_t
            - self._extract(torch.sqrt(1.0 / self.s.alpha_bar - 1.0), t, x_t.shape) * eps
        )

    # ---- DDPM 反向一步：q(x_{t-1} | x_t, x_0) ----
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        b = x_t.shape[0]
        t_batch = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t_batch)
        x0_hat = self.predict_x0_from_eps(x_t, t_batch, eps).clamp(-1.0, 1.0)

        model_mean = (
            self._extract(self.s.sqrt_recip_alphas, t_batch, x_t.shape)
            * (x_t - self._extract(self.s.betas, t_batch, x_t.shape) * eps
               / self._extract(self.s.sqrt_one_minus_alpha_bar, t_batch, x_t.shape))
        )
        if t == 0:
            return model_mean
        noise = torch.randn_like(x_t)
        log_var = self._extract(self.s.posterior_log_variance_clipped, t_batch, x_t.shape)
        return model_mean + torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, device="cpu", return_traj: bool = False, traj_stride: int = 0):
        """完整 DDPM 采样（ancestral sampling，T 步）。"""
        model.eval()
        x = torch.randn(shape, device=device)
        traj = [x.detach().cpu().clone()] if return_traj else None
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if return_traj and (traj_stride > 0) and (t % traj_stride == 0):
                traj.append(x.detach().cpu().clone())
        if return_traj:
            traj.append(x.detach().cpu().clone())
            return x, traj
        return x


# ----------------------------------------------------------------------------------
# 4. 主干网络：小型 UNet（epsilon 预测）
# ----------------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Transformer 风格的 sinusoidal 位置编码 + 两层 MLP（DDPM 附录）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ResBlock(nn.Module):
    """卷积残差块，时间步通过 scale/shift 注入（FiLM 式的等价写法）。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """为 CPU 小规模实验裁剪过的 UNet：32x32 输入，两层下采样。"""

    def __init__(self, in_ch: int = 1, base_ch: int = 32, ch_mults=(1, 2), time_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList()
        cur = base_ch
        for ch in chs:
            self.downs.append(nn.ModuleList([ResBlock(cur, ch, time_dim), nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
            cur = ch

        self.mid = ResBlock(cur, cur, time_dim)

        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(cur, ch, 4, stride=2, padding=1),
                    ResBlock(ch * 2, ch, time_dim),
                ])
            )
            cur = ch

        self.out = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x = self.init_conv(x)
        skips = [x]
        for res, down in self.downs:
            x = res(x, t_emb)
            skips.append(x)
            x = down(x)
        x = self.mid(x, t_emb)
        for up, res in self.ups:
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = res(x, t_emb)
        return self.out(x)
