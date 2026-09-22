"""
DDIM 采样器（Song et al., ICLR 2021, arXiv:2010.02502）。

DDIM 论文的关键结论：DDPM 的训练目标只依赖边缘分布 q(x_t|x_0)，
因此可以用一条非马尔可夫、可确定性的反向过程来复用同一个模型，从而在子序列上大步跳。

DDPM 与 DDIM 在这里被写成同一个公式的两个特例（论文 Eq.12 + Eq.16）：

    x_s = √ᾱ_s · x̂0 + √(1 − ᾱ_s − σ²) · ε_θ + σ · z
    x̂0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
    σ   = η · √((1−ᾱ_s)/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_s)

    η = 0  → DDIM（确定性，对应 ODE，可大步跳）
    η = 1  → DDPM 的随机 ancestral 采样（σ² 等于后验方差）
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 复用 01 的模型与调度实现（不复制代码）
_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent / "01-ddpm-mnist"))
from ddpm import DiffusionSchedule, UNet  # noqa: E402


def timestep_sequence(T: int, S: int, spacing: str = "uniform") -> list[int]:
    """把 {1..T} 压成长度 S 的子序列（末尾补 0 表示最后一步输出 x_0）。

    当前实现为均匀间距（论文 Section 4 的默认做法）；
    按 log-SNR 等距等其他选取策略可作为后续扩展。
    """
    if spacing != "uniform":
        raise ValueError(f"unsupported spacing: {spacing}")
    ts = [int(round(T - (T - 1) * k / (S - 1))) for k in range(S)]
    ts.append(0)
    # 去重（S > T 时可能出现重复）
    out = [ts[0]]
    for v in ts[1:]:
        if v != out[-1]:
            out.append(v)
    return out


@torch.no_grad()
def sample(model: UNet, sch: DiffusionSchedule, n: int, img_size: int,
           steps: int, eta: float, device="cpu", seed: int = 0,
           x_init: torch.Tensor | None = None,
           keep_first: bool = False,
           clip_denoised: bool = True) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """用统一采样式从 x_T 生成 n 张图。

    eta=0 → DDIM；eta=1 → DDPM。
    keep_first=True 时保留起点噪声（用于跨采样器/跨步数的同源对比）。
    clip_denoised=True 时对每步的 x̂0 做静态钳制（[-1,1]）。
    这一点对欠训练的小模型至关重要：实测不钳制时确定性 DDIM 的 ODE 轨迹会发散
    （|x̂0| 可达数千），随机采样因每步重新注入噪声而相对稳健。
    静态钳制是 OpenAI improved-diffusion 等参考实现的标准做法（clip_denoised）。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if x_init is None:
        x = torch.randn(n, 1, img_size, img_size, generator=g).to(device)
    else:
        x = x_init.to(device)
    x0_saved = x.clone() if keep_first else None

    T = sch.timesteps
    ts = timestep_sequence(T, steps)

    def coef(name: str, t: int) -> torch.Tensor:
        arr = getattr(sch, name)
        idx = max(0, min(T, t) - 1) if t > 0 else None
        return torch.as_tensor(1.0 if idx is None else float(arr[idx]), device=device)

    for t, s in zip(ts[:-1], ts[1:]):
        t_batch = torch.full((n,), t - 1, device=device, dtype=torch.long)  # ddpm.py 里 t 是 0-based 索引
        eps = model(x, t_batch)
        ab_t = coef("alpha_bar", t)
        ab_s = coef("alpha_bar", s)
        x0_hat = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        if clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        if eta > 0 and s > 0:
            sigma = eta * torch.sqrt((1 - ab_s) / (1 - ab_t)) * torch.sqrt(torch.clamp(1 - ab_t / ab_s, min=0.0))
        else:
            sigma = torch.tensor(0.0, device=device)

        dir_coef = torch.sqrt(torch.clamp(1 - ab_s - sigma**2, min=0.0))
        x = torch.sqrt(ab_s) * x0_hat + dir_coef * eps
        if float(sigma) > 0:
            x = x + sigma * torch.randn(x.shape, generator=g).to(device)
    return x.clamp(-1, 1), x0_saved


def load_model(ckpt_path: str | Path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    model = UNet(in_ch=1, base_ch=args["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sch = DiffusionSchedule(ckpt["betas"], device)
    return model, sch, args
