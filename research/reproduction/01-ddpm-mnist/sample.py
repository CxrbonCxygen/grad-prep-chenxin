"""
用训练好的 DDPM 权重做最终采样，输出：
    results/final_samples.png   64 张生成样本网格（完整 T 步 ancestral sampling）
    results/trajectory.png      单张样本从 x_T 到 x_0 的去噪轨迹
    results/sample_metrics.json 采样耗时等记录

运行：
    python sample.py                       # 默认读取 results/ddpm_mnist.pt
    python sample.py --ckpt results/ddpm_mnist.pt --num-sample 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ddpm import UNet, GaussianDiffusion, DiffusionSchedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./results/ddpm_mnist.pt")
    p.add_argument("--out-dir", type=str, default="./results")
    p.add_argument("--num-sample", type=int, default=64)
    p.add_argument("--traj-stride", type=int, default=100, help="轨迹图每隔多少步记录一帧")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(a.ckpt, map_location=device)
    args = ckpt["args"]
    betas = ckpt["betas"]
    img = args["image_size"]

    model = UNet(in_ch=1, base_ch=args["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(DiffusionSchedule(betas, device))

    t0 = time.time()
    with torch.no_grad():
        x, traj = diffusion.sample(model, (a.num_sample, 1, img, img), device=device,
                                   return_traj=True, traj_stride=a.traj_stride)
    dt = time.time() - t0

    # ---- 样本网格 ----
    g = ((x.clamp(-1, 1) + 1) / 2).cpu()
    nrow = 8
    ncol = (a.num_sample + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(1.4 * nrow, 1.4 * ncol))
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < a.num_sample:
            ax.imshow(g[i, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(f"DDPM samples (T={args['timesteps']}, {args['schedule']} schedule, "
                 f"{args['epochs']} epochs on MNIST)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "final_samples.png", dpi=150)
    plt.close(fig)

    # ---- 去噪轨迹 ----
    frames = traj[:: max(1, len(traj) // 12)]
    frames = frames + [x.cpu()]
    fig, axes = plt.subplots(1, len(frames), figsize=(1.5 * len(frames), 1.8))
    for i, (ax, fr) in enumerate(zip(axes, frames)):
        ax.imshow(((fr[0].clamp(-1, 1) + 1) / 2)[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t≈{int(round((1 - i / (len(frames) - 1)) * args['timesteps']))}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Reverse process: x_T (pure noise) -> x_0", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=150)
    plt.close(fig)

    json.dump({
        "num_samples": a.num_sample,
        "timesteps": args["timesteps"],
        "total_sec": round(dt, 2),
        "sec_per_sample": round(dt / a.num_sample, 3),
        "device": str(device),
    }, open(out_dir / "sample_metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[done] {a.num_sample} samples in {dt:.1f}s -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
