"""
训练一个无条件 DDPM（epsilon 预测）在 MNIST 上生成手写数字。

环境：仅 CPU（本机无 NVIDIA GPU），因此默认使用「小规模 + 小模型」配置，
      目标是完整跑通论文的训练与采样流程，而不是追求 SOTA 画质。

运行：
    python train.py                       # 默认配置（约 20k 张图，CPU 上数十分钟）
    python train.py --epochs 30 --subset 20000 --schedule cosine

产出：
    results/train_log.csv      每 epoch 的平均 loss
    results/loss_curve.png     loss 曲线
    results/samples_epoch*.png 训练过程中的采样网格
    results/ddpm_mnist.pt      模型权重（供 02-ddim-sampling 复用）
    results/metrics.json       关键配置与最终指标
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ddpm import UNet, GaussianDiffusion, DiffusionSchedule, get_beta_schedule


def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM on MNIST (CPU-friendly)")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./results")
    p.add_argument("--subset", type=int, default=20000, help="使用前 N 张训练图（控制 CPU 训练成本）")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--sample-every", type=int, default=3, help="每多少个 epoch 采样一次并保存图片")
    p.add_argument("--sample-steps", type=int, default=200, help="训练中快速预览用的采样步数")
    p.add_argument("--num-sample", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="torch CPU 线程数，0 表示自动")
    return p.parse_args()


def save_grid(x: torch.Tensor, path: Path, nrow: int = 4, title: str = ""):
    """把 [-1,1] 的张量画成网格图。"""
    x = (x.clamp(-1, 1) + 1) / 2  # -> [0,1]
    n = x.shape[0]
    nrow = min(nrow, n)
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(1.6 * nrow, 1.6 * ncol))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])
    ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    if args.subset > 0:
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[: args.subset]
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    betas = get_beta_schedule(args.schedule, args.timesteps)
    diffusion = GaussianDiffusion(DiffusionSchedule(betas, device))
    model = UNet(in_ch=1, base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[config] device={device}  params={n_params/1e6:.2f}M  "
          f"samples={len(ds)}  T={args.timesteps}  schedule={args.schedule}")

    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for x0, _ in loader:
            x0 = x0.to(device)
            b = x0.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device)
            loss = diffusion.p_losses(model, x0, t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * b
            nb += b
        avg = total / nb
        history.append({"epoch": epoch, "loss": avg})
        print(f"[epoch {epoch:>3}/{args.epochs}] loss={avg:.4f}  elapsed={time.time()-t_start:.0f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                # 用「前 sample_steps 步」的等间距子序列做快速预览（等价于 DDIM 式跳步，仅为省时间）
                x = torch.randn(args.num_sample, 1, args.image_size, args.image_size, device=device)
                step_seq = torch.linspace(args.timesteps - 1, 0, args.sample_steps).round().long().tolist()
                for i, t in enumerate(step_seq):
                    x = diffusion.p_sample(model, x, int(t))
            save_grid(x, out_dir / f"samples_epoch{epoch:03d}.png",
                      title=f"epoch {epoch}  (loss={avg:.4f}, {args.sample_steps} steps)")

    # ---- 保存产物 ----
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "betas": betas.cpu()},
               out_dir / "ddpm_mnist.pt")

    with open(out_dir / "train_log.csv", "w", encoding="utf-8") as f:
        f.write("epoch,loss\n")
        for h in history:
            f.write(f"{h['epoch']},{h['loss']:.6f}\n")

    plt.figure(figsize=(6, 4))
    plt.plot([h["epoch"] for h in history], [h["loss"] for h in history], marker="o", ms=3, color="#2b6cb0")
    plt.xlabel("epoch"); plt.ylabel("L_simple (MSE on eps)")
    plt.title(f"DDPM training loss on MNIST (T={args.timesteps}, {args.schedule} schedule)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=140)
    plt.close()

    metrics = {
        "paper": "Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020",
        "device": str(device),
        "model_params": n_params,
        "timesteps": args.timesteps,
        "schedule": args.schedule,
        "epochs": args.epochs,
        "train_samples": len(ds),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": history[-1]["loss"],
        "best_loss": min(h["loss"] for h in history),
        "wall_clock_sec": round(time.time() - t_start, 1),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[done] ->", out_dir.resolve())


if __name__ == "__main__":
    main()
