"""
对比实验：同一个训练好的 DDPM 权重，分别用 DDPM(ancestral) 与 DDIM 采样，
扫描不同采样步数，量化「质量 - 耗时」的权衡。

  质量：① 分类器特征空间 FID 近似（自训练 MNIST 分类器，见 classifier.py）
        ② 平均最大分类置信度
        ③ 类别分布熵（多样性）
  耗时：端到端采样墙钟时间

关键设计：同一个采样器在所有步数下**共用同一初始噪声 x_T**（种子固定），
以排除随机性、只考察「步数」这一个变量 —— 这正是 DDIM 论文中一致性实验的做法。

运行（先完成 01 的训练）：
    python compare.py
产出见 ./results/
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from classifier import MnistNet, get_classifier
from ddim import load_model, sample

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / "01-ddpm-mnist" / "results" / "ddpm_mnist.pt"
OUT = HERE / "results"
STEPS = [1000, 250, 100, 50, 20, 10]


def save_grid(x: torch.Tensor, path: Path, nrow: int, title: str = ""):
    x = (x.clamp(-1, 1) + 1) / 2
    n = x.shape[0]
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(1.5 * nrow, 1.5 * ncol))
    axes = np.atleast_1d(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """FID 的 Fréchet 距离（用自训练分类器的特征空间近似，样本量有限时仅供相对比较）。"""
    from scipy import linalg

    mu1, mu2 = feats_a.mean(0), feats_b.mean(0)
    c1 = np.cov(feats_a, rowvar=False) + 1e-6 * np.eye(feats_a.shape[1])
    c2 = np.cov(feats_b, rowvar=False) + 1e-6 * np.eye(feats_b.shape[1])
    diff = mu1 - mu2
    try:  # scipy>=1.18 移除了 disp 参数
        covmean = linalg.sqrtm(c1.dot(c2))
    except TypeError:
        covmean, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(c1) + np.trace(c2) - 2 * np.trace(covmean))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    n = 128
    img = 32

    model, sch, args = load_model(CKPT, device)
    print(f"[load] ckpt={CKPT.name}  T={sch.timesteps}  schedule={args['schedule']}  epochs={args['epochs']}")

    # ---- 分类器 + 真实参照特征 ----
    clf = get_classifier(device)
    tf = transforms.Compose([transforms.Resize(img), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test = datasets.MNIST(HERE.parent / "01-ddpm-mnist" / "data", train=False, download=True, transform=tf)
    real = torch.stack([test[i][0] for i in range(2048)])
    with torch.no_grad():
        real_feats = clf.features(real.to(device)).cpu().numpy()

    def metrics(x: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = clf(x.to(device))
            feats = clf.features(x.to(device)).cpu().numpy()
        prob = torch.softmax(logits, dim=1)
        conf = prob.max(1).values.mean().item()
        hist = prob.sum(0)
        hist = hist / hist.sum()
        entropy = -(hist * torch.log(hist + 1e-9)).sum().item()
        return {"fid": frechet_distance(feats, real_feats), "conf": conf, "entropy": entropy}

    # ---- 主循环 ----
    rows = []
    grids = {}
    seeds = {"ddim": 7, "ddpm": 13}
    for name, eta in [("ddim", 0.0), ("ddpm", 1.0)]:
        x_init = None  # 同一采样器在所有步数下共用同一 x_T
        for S in STEPS:
            t0 = time.time()
            x, x_init = sample(model, sch, n, img, S, eta, device,
                               seed=seeds[name], x_init=x_init, keep_first=True)
            dt = time.time() - t0
            m = metrics(x)
            row = {"sampler": name, "steps": S, "time_s": round(dt, 2),
                   "sec_per_image": round(dt / n, 3), **{k: round(v, 4) for k, v in m.items()}}
            rows.append(row)
            print(f"[{name:>4} S={S:>4}] fid≈{m['fid']:8.2f}  conf={m['conf']:.3f}  "
                  f"entropy={m['entropy']:.3f}  time={dt:6.1f}s", flush=True)
            if S in (10, 50, 1000):
                grids[(name, S)] = x.cpu()

    # ---- 表格 ----
    with open(OUT / "ab_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 曲线 ----
    for key, ylabel, fname in [("fid", "FID (classifier-feature, lower better)", "quality_vs_steps.png"),
                               ("time_s", "sampling time (s, 128 images)", "time_vs_steps.png")]:
        plt.figure(figsize=(6, 4))
        for name, color in [("ddim", "#2b6cb0"), ("ddpm", "#c05621")]:
            r = [row for row in rows if row["sampler"] == name]
            plt.plot([x["steps"] for x in r], [x[key] for x in r],
                     marker="o", label=f"{name.upper()} (η=0)" if name == "ddim" else "DDPM (η=1)", color=color)
        plt.xscale("log"); plt.xlabel("sampling steps S"); plt.ylabel(ylabel)
        plt.gca().invert_xaxis(); plt.grid(alpha=0.3); plt.legend()
        plt.title("Quality vs sampling steps" if key == "fid" else "Wall-clock time vs sampling steps")
        plt.tight_layout(); plt.savefig(OUT / fname, dpi=140); plt.close()

    # ---- 同一 x_T 下的步数一致性（DDIM）----
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.6))
    for r, S in enumerate([1000, 50, 10]):
        for c in range(8):
            axes[r, c].imshow(((grids[("ddim", S)][c].clamp(-1, 1) + 1) / 2)[0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(f"S={S}", fontsize=11)
    for r, S in enumerate([1000, 50, 10]):
        axes[r, 0].text(-0.35, 0.5, f"DDIM\nS={S}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=11)
    fig.suptitle("Same initial noise x_T, DDIM with different steps (low-step samples are smoother)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ddim_consistency.png", dpi=140)
    plt.close(fig)

    # ---- S=10 时 DDPM vs DDIM 的画质差距 ----
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    for r, name in enumerate(["ddpm", "ddim"]):
        for c in range(8):
            axes[r, c].imshow(((grids[(name, 10)][c].clamp(-1, 1) + 1) / 2)[0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.35, 0.5, f"{'DDPM' if name=='ddpm' else 'DDIM'}\nS=10",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=11)
    fig.suptitle("Same initial noise x_T, S=10: both samplers still produce digits "
                 "(static clipping stabilizes DDPM's large jumps)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ddpm_vs_ddim_S10.png", dpi=140)
    plt.close(fig)

    summary = {
        "checkpoint": str(CKPT),
        "n_images_per_config": n,
        "real_ref_images": 2048,
        "note": "FID 为自训练 MNIST 分类器特征空间的 Fréchet 距离近似，样本量有限，仅供相对比较。",
        "rows": rows,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[done] ->", OUT.resolve())


if __name__ == "__main__":
    main()
