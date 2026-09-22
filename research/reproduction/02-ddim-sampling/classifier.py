"""
评估用 MNIST 分类器。

用途：为 02 实验提供两个真实指标
  1) 分类器特征空间的 FID 近似（生成集 vs 真实集的 Fréchet 距离）
  2) 生成样本的平均最大分类置信度 + 类别分布熵（类 IS 的简化版）

训练成本极低（CPU 1~2 分钟），权重缓存到 results/mnist_classifier.pt。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "01-ddpm-mnist" / "data"  # 与 01 复用同一份 MNIST


class MnistNet(nn.Module):
    def __init__(self, feat_dim: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)
        self.feat = nn.Linear(128, feat_dim)

    def forward(self, x):
        h = self.body(x)
        return self.head(h)

    def features(self, x):
        return self.feat(self.body(x))


def get_classifier(device="cpu", force_retrain=False) -> MnistNet:
    ckpt = _HERE / "results" / "mnist_classifier.pt"
    model = MnistNet().to(device)
    if ckpt.exists() and not force_retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model

    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST(DATA_DIR, train=True, download=False, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=False, transform=tf)
    tl = DataLoader(train, batch_size=128, shuffle=True)
    vl = DataLoader(test, batch_size=512)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(3):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    acc = correct / total
    print(f"[classifier] test acc = {acc:.4f}")

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    assert acc > 0.95, "分类器精度不足，先修好分类器再做生成质量评估"
    return model


if __name__ == "__main__":
    get_classifier(force_retrain=True)
