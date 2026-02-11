"""
评估指标模块。

分类：accuracy；回归：mse、mae。
"""
from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    分类准确率。logits: [N, C]，targets: [N]。
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """回归任务的均方误差。"""
    return torch.mean((preds - targets) ** 2).item()


@torch.no_grad()
def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """回归任务的平均绝对误差。"""
    return torch.mean(torch.abs(preds - targets)).item()

