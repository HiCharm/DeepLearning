"""
多层感知机（MLP）模块。

支持可变 hidden_sizes，可选 dropout，适用于分类和回归任务。
"""
from __future__ import annotations

from torch import nn
import torch


class MLP(nn.Module):
    """前馈网络：Linear -> ReLU -> [Dropout] -> ... -> Linear。"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.net(x)

